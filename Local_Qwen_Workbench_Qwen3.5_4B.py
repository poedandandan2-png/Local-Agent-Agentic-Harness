"""
Agent harness v2 for a local Qwen3 model.  Standard library only.

What's in here
  * Native tool calling (Ollama `tools` / OpenAI `tools`) with a JSON-text fallback
  * Streaming output, parallel tool calls, Ctrl+C interrupt + /resume
  * Tools: web search + fetch, sandboxed file tools, Python/shell execution,
    date/time, calculator, remember/recall, and any MCP server (mcp.json)
  * Confirmation gates for anything that writes, deletes or runs code
  * Plan -> execute -> verify -> re-plan -> reflect agent
  * Memory: rolling summary, saved facts, semantic recall (local embeddings)
  * Background jobs, model routing (fast / main / thinking models)
  * JSONL tracing and an eval runner (--eval), self-contained

Backends (LLM_BACKEND):
  ollama  (default) -> http://127.0.0.1:11434      e.g. `ollama pull qwen3.5:4b`
  openai            -> any OpenAI-compatible server (llama.cpp, vLLM, LM Studio)

Env vars (all optional)
  QWEN_MODEL (qwen3.5:4b)   QWEN_FAST_MODEL   QWEN_THINK_MODEL   EMBED_MODEL (nomic-embed-text)
  LLM_BACKEND  LLM_BASE_URL  OLLAMA_HOST  QWEN_NUM_CTX (16384)  NATIVE_TOOLS (1)  QWEN_THINK (1)  LOCAL_TIMEZONE (America/Denver)
  AGENT_WORKSPACE (./workspace)  SEARXNG_URL  MCP_CONFIG (./mcp.json)  MAX_STEPS (10)
  PLAN_VERIFY (1)  PLAN_REFLECT (1)  AGENT_DISABLE_TOOLS (e.g. RunShell,DeleteFile)
"""
from __future__ import annotations

import ast
import asyncio
import ipaddress
import json
import logging
import math
import operator
import os
import re
import shutil
import signal
import sqlite3
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE", str(BASE_DIR / "workspace")))
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", str(BASE_DIR / "memory")))
TRACE_DIR = Path(os.getenv("TRACE_DIR", str(BASE_DIR / "traces")))


# --------------------------------------------------------------------------- #
# Enums / dataclasses / exceptions
# --------------------------------------------------------------------------- #
class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class AgentContext:
    task_id: str
    session_id: str
    user_input: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class AgentResponse:
    task_id: str
    status: TaskStatus
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Cancelled(Exception):
    """The user interrupted while the model was generating."""


class TaskInterrupted(Exception):
    """Raised by an agent at a safe point; `snapshot` lets /resume continue."""

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None):
        super().__init__("interrupted")
        self.snapshot = snapshot


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _brief(obj: Any, n: int = 200) -> str:
    """Compact one-line rendering of any object, truncated to n chars."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def system_clock() -> Dict[str, Any]:
    """Return current time in the configured local zone (default: Colorado/Mountain)."""
    zone_name = os.getenv("LOCAL_TIMEZONE", "America/Denver").strip() or "America/Denver"
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone_name = "UTC"
        zone = timezone.utc
    now = datetime.now(zone)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "year": now.year,
        "timezone": now.tzname(),
        "timezone_name": zone_name,
        "utc_offset": now.strftime("%z"),
        "iso": now.isoformat(timespec="seconds"),
        "unix": int(time.time()),
    }


def clock_line() -> str:
    c = system_clock()
    return (f"CURRENT LOCAL DATE/TIME: {c['weekday']}, {c['date']} {c['time']} "
            f"{c['timezone']} (IANA zone {c['timezone_name']}, UTC offset {c['utc_offset']}). "
            "Use this supplied timestamp for current-time questions; do not guess from model knowledge.")


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles ``` fences/prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _new_call_id() -> str:
    return "call_" + uuid4().hex[:8]


def _call_from_obj(obj: Any, known: set) -> Optional[Dict[str, Any]]:
    """Normalise the many shapes models use for a tool call into {name, arguments}."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # {"function": {"name":..,"arguments":..}}
        obj = obj["function"]
    name = obj.get("tool") or obj.get("name") or obj.get("tool_name") or obj.get("function")
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or obj.get("input") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if isinstance(name, str) and (not known or name in known):
        return {"id": _new_call_id(), "name": name, "arguments": args if isinstance(args, dict) else {}}
    return None


_TOOL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_FN_RE = re.compile(r"<function=([\w\-.]+)>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([\w\-.]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_calls_from_text(text: str, known: set) -> List[Dict[str, Any]]:
    """Rescue tool calls the model wrote as text instead of using the tool API."""
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_TAG_RE.finditer(text):
        body = m.group(1).strip()
        fn = _XML_FN_RE.search(body)
        if fn:  # Qwen3-Coder's XML flavour
            args = {k: v for k, v in _XML_PARAM_RE.findall(fn.group(2))}
            if not known or fn.group(1) in known:
                calls.append({"id": _new_call_id(), "name": fn.group(1), "arguments": args})
            continue
        try:
            c = _call_from_obj(json.loads(body), known)
        except ValueError:
            c = None
        if c:
            calls.append(c)
    if not calls:
        try:
            c = _call_from_obj(extract_json(text), known)
            if c:
                calls.append(c)
        except ValueError:
            pass
    return calls


class Tracer:
    """Append-only JSONL log of everything the agents do (for debugging and evals)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def log(self, event: str, **data: Any) -> None:
        if not self.path:
            return
        rec = {"ts": round(time.time(), 3), "event": event, **data}
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Local LLM client (Qwen3): native tools, streaming, embeddings, model roles
# --------------------------------------------------------------------------- #
class LocalQwenClient:
    """Async client for a locally hosted Qwen3 model (Ollama or OpenAI-compatible)."""

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        backend: str = None,
        temperature: float = 0.6,
        think: bool = None,
        timeout: float = 300.0,
        num_ctx: int = None,
        fast_model: str = None,
        think_model: str = None,
        embed_model: str = None,
    ):
        self.model = model or os.getenv("QWEN_MODEL", "qwen3.5:4b")
        self.fast_model = fast_model or os.getenv("QWEN_FAST_MODEL") or None
        self.think_model = think_model or os.getenv("QWEN_THINK_MODEL") or None
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.backend = (backend or os.getenv("LLM_BACKEND", "ollama")).lower()
        if self.backend == "ollama":
            host = os.getenv("OLLAMA_HOST", "").strip()
            if host and not host.startswith("http"):
                host = "http://" + host
            host = host.replace("0.0.0.0", "127.0.0.1")
            default_url = host or "http://127.0.0.1:11434"
        else:
            default_url = "http://127.0.0.1:8000"
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", default_url)).rstrip("/")
        self.temperature = temperature
        self.think = (os.getenv("QWEN_THINK", "1").strip().lower() not in ("0", "false", "off", "no")) if think is None else bool(think)
        self.timeout = timeout
        self.num_ctx = num_ctx or int(os.getenv("QWEN_NUM_CTX", "16384"))
        # Keep the model resident between requests. This avoids repeated cold-load
        # delays without reducing context, output quality, or available features.
        # Set OLLAMA_KEEP_ALIVE=0 to restore Ollama's normal unload behavior.
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.native_tools = os.getenv("NATIVE_TOOLS", "1") != "0"
        self._probed = False
        self._embed_ok: Optional[bool] = None

    # ----- model roles ----------------------------------------------------- #
    def model_for(self, role: str) -> str:
        """role: 'main' (default), 'fast' (routing/verify/summaries), 'think' (planning)."""
        return {"fast": self.fast_model, "think": self.think_model}.get(role) or self.model

    def _available_models(self) -> List[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as r:
                return [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception as e:  # diagnostics only
            return [f"<could not list models: {e}>"]

    # ----- wire format ----------------------------------------------------- #
    def _wire_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Internal messages -> what the backend expects (tool calls / results differ)."""
        out = []
        for m in messages:
            m = {k: v for k, v in m.items() if k != "timestamp"}
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                if self.backend == "ollama":
                    m["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in m["tool_calls"]]
                else:
                    m["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in m["tool_calls"]]
            elif role == "tool":
                if self.backend == "ollama":
                    m = {"role": "tool", "content": m.get("content", ""), "tool_name": m.get("name", "")}
                else:
                    m = {"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m.get("content", "")}
            out.append(m)
        return out

    @staticmethod
    def _tool_payload(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools]

    @staticmethod
    def _filter_think(state: Dict[str, bool], piece: str) -> str:
        if "<think>" in piece:
            state["in_think"] = True
        visible = "" if state.get("in_think") else piece
        if "</think>" in piece:
            state["in_think"] = False
            visible = piece.split("</think>", 1)[1]
        return visible

    @staticmethod
    def _norm_args(a: Any) -> Dict[str, Any]:
        if isinstance(a, dict):
            return a
        if isinstance(a, str) and a.strip():
            try:
                v = json.loads(a)
                return v if isinstance(v, dict) else {}
            except ValueError:
                return {}
        return {}

    # ----- streaming readers ----------------------------------------------- #
    def _read_ollama(self, resp, cancel, on_token):
        content, calls, usage, streamed, state = "", [], {}, False, {}
        for raw in resp:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            raw = raw.strip()
            if not raw:
                continue
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise RuntimeError(f"Model server error: {chunk['error']}")
            msg = chunk.get("message") or {}
            piece = msg.get("content") or ""
            if piece:
                content += piece
                vis = self._filter_think(state, piece)
                if on_token and vis:
                    try:
                        on_token(vis)
                        streamed = True
                    except Exception:
                        pass
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append({"id": tc.get("id") or _new_call_id(), "name": fn.get("name", ""),
                              "arguments": self._norm_args(fn.get("arguments"))})
            if chunk.get("done"):
                usage = {"prompt_tokens": chunk.get("prompt_eval_count"), "completion_tokens": chunk.get("eval_count")}
        return content, calls, usage, streamed

    def _read_openai(self, resp, cancel, on_token):
        content, slots, usage, streamed, state = "", {}, {}, False, {}
        for raw in resp:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = {"prompt_tokens": chunk["usage"].get("prompt_tokens"),
                         "completion_tokens": chunk["usage"].get("completion_tokens")}
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                if piece:
                    content += piece
                    vis = self._filter_think(state, piece)
                    if on_token and vis:
                        try:
                            on_token(vis)
                            streamed = True
                        except Exception:
                            pass
                for tc in d.get("tool_calls") or []:
                    slot = slots.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        calls = [{"id": s["id"] or _new_call_id(), "name": s["name"], "arguments": self._norm_args(s["arguments"])}
                 for _, s in sorted(slots.items())]
        return content, calls, usage, streamed

    # ----- core request ---------------------------------------------------- #
    def _chat_sync(self, messages, tools=None, cancel=None, on_token=None, role="main", json_mode=False):
        model = self.model_for(role)
        use_tools = bool(tools) and self.native_tools
        wire = self._wire_messages(messages)
        if self.backend == "ollama":
            url = f"{self.base_url}/api/chat"
            payload: Dict[str, Any] = {
                "model": model, "messages": wire, "stream": True,
                "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            }
            # Explicitly disable Qwen's hidden reasoning unless requested. This
            # prevents an installation's model defaults from silently adding a
            # long thinking phase to ordinary responses.
            payload["think"] = bool(self.think)
            if use_tools:
                payload["tools"] = self._tool_payload(tools)
            elif json_mode:
                payload["format"] = "json"  # only for calls that must return JSON
            if self.think:
                payload["think"] = True
        else:
            url = f"{self.base_url}/v1/chat/completions"
            if not self.think and wire and wire[-1].get("role") == "user" and isinstance(wire[-1].get("content"), str):
                wire[-1]["content"] += " /no_think"  # Qwen3 soft switch
            payload = {"model": model, "messages": wire, "temperature": self.temperature, "stream": True}
            if use_tools:
                payload["tools"] = self._tool_payload(tools)

        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                reader = self._read_ollama if self.backend == "ollama" else self._read_openai
                content, calls, usage, streamed = reader(resp, cancel, on_token)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 404:
                raise RuntimeError(
                    f"Server at {self.base_url} says model '{model}' was not found. "
                    f"Models that server can see: {self._available_models()}. Server said: {body}") from e
            if e.code == 400 and use_tools and "tool" in body.lower():
                raise RuntimeError(f"Model '{model}' rejected native tool calling ({body}). "
                                   f"Set NATIVE_TOOLS=0 to use the JSON-text fallback.") from e
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not reach local model server at {self.base_url} ({e}). "
                                  f"Is it running (`ollama serve`)?") from e

        return {"content": THINK_RE.sub("", content).strip(), "tool_calls": [c for c in calls if c["name"]],
                "usage": usage, "streamed": streamed, "model": model, "elapsed": time.time() - t0}

    # ----- public async API ------------------------------------------------ #
    async def chat_tools(self, messages, tools=None, cancel=None, on_token=None, role="main", json_mode=False):
        """Full reply dict: content, tool_calls, usage, streamed, model, elapsed."""
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return await asyncio.to_thread(self._chat_sync, messages, tools, cancel, on_token, role, json_mode)

    async def chat(self, messages, role="main", cancel=None, on_token=None, json_mode=False) -> str:
        r = await self.chat_tools(messages, None, cancel, on_token, role, json_mode)
        return r["content"]

    async def probe_native_tools(self) -> bool:
        """Once per client: does this Ollama model support the `tools` field?"""
        if self._probed:
            return self.native_tools
        self._probed = True
        if not self.native_tools or self.backend != "ollama":
            return self.native_tools
        try:
            await asyncio.to_thread(self._probe_sync)
        except _NoNativeTools:
            logger.warning(f"{self.model} has no native tool support; using the JSON-text protocol")
            self.native_tools = False
        except Exception:
            pass  # connection problems surface on the real call
        return self.native_tools

    def _probe_sync(self) -> None:
        payload = {"model": self.model, "stream": False, "messages": [{"role": "user", "content": "hi"}],
                   "options": {"num_predict": 1},
                   "tools": [{"type": "function", "function": {"name": "noop", "description": "noop",
                                                                 "parameters": {"type": "object", "properties": {}}}}]}
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read()
        except urllib.error.HTTPError as e:
            if e.code == 400 and "tool" in e.read().decode(errors="replace").lower():
                raise _NoNativeTools() from e
            raise

    # ----- embeddings ------------------------------------------------------ #
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "ollama":
            url, payload = f"{self.base_url}/api/embed", {"model": self.embed_model, "input": texts}
        else:
            url, payload = f"{self.base_url}/v1/embeddings", {"model": self.embed_model, "input": texts}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["embeddings"] if self.backend == "ollama" else [d["embedding"] for d in data["data"]]

    async def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embeddings, or None if unavailable (callers fall back to keyword search)."""
        if self._embed_ok is False or not texts:
            return None
        try:
            vecs = await asyncio.to_thread(self._embed_sync, texts)
            self._embed_ok = True
            return vecs
        except Exception as e:
            if self._embed_ok is None:
                logger.warning(f"Embeddings unavailable ({e}); memory search falls back to keywords. "
                               f"For semantic recall run: ollama pull {self.embed_model}")
            self._embed_ok = False
            return None


class _NoNativeTools(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tools: base class, schema validation
# --------------------------------------------------------------------------- #
class Tool(ABC):
    dangerous = False  # True -> the user must approve each call

    def __init__(self, name: str, description: str, parameters: Dict[str, str] = None,
                 schema: Dict[str, Any] = None):
        self.name = name
        self.description = description
        if schema is None:  # legacy style: {param: description}, all strings, all required
            parameters = parameters or {}
            schema = {"type": "object",
                      "properties": {k: {"type": "string", "description": v} for k, v in parameters.items()},
                      "required": list(parameters)}
        self.schema = schema
        self.parameters = {k: v.get("description", "") for k, v in schema.get("properties", {}).items()}

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        ...

    def needs_confirmation(self, args: Dict[str, Any]) -> bool:
        return self.dangerous

    def describe_call(self, args: Dict[str, Any]) -> str:
        return f"{self.name}({_brief(args, 400)})"

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.schema}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters,
                "schema": self.schema, "dangerous": self.dangerous, "type": self.__class__.__name__}


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Check + gently coerce args against a (small subset of) JSON schema."""
    props = schema.get("properties", {})
    errors: List[str] = []
    clean: Dict[str, Any] = {}
    for k in schema.get("required", []):
        if args.get(k) is None:
            errors.append(f"missing required argument '{k}'")
    for k, v in args.items():
        spec = props.get(k)
        if spec is None:
            errors.append(f"unknown argument '{k}' (allowed: {sorted(props)})")
            continue
        if v is None:
            continue
        typ = spec.get("type", "string")
        try:
            if typ == "string":
                if isinstance(v, (dict, list)):
                    raise ValueError("expected a string")
                v = v if isinstance(v, str) else str(v)
            elif typ == "integer":
                if isinstance(v, bool):
                    raise ValueError("expected an integer")
                if isinstance(v, str):
                    v = float(v.strip())
                if isinstance(v, float) and not v.is_integer():
                    raise ValueError("expected an integer")
                v = int(v)
            elif typ == "number":
                if isinstance(v, bool):
                    raise ValueError("expected a number")
                v = float(v)
            elif typ == "boolean":
                if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                    v = v.strip().lower() == "true"
                if not isinstance(v, bool):
                    raise ValueError("expected true or false")
            elif typ == "array":
                if isinstance(v, str):
                    v = json.loads(v)
                if not isinstance(v, list):
                    raise ValueError("expected an array")
            elif typ == "object":
                if isinstance(v, str):
                    v = json.loads(v)
                if not isinstance(v, dict):
                    raise ValueError("expected an object")
        except (ValueError, TypeError):
            errors.append(f"argument '{k}' should be of type {typ}, got {_brief(v, 60)}")
            continue
        if "enum" in spec and v not in spec["enum"]:
            errors.append(f"argument '{k}' must be one of {spec['enum']}")
            continue
        clean[k] = v
    return clean, errors


# --------------------------------------------------------------------------- #
# Basic tools
# --------------------------------------------------------------------------- #
class TextProcessingTool(Tool):
    def __init__(self):
        super().__init__("TextProcessor", "Count words and characters in a piece of text",
                         {"text": "The text to analyze"})

    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text")
        if not text:
            raise ValueError("No input text provided")
        return {"word_count": len(text.split()), "char_count": len(text)}


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


class MathTool(Tool):
    def __init__(self):
        super().__init__("Calculator", "Evaluate an arithmetic expression (+ - * / // % **)",
                         {"expression": "Arithmetic expression, e.g. '25 * 4 + 10'"})

    async def execute(self, **kwargs) -> Dict[str, Any]:
        expression = kwargs.get("expression")
        if not expression:
            raise ValueError("No mathematical expression provided")
        try:
            result = _safe_eval(ast.parse(str(expression), mode="eval"))  # no eval()
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e
        return {"expression": expression, "result": result}


class DateTimeTool(Tool):
    def __init__(self):
        super().__init__("DateTime", "Returns the current date and time from this computer's clock.", schema={
            "type": "object", "properties": {}, "required": []})

    async def execute(self, **kwargs) -> Any:
        return system_clock()


# --------------------------------------------------------------------------- #
# Workspace sandbox + file tools
# --------------------------------------------------------------------------- #
class Workspace:
    """All file/exec tools are confined to this folder (path escapes are rejected)."""

    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".agent_tmp", ".venv", "venv"}

    def __init__(self, root: Path = None):
        self.root = Path(root or WORKSPACE_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        p = (self.root / str(rel or ".")).resolve()  # resolves symlinks and '..'
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"Path {rel!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."


class ReadFileTool(Tool):
    def __init__(self, ws: Workspace):
        super().__init__("ReadFile", "Read a text file from the workspace (optionally a line range).", schema={
            "type": "object", "required": ["path"], "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"},
                "start_line": {"type": "integer", "description": "First line to return (1-based, default 1)"},
                "end_line": {"type": "integer", "description": "Last line to return (default: to the end)"}}})
        self.ws = ws

    async def execute(self, **kw) -> Any:
        p = self.ws.resolve(kw["path"])
        if not p.is_file():
            raise FileNotFoundError(f"No such file: {kw['path']}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        a = max(1, kw.get("start_line") or 1)
        b = min(len(lines), kw.get("end_line") or len(lines))
        text = "\n".join(lines[a - 1:b])
        truncated = len(text) > 20000
        return {"path": self.ws.rel(p), "total_lines": len(lines), "from_line": a, "content": text[:20000],
                "truncated": truncated}


class WriteFileTool(Tool):
    dangerous = True

    def __init__(self, ws: Workspace):
        super().__init__("WriteFile", "Create or overwrite a file in the workspace (or append to it).", schema={
            "type": "object", "required": ["path", "content"], "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"},
                "content": {"type": "string", "description": "Full text to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite (default false)"}}})
        self.ws = ws

    def describe_call(self, a) -> str:
        return f"WriteFile {'(append) ' if a.get('append') else ''}-> {a.get('path')}  [{len(a.get('content', ''))} chars]\n" \
               f"    {_brief(a.get('content', ''), 300)}"

    async def execute(self, **kw) -> Any:
        if len(kw["content"]) > 1_000_000:
            raise ValueError("Content too large (1 MB limit)")
        p = self.ws.resolve(kw["path"])
        if p == self.ws.root or p.is_dir():
            raise ValueError("Path is a directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if kw.get("append") else "w", encoding="utf-8", newline="") as f:
            f.write(kw["content"])
        return {"path": self.ws.rel(p), "bytes": p.stat().st_size}


class EditFileTool(Tool):
    dangerous = True

    def __init__(self, ws: Workspace):
        super().__init__("EditFile", "Replace one exact piece of text in a workspace file (must occur once).", schema={
            "type": "object", "required": ["path", "old_str", "new_str"], "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"},
                "old_str": {"type": "string", "description": "Exact text to replace; must appear exactly once"},
                "new_str": {"type": "string", "description": "Replacement text"}}})
        self.ws = ws

    def describe_call(self, a) -> str:
        return f"EditFile {a.get('path')}\n    - {_brief(a.get('old_str', ''), 150)}\n    + {_brief(a.get('new_str', ''), 150)}"

    async def execute(self, **kw) -> Any:
        p = self.ws.resolve(kw["path"])
        if not p.is_file():
            raise FileNotFoundError(f"No such file: {kw['path']}")
        text = p.read_text(encoding="utf-8")
        n = text.count(kw["old_str"])
        if n != 1:
            raise ValueError(f"old_str must occur exactly once (found {n})")
        p.write_text(text.replace(kw["old_str"], kw["new_str"]), encoding="utf-8")
        return {"path": self.ws.rel(p), "status": "edited"}


class DeleteFileTool(Tool):
    dangerous = True

    def __init__(self, ws: Workspace):
        super().__init__("DeleteFile", "Delete a single file from the workspace.", schema={
            "type": "object", "required": ["path"], "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace"}}})
        self.ws = ws

    async def execute(self, **kw) -> Any:
        p = self.ws.resolve(kw["path"])
        if not p.is_file():
            raise FileNotFoundError(f"No such file: {kw['path']}")
        p.unlink()
        return {"path": kw["path"], "status": "deleted"}


class ListDirTool(Tool):
    def __init__(self, ws: Workspace):
        super().__init__("ListDir", "List files and folders in the workspace.", schema={
            "type": "object", "required": [], "properties": {
                "path": {"type": "string", "description": "Folder relative to the workspace (default '.')"},
                "depth": {"type": "integer", "description": "How many levels to descend, 1-3 (default 1)"}}})
        self.ws = ws

    async def execute(self, **kw) -> Any:
        base = self.ws.resolve(kw.get("path") or ".")
        if not base.is_dir():
            raise NotADirectoryError(f"Not a folder: {kw.get('path')}")
        depth = max(1, min(3, kw.get("depth") or 1))
        entries: List[str] = []

        def walk(d: Path, level: int):
            for c in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                if c.name in Workspace.SKIP_DIRS or len(entries) >= 200:
                    continue
                entries.append(self.ws.rel(c) + ("/" if c.is_dir() else f"  ({c.stat().st_size} B)"))
                if c.is_dir() and level < depth:
                    walk(c, level + 1)

        walk(base, 1)
        return {"path": self.ws.rel(base), "entries": entries, "truncated": len(entries) >= 200}


class SearchFilesTool(Tool):
    def __init__(self, ws: Workspace):
        super().__init__("SearchFiles", "Search workspace files for a regex (like grep). Returns file:line: text.", schema={
            "type": "object", "required": ["pattern"], "properties": {
                "pattern": {"type": "string", "description": "Regular expression (case-insensitive)"},
                "path": {"type": "string", "description": "Folder to search (default '.')"},
                "glob": {"type": "string", "description": "File name filter, e.g. '*.py' (default '*')"}}})
        self.ws = ws

    async def execute(self, **kw) -> Any:
        rx = re.compile(kw["pattern"], re.IGNORECASE)
        base = self.ws.resolve(kw.get("path") or ".")
        matches: List[str] = []
        for f in sorted(base.rglob(kw.get("glob") or "*")):
            if not f.is_file() or any(part in Workspace.SKIP_DIRS for part in f.relative_to(self.ws.root).parts):
                continue
            try:
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        matches.append(f"{self.ws.rel(f)}:{n}: {line.strip()[:200]}")
                        if len(matches) >= 50:
                            return {"matches": matches, "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue  # binary/unreadable
        return {"matches": matches, "truncated": False}


# --------------------------------------------------------------------------- #
# Code execution (gated; confined to the workspace as a working dir, NOT a real sandbox)
# --------------------------------------------------------------------------- #
def _run_proc(cmd, cwd: Path, timeout: int, shell: bool = False) -> Dict[str, Any]:
    try:
        r = subprocess.run(cmd, cwd=str(cwd), shell=shell, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return {"returncode": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-3000:],
                "timed_out": False}
    except subprocess.TimeoutExpired as e:
        def _s(x):
            return (x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or ""))[-3000:]
        return {"returncode": None, "stdout": _s(e.stdout), "stderr": _s(e.stderr), "timed_out": True}


class RunPythonTool(Tool):
    dangerous = True

    def __init__(self, ws: Workspace):
        super().__init__("RunPython", "Run a Python script (working dir = workspace) and return stdout/stderr. "
                                      "Use print() to output results.", schema={
            "type": "object", "required": ["code"], "properties": {
                "code": {"type": "string", "description": "Python source code to run"},
                "timeout": {"type": "integer", "description": "Seconds before it is killed (default 20, max 120)"}}})
        self.ws = ws

    def describe_call(self, a) -> str:
        return f"RunPython (timeout {a.get('timeout', 20)}s):\n" + "\n".join("    | " + l for l in a.get("code", "").splitlines()[:40])

    async def execute(self, **kw) -> Any:
        tmp = self.ws.root / ".agent_tmp"
        tmp.mkdir(exist_ok=True)
        script = tmp / f"run_{uuid4().hex[:8]}.py"
        script.write_text(kw["code"], encoding="utf-8")
        try:
            return await asyncio.to_thread(_run_proc, [sys.executable, "-I", str(script)], self.ws.root,
                                           max(1, min(120, kw.get("timeout") or 20)))
        finally:
            script.unlink(missing_ok=True)


class RunShellTool(Tool):
    dangerous = True

    def __init__(self, ws: Workspace):
        super().__init__("RunShell", "Run a shell command (working dir = workspace); returns stdout/stderr.", schema={
            "type": "object", "required": ["command"], "properties": {
                "command": {"type": "string", "description": "The command line to run"},
                "timeout": {"type": "integer", "description": "Seconds before it is killed (default 20, max 120)"}}})
        self.ws = ws

    def describe_call(self, a) -> str:
        return f"RunShell (timeout {a.get('timeout', 20)}s): {a.get('command')}"

    async def execute(self, **kw) -> Any:
        return await asyncio.to_thread(_run_proc, kw["command"], self.ws.root,
                                       max(1, min(120, kw.get("timeout") or 20)), True)


# --------------------------------------------------------------------------- #
# Web: fetch (existing, SSRF-safe) and search
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Collects visible text from HTML, skipping scripts/styles."""

    SKIP = {"script", "style", "noscript", "svg", "template"}
    BLOCKS = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _check_public_url(url: str) -> None:
    """Refuse non-http(s) URLs and anything resolving to a local/private address."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"Blocked: {host} resolves to non-public address {ip}")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_url(newurl)  # redirects can't sneak us onto the local network
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebFetchTool(Tool):
    """Fetch a public web page and return its readable text."""

    MAX_BYTES = 1_000_000

    def __init__(self, max_chars: int = 6000, timeout: float = 20.0):
        super().__init__("WebFetch", "Fetch a public web page (http/https) and return its readable text",
                         {"url": "Full URL of the page, e.g. https://example.com/article"})
        self.max_chars = max_chars
        self.timeout = timeout

    async def execute(self, **kwargs) -> Dict[str, Any]:
        url = kwargs.get("url")
        if not url:
            raise ValueError("No URL provided")
        url = str(url).strip()
        if "://" not in url:
            url = "https://" + url
        return await asyncio.to_thread(self._fetch, url)

    def _fetch(self, url: str) -> Dict[str, Any]:
        _check_public_url(url)
        opener = urllib.request.build_opener(_SafeRedirect)
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AgentHarness/2.0)",
            "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5"})
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                ctype = resp.headers.get("Content-Type", "").lower()
                charset = resp.headers.get_content_charset() or "utf-8"
                final_url = resp.geturl()
                raw = resp.read(self.MAX_BYTES)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} fetching {url}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not fetch {url}: {e.reason}") from e

        if not any(t in ctype for t in ("text/", "json", "xml")):
            raise ValueError(f"Unsupported content type: {ctype or 'unknown'}")
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        title = ""
        if "html" in ctype:
            parser = _TextExtractor()
            parser.feed(text)
            title = parser.title.strip()
            text = "".join(parser.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return {"url": final_url, "title": title, "text": text[: self.max_chars],
                "truncated": len(text) > self.max_chars}


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect; recover the real URL."""
    if "uddg=" in href:
        q = urllib.parse.urlparse(href if "://" in href else "https:" + href).query
        vals = urllib.parse.parse_qs(q).get("uddg")
        if vals:
            return vals[0]
    return "https:" + href if href.startswith("//") else href


class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._cur: Optional[Dict[str, str]] = None
        self._mode: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "a" and "result__a" in cls:
            self._cur = {"title": "", "url": _ddg_unwrap(a.get("href", "")), "snippet": ""}
            self.results.append(self._cur)
            self._mode = "title"
        elif self._cur is not None and "result__snippet" in cls:
            self._mode = "snippet"

    def handle_endtag(self, tag):
        if tag == "a" and self._mode == "title":
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "td", "div", "span"):
            self._mode = None

    def handle_data(self, data):
        if self._cur is not None and self._mode:
            self._cur[self._mode] += data


class WebSearchTool(Tool):
    """Web search: your SearXNG instance if SEARXNG_URL is set, else DuckDuckGo's HTML endpoint."""

    def __init__(self, timeout: float = 20.0):
        super().__init__("WebSearch", "Search the web. Returns titles, URLs and snippets; use WebFetch to read a result.",
                         schema={"type": "object", "required": ["query"], "properties": {
                             "query": {"type": "string", "description": "Search query"},
                             "max_results": {"type": "integer", "description": "1-10 (default 5)"}}})
        self.timeout = timeout

    async def execute(self, **kw) -> Any:
        n = max(1, min(10, kw.get("max_results") or 5))
        results = await asyncio.to_thread(self._search, kw["query"], n)
        if not results:
            return {"query": kw["query"], "results": [], "note": "No results; try different keywords."}
        return {"query": kw["query"], "results": results}

    def _search(self, query: str, n: int) -> List[Dict[str, str]]:
        searx = os.getenv("SEARXNG_URL", "").rstrip("/")
        if searx:  # user-configured, trusted (usually localhost) so the public-IP check is skipped
            url = f"{searx}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
            req = urllib.request.Request(url, headers={"User-Agent": "AgentHarness/2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": (x.get("content") or "")[:300]}
                    for x in data.get("results", [])[:n]]
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": query}).encode(),
            headers={"User-Agent": "Mozilla/5.0 (compatible; AgentHarness/2.0)"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                html = r.read().decode("utf-8", "replace")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Search failed: {e}") from e
        return self.parse_ddg(html, n)

    @staticmethod
    def parse_ddg(html: str, n: int) -> List[Dict[str, str]]:
        p = _DDGParser()
        p.feed(html)
        out = []
        for r in p.results:
            r = {k: " ".join(v.split()) for k, v in r.items()}
            if r["url"] and "duckduckgo.com/y.js" not in r["url"]:  # skip ads
                out.append(r)
        return out[:n]


# --------------------------------------------------------------------------- #
# Memory tools (backed by FactStore, defined below)
# --------------------------------------------------------------------------- #
class RememberTool(Tool):
    def __init__(self, facts: "FactStore"):
        super().__init__("Remember", "Save a durable fact (user preferences, names, project details) for future "
                                     "conversations.", {"fact": "One self-contained sentence to remember"})
        self.facts = facts

    async def execute(self, **kw) -> Any:
        fid = self.facts.add(kw["fact"])
        return {"saved": kw["fact"], "id": fid}


class RecallTool(Tool):
    def __init__(self, facts: "FactStore", llm):
        super().__init__("Recall", "Search saved facts by meaning.", {"query": "What to look for"})
        self.facts, self.llm = facts, llm

    async def execute(self, **kw) -> Any:
        hits = await self.facts.search(self.llm, kw["query"], k=5)
        return {"facts": [{"id": it["id"], "text": it["text"]} for _, it in hits]}


class ForgetTool(Tool):
    dangerous = True

    def __init__(self, facts: "FactStore"):
        super().__init__("Forget", "Delete a saved fact by its id.", {"id": "Fact id from Recall"})
        self.facts = facts

    async def execute(self, **kw) -> Any:
        if not self.facts.remove(kw["id"]):
            raise ValueError(f"No fact with id {kw['id']}")
        return {"deleted": kw["id"]}


# --------------------------------------------------------------------------- #
# MCP client (stdio transport, JSON-RPC 2.0, newline-delimited)
# --------------------------------------------------------------------------- #
class MCPServer:
    def __init__(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None,
                 cwd: str = None, timeout: float = 30.0):
        self.name, self.command, self.args = name, command, args or []
        self.env, self.cwd, self.timeout = env or {}, cwd, timeout
        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._wlock = threading.Lock()
        self.tools: List[Dict[str, Any]] = []

    def start(self) -> None:
        exe = shutil.which(self.command) or self.command
        self.proc = subprocess.Popen([exe, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                                     env={**os.environ, **self.env}, cwd=self.cwd)
        threading.Thread(target=self._reader, daemon=True).start()
        self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                    "clientInfo": {"name": "agent-harness", "version": "2.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        cursor = None
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            self.tools.extend(res.get("tools", []))
            cursor = res.get("nextCursor")
            if not cursor:
                break

    def _send(self, msg: Dict[str, Any]) -> None:
        with self._wlock:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _reader(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "method" in msg and "id" in msg:  # server -> client request: answer politely
                reply = {"jsonrpc": "2.0", "id": msg["id"]}
                if msg["method"] == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "Method not supported"}
                try:
                    self._send(reply)
                except Exception:
                    pass
            elif "id" in msg:
                slot = self._pending.get(msg["id"])
                if slot:
                    slot["msg"] = msg
                    slot["event"].set()
        for slot in list(self._pending.values()):  # process died: wake everyone
            slot.setdefault("msg", {"error": {"message": "MCP server exited"}})
            slot["event"].set()

    def request(self, method: str, params: Dict[str, Any] = None, timeout: float = None) -> Dict[str, Any]:
        with self._wlock:
            self._next_id += 1
            rid = self._next_id
        slot = {"event": threading.Event()}
        self._pending[rid] = slot
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        if not slot["event"].wait(timeout or self.timeout):
            self._pending.pop(rid, None)
            raise TimeoutError(f"MCP {self.name}: {method} timed out")
        self._pending.pop(rid, None)
        msg = slot["msg"]
        if "error" in msg:
            raise RuntimeError(f"MCP {self.name}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result", {})

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class MCPTool(Tool):
    def __init__(self, server: MCPServer, spec: Dict[str, Any]):
        schema = spec.get("inputSchema") or {}
        schema = {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}
        name = re.sub(r"[^A-Za-z0-9_\-]", "_", f"{server.name}__{spec['name']}")
        super().__init__(name, (spec.get("description") or spec["name"])[:500], schema=schema)
        self.server, self.remote_name = server, spec["name"]
        # Only skip the confirmation gate when the server says the tool is read-only.
        self.dangerous = not (spec.get("annotations") or {}).get("readOnlyHint", False)

    async def execute(self, **kw) -> Any:
        res = await asyncio.to_thread(self.server.request, "tools/call",
                                      {"name": self.remote_name, "arguments": kw}, 120)
        parts = [c.get("text", "") if c.get("type") == "text" else f"[{c.get('type', 'content')} omitted]"
                 for c in res.get("content", [])]
        text = "\n".join(parts)
        if res.get("isError"):
            raise RuntimeError(text or "MCP tool reported an error")
        return text


def load_mcp(config_path: Path = None) -> Tuple[List[MCPServer], List[Tool]]:
    """Read mcp.json ({"mcpServers": {name: {command, args, env}}}) and start every server."""
    path = Path(config_path or os.getenv("MCP_CONFIG", str(BASE_DIR / "mcp.json")))
    if not path.exists():
        return [], []
    cfg = json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {})
    servers: List[MCPServer] = []
    tools: List[Tool] = []
    for name, c in cfg.items():
        srv = MCPServer(name, c["command"], c.get("args"), c.get("env"), c.get("cwd"))
        try:
            srv.start()
        except Exception as e:
            logger.warning(f"MCP server '{name}' failed to start: {e}")
            srv.close()
            continue
        servers.append(srv)
        tools.extend(MCPTool(srv, t) for t in srv.tools)
    return servers, tools


def build_tools(llm, facts: "FactStore", ws: Workspace, mcp_tools: List[Tool] = None,
                read_only: bool = False) -> List[Tool]:
    """The full toolbox. read_only=True drops every tool that needs confirmation (for background jobs)."""
    tools: List[Tool] = [
        TextProcessingTool(), MathTool(), DateTimeTool(), WebFetchTool(), WebSearchTool(),
        ReadFileTool(ws), ListDirTool(ws), SearchFilesTool(ws), WriteFileTool(ws), EditFileTool(ws),
        DeleteFileTool(ws), RunPythonTool(ws), RunShellTool(ws),
        RememberTool(facts), RecallTool(facts, llm), ForgetTool(facts), *(mcp_tools or []),
    ]
    off = {s.strip() for s in os.getenv("AGENT_DISABLE_TOOLS", "").split(",") if s.strip()}
    tools = [t for t in tools if t.name not in off]
    if read_only:
        tools = [t for t in tools if not t.dangerous]
    return tools


# --------------------------------------------------------------------------- #
# Memory: vector store, facts, conversation (summary + semantic recall)
# --------------------------------------------------------------------------- #
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))  # sent verbatim to the model per turn
MIN_SIM = float(os.getenv("MEMORY_MIN_SIM", "0.35"))  # cosine cutoff; tune for your embedding model
_STOP = set("the a an and or of to in on for is are was were be it this that with as at by from i you my me we do "
            "does did what which who how when where can could would should please about".split())


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in _STOP}


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class VectorStore:
    """Tiny JSON-backed store. Embeds lazily; falls back to keyword overlap if no embeddings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.items: List[Dict[str, Any]] = []
        try:
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            self.items = []

    def save(self) -> None:
        try:
            _atomic_write(self.path, self.items)
        except OSError as e:
            logger.warning(f"Could not save {self.path}: {e}")

    def add(self, text: str, key: str = None) -> str:
        item = {"id": uuid4().hex[:6], "key": key, "text": text.strip(), "ts": time.time(), "vec": None}
        self.items.append(item)
        self.save()
        return item["id"]

    def has(self, key: str) -> bool:
        return any(it.get("key") == key for it in self.items)

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        return next((it for it in self.items if it.get("key") == key), None)

    def remove(self, item_id: str) -> bool:
        n = len(self.items)
        self.items = [it for it in self.items if it["id"] != item_id]
        if len(self.items) != n:
            self.save()
        return len(self.items) != n

    async def search(self, llm, query: str, k: int = 4, min_sim: float = None,
                     candidates: List[Dict[str, Any]] = None) -> List[Tuple[float, Dict[str, Any]]]:
        items = self.items if candidates is None else candidates
        if not items:
            return []
        min_sim = MIN_SIM if min_sim is None else min_sim
        qv = None
        if llm is not None:
            missing = [it for it in items if not it.get("vec")][:64]
            out = await llm.embed([query] + [it["text"][:1000] for it in missing])
            if out and len(out) == len(missing) + 1:
                qv = out[0]
                for it, v in zip(missing, out[1:]):
                    it["vec"] = [round(x, 5) for x in v]
                if missing:
                    self.save()
        if qv:
            scored = [(_cosine(qv, it["vec"]), it) for it in items if it.get("vec")]
            scored = [(s, it) for s, it in scored if s >= min_sim]
        else:
            qt = _tokens(query)
            scored = [((len(qt & _tokens(it["text"])) / max(1, len(qt))), it) for it in items] if qt else []
            scored = [(s, it) for s, it in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


class FactStore(VectorStore):
    """Durable facts shared by every session (preferences, names, project details)."""

    def __init__(self, path: Path = None):
        super().__init__(path or (MEMORY_DIR / "facts.json"))


class LearningStore:
    """SQLite-backed mistake ledger and curated lessons.

    Only model-reviewed lessons are retrieved automatically. The local verifier is a
    heuristic, not ground truth; uncertain items stay out of the trusted lesson set.
    The store is deliberately keyword-based on the hot path to avoid embedding/model
    calls on every user prompt.
    """

    def __init__(self, path: Path = None):
        self.path = Path(path or (MEMORY_DIR / "learning.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                source TEXT NOT NULL,
                task TEXT NOT NULL,
                failure TEXT NOT NULL,
                correction TEXT NOT NULL,
                lesson TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uncertain',
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence TEXT NOT NULL DEFAULT '',
                verifier TEXT NOT NULL DEFAULT '',
                exported INTEGER NOT NULL DEFAULT 0
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_lessons_status ON lessons(status)")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def add(self, *, source: str, task: str, failure: str, correction: str = "",
            lesson: str = "", status: str = "uncertain", confidence: float = 0.0,
            evidence: str = "", verifier: str = "") -> int:
        with self._connect() as db:
            cur = db.execute("""INSERT INTO lessons
                (created,source,task,failure,correction,lesson,status,confidence,evidence,verifier)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), source, task[:12000], failure[:6000], correction[:6000],
                 lesson[:4000], status, max(0.0, min(1.0, float(confidence))), evidence[:4000], verifier[:300]))
            return int(cur.lastrowid)

    def get(self, lesson_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
        return dict(row) if row else None

    def set_status(self, lesson_id: int, status: str, verifier: str = "user") -> bool:
        if status not in ("verified", "uncertain", "rejected"):
            raise ValueError("invalid learning status")
        with self._connect() as db:
            cur = db.execute("UPDATE lessons SET status=?, verifier=? WHERE id=?",
                             (status, verifier[:300], lesson_id))
            return cur.rowcount > 0

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        q = _tokens(query)
        if not q:
            return []
        with self._connect() as db:
            rows = db.execute("SELECT * FROM lessons WHERE status='verified' ORDER BY created DESC LIMIT 500").fetchall()
        ranked = []
        for row in rows:
            item = dict(row)
            searchable = " ".join((item["task"], item["failure"], item["correction"], item["lesson"]))
            tokens = _tokens(searchable)
            overlap = len(q & tokens) / max(1, len(q))
            if overlap >= 0.18:
                ranked.append((overlap * (0.75 + 0.25 * item["confidence"]), item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:k]]

    def list_recent(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM lessons ORDER BY created DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()]

    def stats(self) -> Dict[str, int]:
        with self._connect() as db:
            rows = db.execute("SELECT status, COUNT(*) n FROM lessons GROUP BY status").fetchall()
        out = {"verified": 0, "uncertain": 0, "rejected": 0}
        out.update({r["status"]: r["n"] for r in rows})
        out["total"] = sum(out.values())
        return out

    def export_training(self, path: Path = None) -> Tuple[Path, int]:
        """Export only verified, explicit corrections as chat fine-tuning JSONL."""
        out = Path(path or (MEMORY_DIR / "learning_training.jsonl"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            rows = db.execute("""SELECT id,task,correction,lesson FROM lessons
                WHERE status='verified' AND source='user_feedback' AND correction != ''
                ORDER BY created""").fetchall()
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                record = {"messages": [
                    {"role": "system", "content": "Answer the user's task accurately. Follow the verified correction when it applies."},
                    {"role": "user", "content": r["task"]},
                    {"role": "assistant", "content": r["correction"]}
                ]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out, len(rows)

    async def review(self, llm, lesson_id: int) -> Dict[str, Any]:
        """Use the configured fast role to triage a candidate; uncertain is the safe default."""
        item = self.get(lesson_id)
        if not item:
            raise ValueError(f"No learning record with ID {lesson_id}")
        prompt = [
            {"role": "system", "content": (
                "You are a cautious quality reviewer for a local AI agent. Review a proposed lesson using only "
                "the supplied task, failure, correction, and evidence. Do not assume the user's correction is "
                "automatically true. Return ONLY JSON: {\"verdict\":\"supported\"|\"uncertain\"|\"contradicted\", "
                "\"confidence\":0.0, \"reason\":\"short explanation\"}. Use supported only when the correction/lesson "
                "is internally consistent and supported by the evidence; if external facts or missing context are needed, use uncertain."
            )},
            {"role": "user", "content": json.dumps({k: item.get(k) for k in
                ("task", "failure", "correction", "lesson", "evidence")}, ensure_ascii=False)}
        ]
        try:
            raw = await llm.chat(prompt, role="fast", json_mode=True)
            verdict = extract_json(raw)
            name = str(verdict.get("verdict", "uncertain")).lower()
            conf = float(verdict.get("confidence", 0.0))
            reason = str(verdict.get("reason", "No reason supplied"))[:1000]
            if name == "supported" and conf >= 0.80:
                status = "verified"
            elif name == "contradicted" and conf >= 0.80:
                status = "rejected"
            else:
                status = "uncertain"
            self.set_status(lesson_id, status, verifier=f"local-model-review: {reason}")
            with self._connect() as db:
                db.execute("UPDATE lessons SET confidence=?, evidence=? WHERE id=?",
                           (max(0.0, min(1.0, conf)), reason, lesson_id))
            return {"status": status, "confidence": conf, "reason": reason}
        except Exception as e:
            self.set_status(lesson_id, "uncertain", verifier=f"review failed: {e}")
            return {"status": "uncertain", "confidence": 0.0, "reason": f"Review failed: {e}"}


class ConversationMemory:
    """Per-session history on disk + rolling summary + semantic recall of older turns."""

    def __init__(self, session: str = "default", directory: str = None, max_stored: int = 1000):
        self.session = re.sub(r"[^\w\-]", "_", session) or "default"
        self.directory = Path(directory) if directory else BASE_DIR / "sessions"
        self.path = self.directory / f"{self.session}.json"
        self.max_stored = max_stored
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""
        self.summary_upto = 0  # highest message seq folded into the summary
        self._seq = 0
        self._load()
        self.vindex = VectorStore(self.directory / f"{self.session}.vec.json")

    # ----- persistence ----------------------------------------------------- #
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (ValueError, OSError) as e:
            logger.warning(f"Could not read {self.path} ({e}); starting with empty history")
            return
        if isinstance(data, list):  # legacy v1 format: a bare list of messages
            data = {"messages": data}
        msgs = [m for m in data.get("messages", [])
                if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m]
        for m in msgs:
            self._seq = max(self._seq, m.get("seq", 0))
        for m in msgs:
            if "seq" not in m:
                self._seq += 1
                m["seq"] = self._seq
        self.messages = msgs
        self.summary = data.get("summary", "")
        self.summary_upto = data.get("summary_upto", 0)

    def _save(self) -> None:
        try:
            _atomic_write(self.path, {"summary": self.summary, "summary_upto": self.summary_upto,
                                      "messages": self.messages})
        except OSError as e:
            logger.warning(f"Could not save history to {self.path}: {e}")

    def add(self, role: str, content: str) -> None:
        self._seq += 1
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), "seq": self._seq})
        self.messages = self.messages[-self.max_stored:]
        self._save()

    def recent(self, n: int = MAX_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[-n:]]

    def clear(self) -> None:
        self.messages, self.summary, self.summary_upto = [], "", 0
        self._save()
        self.vindex.items = []
        self.vindex.save()
        self.clear_checkpoint()

    # ----- rolling summary ------------------------------------------------- #
    async def maybe_summarize(self, llm, keep: int = MAX_CONTEXT_MESSAGES, batch: int = 10) -> bool:
        """Fold messages that fell out of the window into a running summary (uses the fast model)."""
        old = [m for m in self.messages[:-keep] if m["seq"] > self.summary_upto] if keep else []
        if len(old) < batch:
            return False
        text = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in old)
        prompt = [
            {"role": "system", "content": (
                "You maintain a concise running summary of a conversation between a user and an AI assistant. "
                "Merge the new messages into the summary. Keep names, decisions, preferences, file names, "
                "results and open tasks. Maximum 200 words. Reply with ONLY the updated summary.")},
            {"role": "user", "content": f"Current summary:\n{self.summary or '(empty)'}\n\nNew messages:\n{text}"},
        ]
        try:
            new = (await llm.chat(prompt, role="fast")).strip()
        except Exception as e:
            logger.warning(f"Summarization skipped: {e}")
            return False
        if not new:
            return False
        self.summary, self.summary_upto = new[:2000], old[-1]["seq"]
        self._save()
        return True

    # ----- semantic recall of older turns ---------------------------------- #
    async def recall(self, llm, query: str, k: int = 3, keep: int = MAX_CONTEXT_MESSAGES) -> List[str]:
        older = (self.messages[:-keep] if keep else list(self.messages))[-300:]
        if not older:
            return []
        cands = []
        for m in older:
            key = f"{self.session}:{m['seq']}"
            it = self.vindex.get_by_key(key)
            if it is None:
                self.vindex.add(f"{m['role']}: {m['content'][:800]}", key=key)
                it = self.vindex.get_by_key(key)
            cands.append(it)
        hits = await self.vindex.search(llm, query, k=k, candidates=cands)
        return [it["text"] for _, it in hits]

    async def build_notes(self, llm, facts: FactStore, query: str, max_chars: int = 3500) -> str:
        """Context injected into the system prompt: summary + relevant facts + relevant old turns."""
        parts = []
        if self.summary:
            parts.append("Summary of the earlier conversation:\n" + self.summary)
        hits = await facts.search(llm, query, k=4)
        if hits:
            parts.append("Saved facts:\n" + "\n".join(f"- {it['text']}" for _, it in hits))
        recalled = await self.recall(llm, query, k=3)
        if recalled:
            parts.append("Possibly relevant earlier messages:\n" + "\n".join("- " + r[:300] for r in recalled))
        return "\n\n".join(parts)[:max_chars]

    # ----- checkpoints (interrupt/resume) ---------------------------------- #
    @property
    def checkpoint_path(self) -> Path:
        return self.directory / f"{self.session}.checkpoint.json"

    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        try:
            _atomic_write(self.checkpoint_path, data)
        except OSError as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None

    def clear_checkpoint(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except OSError:
            pass

    @staticmethod
    def list_sessions(directory: str = None) -> List[str]:
        d = Path(directory) if directory else BASE_DIR / "sessions"
        return sorted(p.stem for p in d.glob("*.json") if "." not in p.stem) if d.exists() else []


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
ConfirmFn = Callable[[str, str], Awaitable[bool]]


class Agent(ABC):
    def __init__(self, name: str, description: str, tools: List[Tool], llm: LocalQwenClient = None):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm = llm or LocalQwenClient()
        self.state = AgentState.IDLE
        self.context: Optional[AgentContext] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.history: List[Dict[str, str]] = []   # prior turns supplied by the caller
        self.notes: str = ""                      # memory notes (summary/facts/recall) for the system prompt
        self.resume: Optional[Dict[str, Any]] = None
        self.on_event: Optional[Callable[[str, str], None]] = None   # live progress hook
        self.on_token: Optional[Callable[[str], None]] = None        # streaming hook
        self.cancel: Optional[threading.Event] = None                # set by Ctrl+C
        self.tracer: Optional[Tracer] = None
        self.confirm: Optional[ConfirmFn] = None                     # async (tool_name, description) -> bool
        self.auto_approve = False
        self.task_id = str(uuid4())

    # ----- plumbing -------------------------------------------------------- #
    def share_runtime(self, src: "Agent") -> None:
        for a in ("on_event", "on_token", "cancel", "tracer", "confirm", "auto_approve", "notes", "history"):
            setattr(self, a, getattr(src, a))

    def _emit(self, kind: str, text: str) -> None:
        self._trace("event", kind=kind, text=str(text)[:2000])
        if self.on_event:
            try:
                self.on_event(kind, text)
            except Exception:  # a broken printer must never break the agent
                pass

    def _trace(self, event: str, **data: Any) -> None:
        if self.tracer:
            self.tracer.log(event, task=self.context.task_id if self.context else None, agent=self.name, **data)

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _with_notes(self, text: str) -> str:
        return text + (f"\n\nMemory notes (may be relevant; may be outdated):\n{self.notes}" if self.notes else "")

    async def _llm(self, messages, tools=None, role="main", json_mode=False, stream=False) -> Dict[str, Any]:
        reply = await self.llm.chat_tools(messages, tools, cancel=self.cancel,
                                          on_token=self.on_token if stream else None, role=role, json_mode=json_mode)
        self._trace("llm", role=role, model=reply.get("model"), seconds=round(reply.get("elapsed", 0), 2),
                    usage=reply.get("usage"), tool_calls=[c["name"] for c in reply["tool_calls"]],
                    chars=len(reply["content"]))
        return reply

    async def execute_task(self, task_input: str, metadata: Dict[str, Any] = None) -> AgentResponse:
        self.state = AgentState.THINKING
        start_time = time.time()
        self.conversation_history = []
        meta = dict(metadata or {})
        self.history = meta.pop("history", None) or []
        self.notes = meta.pop("notes", "") or ""
        self.resume = meta.pop("resume", None)
        context = AgentContext(task_id=str(uuid4()), session_id=str(uuid4()), user_input=task_input,
                               timestamp=time.time(), metadata=meta)
        self.context = context
        self._trace("task_start", input=task_input, resumed=bool(self.resume))
        try:
            logger.info(f"Agent {self.name} executing task: {task_input}")
            result = await self._execute_task_logic(task_input)
            self.state = AgentState.DONE
            self._trace("task_done", seconds=round(time.time() - start_time, 2))
            return AgentResponse(task_id=context.task_id, status=TaskStatus.COMPLETED, result=result,
                                 execution_time=time.time() - start_time,
                                 metadata={**context.metadata, "session_id": context.session_id})
        except TaskInterrupted as ti:
            self.state = AgentState.IDLE
            self._trace("task_interrupted")
            return AgentResponse(task_id=context.task_id, status=TaskStatus.INTERRUPTED, result=None,
                                 error="interrupted", execution_time=time.time() - start_time,
                                 metadata={"snapshot": ti.snapshot})
        except Cancelled:
            self.state = AgentState.IDLE
            self._trace("task_interrupted")
            return AgentResponse(task_id=context.task_id, status=TaskStatus.INTERRUPTED, result=None,
                                 error="interrupted", execution_time=time.time() - start_time,
                                 metadata={"snapshot": None})
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent {self.name} failed to execute task: {e}")
            self._trace("task_failed", error=str(e))
            return AgentResponse(task_id=context.task_id, status=TaskStatus.FAILED, result=None, error=str(e),
                                 execution_time=time.time() - start_time)

    @abstractmethod
    async def _execute_task_logic(self, task_input: str) -> Any:
        ...

    def get_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tools]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "state": self.state.value,
                "task_id": self.task_id, "tools": self.get_tools()}


class ChatAgent(Agent):
    """Plain conversation, no tools. Streams."""

    def __init__(self, name: str = "Chat", tools: List[Tool] = None, llm: LocalQwenClient = None):
        super().__init__(name, "Plain conversation with the model", tools or [], llm)

    async def _execute_task_logic(self, task_input: str) -> Any:
        system = self._with_notes(f"{clock_line()} You are a helpful assistant.")
        reply = await self._llm([{"role": "system", "content": system}, *self.history,
                                 {"role": "user", "content": task_input}], stream=True)
        return {"task": task_input, "answer": reply["content"], "streamed": reply["streamed"],
                "executed_tools": 0, "tool_results": []}


class PlanningAgent(Agent):
    """Uses the LLM (thinking model if configured) to decompose a task into steps."""

    def __init__(self, name: str = "Planner", tools: List[Tool] = None, llm: LocalQwenClient = None):
        super().__init__(name, "Agent that plans complex tasks", tools or [], llm)
        self.planning_history: List[Dict[str, Any]] = []

    async def _execute_task_logic(self, task_input: str) -> Any:
        messages = [
            {"role": "system", "content": self._with_notes(
                f"{clock_line()} You are a planning assistant. Break the user's task into 3-6 concrete steps. "
                'Reply with ONLY JSON: {"steps": ["step 1", "step 2", ...]}')},
            *self.history,
            {"role": "user", "content": task_input},
        ]
        raw = (await self._llm(messages, role="think", json_mode=True))["content"]
        try:
            steps = [str(s) for s in extract_json(raw).get("steps", [])]
        except ValueError:
            steps = [line.strip("-*0123456789. ") for line in raw.splitlines() if line.strip()]
        self.planning_history.append({"task": task_input, "timestamp": time.time(), "steps": steps})
        return {"task": task_input, "planning_steps": steps, "status": "planned"}


class ExecutionAgent(Agent):
    """Tool loop. Native tool calling when the model supports it, JSON-text protocol otherwise."""

    MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
    MAX_OBSERVATION_CHARS = 8000

    def __init__(self, name: str = "Executor", tools: List[Tool] = None, llm: LocalQwenClient = None,
                 max_steps: int = None):
        super().__init__(name, "Agent that executes tasks using available tools", tools or [], llm)
        self.max_steps = max_steps or self.MAX_STEPS

    # ----- prompts --------------------------------------------------------- #
    def _system_prompt_native(self) -> str:
        return self._with_notes(
            f"{clock_line()} You are a capable assistant with tools. Use tools for anything you cannot know or "
            "compute reliably yourself (live web content, exact math, files, running code); otherwise answer "
            "directly. You may call several independent tools in one turn. Never repeat an identical call. "
            "File tools work inside the workspace folder. When you are finished, reply with the final answer "
            "in plain text.")

    def _system_prompt_json(self) -> str:
        tool_specs = json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters}
                                 for t in self.tools], indent=2)
        return self._with_notes(
            f"{clock_line()}\nYou are a task-executing agent. You can use these tools:\n{tool_specs}\n\n"
            "On every turn reply with ONLY one JSON object, no other text. "
            'Always begin with a short "thought" explaining your reasoning for this step.\n'
            'To call a tool:  {"thought": "<why>", "action": "tool", "tool": "<n>", "args": {<param>: <value>}}\n'
            'To finish:       {"thought": "<why>", "action": "final", "answer": "<answer>"}\n'
            'Example: {"thought": "I need to compute this", "action": "tool", "tool": "Calculator", '
            '"args": {"expression": "2+2"}}\n'
            "After a tool call you will receive its result; use it to decide the next step. "
            "Never repeat an identical tool call; use the earlier result or try something different.")

    # ----- context budget -------------------------------------------------- #
    def _compact(self, messages: List[Dict[str, Any]]) -> None:
        """Shrink old tool outputs when the prompt nears the context window."""
        budget = int(self.llm.num_ctx * 3.0 * 0.7)  # ~3 chars/token, keep 30% headroom for the reply

        def size(m):
            return len(str(m.get("content") or "")) + sum(len(json.dumps(c.get("arguments", {})))
                                                           for c in m.get("tool_calls") or [])
        total = sum(size(m) for m in messages)
        if total <= budget:
            return
        for m in messages[:-4]:
            c = str(m.get("content") or "")
            if (m["role"] == "tool" or c.startswith("Tool result:")) and len(c) > 300:
                total -= len(c) - 300
                m["content"] = c[:300] + " ...[trimmed to save context]"
                if total <= budget:
                    break

    # ----- one tool call --------------------------------------------------- #
    async def _run_call(self, call: Dict[str, Any], by_name: Dict[str, Tool]) -> Dict[str, Any]:
        name, args = call["name"], call["arguments"]
        tool = by_name.get(name)
        self._emit("tool", f"{name}({json.dumps(args, ensure_ascii=False, default=str)})")
        t0 = time.time()
        if tool is None:
            obs = {"tool": name, "status": "failed", "error": f"Unknown tool {name!r}. Available: {list(by_name)}"}
        else:
            clean, errors = validate_args(tool.schema, args)
            if errors:
                obs = {"tool": name, "status": "failed",
                       "error": "Invalid arguments: " + "; ".join(errors) + f". Expected schema: {json.dumps(tool.schema)}"}
            elif tool.needs_confirmation(clean) and not await self._approved(tool, clean):
                obs = {"tool": name, "status": "failed",
                       "error": "The user denied this action. Do not retry it; choose another approach or ask the user."}
            else:
                try:
                    obs = {"tool": name, "result": await tool.execute(**clean), "status": "success"}
                except Exception as e:
                    obs = {"tool": name, "result": None, "status": "failed", "error": str(e)}
        self._trace("tool", name=name, args=args, status=obs["status"], seconds=round(time.time() - t0, 3),
                    output=_brief(obs.get("result") if obs["status"] == "success" else obs.get("error"), 500))
        self._emit("result", f"{name}: " + _brief(obs.get("result") if obs["status"] == "success" else obs.get("error")))
        return obs

    async def _approved(self, tool: Tool, args: Dict[str, Any]) -> bool:
        if self.auto_approve:
            return True
        if self.confirm is None:
            return False  # no way to ask -> safe default
        try:
            return bool(await self.confirm(tool.name, tool.describe_call(args)))
        except Exception:
            return False

    # ----- the loop -------------------------------------------------------- #
    async def _execute_task_logic(self, task_input: str) -> Any:
        resume = self.resume if (self.resume and self.resume.get("kind") == "exec") else None
        return await self.run_loop(task_input, resume)

    def _snapshot(self, task, messages, steps, tool_results, seen, step) -> Dict[str, Any]:
        return {"kind": "exec", "task": task, "messages": messages, "steps": steps, "tool_results": tool_results,
                "seen": sorted(seen), "step": step}

    async def run_loop(self, task_input: str, resume: Dict[str, Any] = None) -> Dict[str, Any]:
        native = await self.llm.probe_native_tools() and bool(self.tools)
        by_name = {t.name: t for t in self.tools}
        specs = [t.spec() for t in self.tools] if native else None
        if resume:
            messages, steps = resume["messages"], resume["steps"]
            tool_results, seen, step0 = resume["tool_results"], set(resume["seen"]), resume["step"]
            task_input = resume.get("task", task_input)
        else:
            system = self._system_prompt_native() if native else self._system_prompt_json()
            messages = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": task_input}]
            steps, tool_results, seen, step0 = [], [], set(), 0

        def snap(step):
            return self._snapshot(task_input, messages, steps, tool_results, seen, step)

        def final(answer, streamed=False, incomplete=False):
            return {"task": task_input, "answer": answer, "executed_tools": len(tool_results),
                    "tool_results": tool_results, "steps": steps, "streamed": streamed, "incomplete": incomplete}

        for step in range(step0, self.max_steps):
            if self._cancelled():
                raise TaskInterrupted(snap(step))
            self._compact(messages)
            self.state = AgentState.THINKING
            try:
                reply = await self._llm(messages, specs, json_mode=not native, stream=native)
            except Cancelled:
                raise TaskInterrupted(snap(step))
            content = reply["content"]
            self.conversation_history.append({"role": "assistant", "content": content})

            # ---- interpret the reply: -> (calls, final_answer, thought) ---- #
            calls: List[Dict[str, Any]] = []
            answer: Optional[str] = None
            thought = ""
            if native:
                calls = reply["tool_calls"] or parse_tool_calls_from_text(content, set(by_name))
                if calls:
                    thought = content if reply["tool_calls"] else ""
                elif content:
                    answer = content
                else:
                    messages += [{"role": "assistant", "content": ""},
                                 {"role": "user", "content": "You returned nothing. Call a tool or give your final answer."}]
                    continue
            else:
                try:
                    decision = extract_json(content)
                except ValueError as e:
                    messages += [{"role": "assistant", "content": content},
                                 {"role": "user", "content": f"Invalid output ({e}). Reply with ONLY valid JSON."}]
                    continue
                thought = str(decision.get("thought") or "").strip()
                c = _call_from_obj(decision, set())
                if decision.get("action") == "final" or ("answer" in decision and not c):
                    answer = str(decision.get("answer") if decision.get("answer") is not None else "")
                elif c:
                    calls = [c]
                else:
                    messages += [{"role": "assistant", "content": content}, {"role": "user", "content": (
                        f"Your reply {_brief(decision, 300)} had no tool name. Reply with ONLY one of:\n"
                        '{"thought": "...", "action": "tool", "tool": "<name>", "args": {...}}\n'
                        'or {"thought": "...", "action": "final", "answer": "..."}')}]
                    continue

            if thought and not (native and self.on_token):
                self._emit("thought", thought)
            if answer is not None:
                return final(answer, streamed=reply["streamed"])

            # ---- run tool calls (independent ones in parallel) ------------- #
            self.state = AgentState.ACTING
            keys = [f"{c['name']}|{json.dumps(c['arguments'], sort_keys=True, default=str)}" for c in calls]
            todo, dup = [], {}
            for i, (c, k) in enumerate(zip(calls, keys)):
                if k in seen:
                    dup[i] = {"tool": c["name"], "status": "failed",
                              "error": "Duplicate call. Use the earlier result or try something different."}
                else:
                    seen.add(k)
                    todo.append(i)
            ran = await asyncio.gather(*(self._run_call(calls[i], by_name) for i in todo))
            observations = dict(dup)
            observations.update(dict(zip(todo, ran)))
            for i, c in enumerate(calls):
                steps.append({"thought": thought, "tool": c["name"], "args": c["arguments"]})
                if i in todo:
                    tool_results.append(observations[i])

            if native:
                messages.append({"role": "assistant", "content": content, "tool_calls": calls})
                for i, c in enumerate(calls):
                    messages.append({"role": "tool", "name": c["name"], "tool_call_id": c["id"],
                                     "content": json.dumps(observations[i], ensure_ascii=False,
                                                           default=str)[: self.MAX_OBSERVATION_CHARS]})
            else:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Tool result: " + json.dumps(
                    observations[0], ensure_ascii=False, default=str)[: self.MAX_OBSERVATION_CHARS]})

        # Out of steps: ask for the best answer from what we have rather than failing outright.
        if self._cancelled():
            raise TaskInterrupted(snap(self.max_steps))
        messages.append({"role": "user", "content": "You are out of steps. Give your best final answer now from "
                                                    "what you have, in plain text, and say what is missing."})
        reply = await self._llm(messages, None, json_mode=False, stream=native)
        text = reply["content"]
        if not native:
            try:
                text = str(extract_json(text).get("answer", text))
            except ValueError:
                pass
        return final(text or f"No final answer after {self.max_steps} steps", reply["streamed"], incomplete=True)


class PlanExecuteAgent(Agent):
    """Plan -> execute each step -> verify -> re-plan on failure -> reflect -> synthesize."""

    MAX_PLAN_STEPS = 6
    MAX_PRIOR_CHARS = 1500
    MAX_REPLANS = 2
    MAX_STEP_RETRIES = 1
    MAX_REFLECTIONS = 1

    def __init__(self, name: str = "PlanAndExecute", tools: List[Tool] = None, llm: LocalQwenClient = None,
                 verify: bool = None, reflect: bool = None):
        super().__init__(name, "Plans a task, runs each step with tools, verifies, re-plans, then summarizes",
                         tools or [], llm)
        self.executor = ExecutionAgent("StepExecutor", self.tools, self.llm)
        self.verify = (os.getenv("PLAN_VERIFY", "1") != "0") if verify is None else verify
        self.reflect = (os.getenv("PLAN_REFLECT", "1") != "0") if reflect is None else reflect

    # ----- planning -------------------------------------------------------- #
    async def _make_plan(self, task_input: str, done: List[Dict[str, Any]] = None, problem: str = None) -> List[str]:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.tools) or "(none)"
        system = (f"{clock_line()} You are a planner. Break the user's goal into the fewest concrete steps "
                  f"(at most {self.MAX_PLAN_STEPS}). Each step must be doable with these tools or by reasoning "
                  f"alone:\n{tool_lines}\nIf the goal is simple, use just 1 or 2 steps. "
                  'Reply with ONLY JSON: {"steps": ["step 1", "step 2"]}')
        user = task_input
        if done:
            progress = "\n".join(f"- {'OK' if d['ok'] else 'FAILED'} | {d['step']} -> {d['answer'][:400]}" for d in done)
            user = (f"Goal: {task_input}\n\nProgress so far:\n{progress}\n\nProblem: {problem}\n"
                    "Write ONLY the remaining steps (do not repeat finished steps), taking a different approach "
                    "for whatever failed.")
        raw = (await self._llm([{"role": "system", "content": self._with_notes(system)}, *self.history,
                                {"role": "user", "content": user}], role="think", json_mode=True))["content"]
        try:
            steps = [str(s).strip() for s in extract_json(raw).get("steps", []) if str(s).strip()]
        except ValueError:
            steps = []
        return (steps or [task_input])[: self.MAX_PLAN_STEPS]

    # ----- verify / reflect ------------------------------------------------ #
    async def _judge(self, system: str, user: str) -> Dict[str, Any]:
        raw = (await self._llm([{"role": "system", "content": system}, {"role": "user", "content": user}],
                               role="fast", json_mode=True))["content"]
        try:
            return extract_json(raw)
        except ValueError:
            return {}

    async def _verify_step(self, step: str, answer: str, tool_results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        used = "; ".join(f"{t.get('tool')}:{t.get('status')}" for t in tool_results) or "none"
        d = await self._judge(
            "You check whether one step of a plan was actually completed. Reply with ONLY JSON "
            '{"ok": true|false, "reason": "..."}. Use ok=false only if the result is empty, an error, a refusal, '
            "or clearly does not do what the step asked.",
            f"Step: {step}\nTools used: {used}\nResult: {answer[:1500]}")
        return bool(d.get("ok", True)), str(d.get("reason", ""))

    async def _reflect(self, task_input: str, done: List[Dict[str, Any]]) -> Tuple[bool, str]:
        results = "\n".join(f"{i}. {d['step']} -> {d['answer'][:500]}" for i, d in enumerate(done, 1))
        d = await self._judge(
            "You review gathered results against the user's request. Reply with ONLY JSON "
            '{"satisfied": true|false, "missing": "what is missing or wrong"}. satisfied=false only if the '
            "results cannot support a complete, correct answer.",
            f"Request: {task_input}\n\nResults so far:\n{results}")
        return bool(d.get("satisfied", True)), str(d.get("missing", ""))

    # ----- run ------------------------------------------------------------- #
    def _snap(self, task, plan, done, replans, reflections, inner=None) -> Dict[str, Any]:
        return {"kind": "plan_exec", "task": task, "plan": plan, "done": done, "replans": replans,
                "reflections": reflections, "inner": inner}

    async def _run_step(self, task_input, plan, i, done, inner_resume) -> Tuple[bool, str, Dict[str, Any]]:
        step = plan[i]
        prior = "\n".join(f"Step {j} result: {d['answer'][: self.MAX_PRIOR_CHARS]}" for j, d in enumerate(done, 1))
        base = (f"Overall goal: {task_input}\n" + (f"Completed so far:\n{prior}\n" if prior else "")
                + f"\nYour current step ({i + 1}/{len(plan)}): {step}\n"
                  "Complete ONLY this step, then give its result as your final answer.")
        feedback, out, ok, answer = "", {"tool_results": []}, False, ""
        for attempt in range(1 + self.MAX_STEP_RETRIES):
            prompt = base + (f"\n\nYour previous attempt was rejected: {feedback}\nTry a different approach."
                             if feedback else "")
            try:
                out = await self.executor.run_loop(prompt, inner_resume if attempt == 0 else None)
                inner_resume = None
                answer = str(out.get("answer"))
                ok, feedback = (not out.get("incomplete")), ("ran out of steps" if out.get("incomplete") else "")
            except TaskInterrupted:
                raise
            except Cancelled:
                raise
            except Exception as e:
                answer, ok, feedback = f"FAILED: {e}", False, str(e)
                self._emit("result", answer)
            if ok and self.verify:
                ok, feedback = await self._verify_step(step, answer, out.get("tool_results", []))
                if not ok:
                    self._emit("verify", f"step {i + 1} rejected: {feedback}")
            if ok:
                break
        return ok, answer, out

    async def _synthesize(self, task_input: str, done: List[Dict[str, Any]]) -> Dict[str, Any]:
        results_text = "\n".join(f"Step {i} ({d['step']}): {d['answer']}" for i, d in enumerate(done, 1))
        messages = [
            {"role": "system", "content": self._with_notes(
                f"{clock_line()} Write the final answer to the user's request using the step results. "
                "Be accurate and do not invent facts beyond the results. If a step failed, say what could not "
                "be done. Answer directly in plain text, not JSON.")},
            *self.history,
            {"role": "user", "content": f"Request: {task_input}\n\nStep results:\n{results_text}"},
        ]
        return await self._llm(messages, stream=True)

    async def _execute_task_logic(self, task_input: str) -> Any:
        self.executor.share_runtime(self)
        self.executor.on_token = None  # only the final answer streams
        snap = self.resume if (self.resume and self.resume.get("kind") == "plan_exec") else None
        all_tool_results: List[Dict[str, Any]] = []
        plan = (snap["plan"] or None) if snap else None
        done = snap["done"] if snap else []
        replans = snap["replans"] if snap else 0
        reflections = snap["reflections"] if snap else 0
        inner = snap["inner"] if snap else None
        if snap:
            task_input = snap.get("task", task_input)

        try:
            self.state = AgentState.THINKING
            if plan is None:
                plan = await self._make_plan(task_input)
            self._emit("plan", " | ".join(f"{i}. {s}" for i, s in enumerate(plan, 1)))

            i = len(done)
            while True:
                while i < len(plan):
                    self.state = AgentState.ACTING
                    self._emit("step", f"{i + 1}/{len(plan)}: {plan[i]}")
                    ok, answer, out = await self._run_step(task_input, plan, i, done, inner)
                    inner = None
                    all_tool_results.extend(out.get("tool_results", []))
                    done.append({"step": plan[i], "answer": answer, "ok": ok})
                    if not ok and replans < self.MAX_REPLANS:
                        replans += 1
                        self._emit("replan", f"step {i + 1} failed ({replans}/{self.MAX_REPLANS}); re-planning")
                        new = await self._make_plan(task_input, done, f"Step '{plan[i]}' failed: {answer[:300]}")
                        plan = [d["step"] for d in done] + new
                    i = len(done)
                if self.reflect and reflections < self.MAX_REFLECTIONS:
                    reflections += 1
                    satisfied, missing = await self._reflect(task_input, done)
                    if not satisfied:
                        self._emit("reflect", f"not enough yet: {missing}")
                        plan = plan + [f"Fix this gap: {missing}"]
                        continue
                break

            self.state = AgentState.THINKING
            self._emit("step", "writing final answer")
            reply = await self._synthesize(task_input, done)
        except TaskInterrupted as ti:
            raise TaskInterrupted(self._snap(task_input, plan, done, replans, reflections, ti.snapshot))
        except Cancelled:
            raise TaskInterrupted(self._snap(task_input, plan or [], done, replans, reflections, None))

        return {"task": task_input, "plan": plan, "step_results": done, "answer": reply["content"],
                "streamed": reply["streamed"], "executed_tools": len(all_tool_results),
                "tool_results": all_tool_results, "replans": replans}


# --------------------------------------------------------------------------- #
# Task queue
# --------------------------------------------------------------------------- #
class TaskQueue:
    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []

    async def add_task(self, task: Dict[str, Any]) -> str:
        task_id = str(uuid4())
        task["task_id"] = task_id
        task["status"] = TaskStatus.PENDING
        task["created_at"] = time.time()
        self.queue.append(task)
        logger.info(f"Task {task_id} added to queue")
        return task_id

    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        if self.queue:
            task = self.queue.pop(0)
            task["status"] = TaskStatus.IN_PROGRESS
            task["started_at"] = time.time()
            self.active_tasks[task["task_id"]] = task
            return task
        return None

    async def complete_task(self, task_id: str, result: Any, error: str = None):
        task = self.active_tasks.pop(task_id, None)
        if task:
            task["status"] = TaskStatus.FAILED if error else TaskStatus.COMPLETED
            task["result"] = result
            task["completed_at"] = time.time()
            task["error"] = error
            self.completed_tasks.append(task)

    def find(self, task_id_prefix: str) -> Optional[Dict[str, Any]]:
        for pool in (self.queue, list(self.active_tasks.values()), self.completed_tasks):
            for t in pool:
                if t["task_id"].startswith(task_id_prefix):
                    return t
        return None

    def get_task_status(self, task_id: str) -> str:
        t = self.find(task_id)
        return t["status"].value if t else "not_found"


# --------------------------------------------------------------------------- #
# Harness (routing)
# --------------------------------------------------------------------------- #
class AgentHarness:
    def __init__(self, router_llm: LocalQwenClient = None):
        self.agents: Dict[str, Agent] = {}
        self.task_queue = TaskQueue()
        self.running = False
        self.router_llm = router_llm  # if it has a fast model, prompts are classified by it
        self.last_route = ""

    def register_agent(self, agent: Agent) -> str:
        agent_id = str(uuid4())
        self.agents[agent_id] = agent
        logger.info(f"Agent {agent.name} registered with ID {agent_id}")
        return agent_id

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    async def submit_task(self, task_input: str, agent_id: str = None, metadata: Dict[str, Any] = None) -> AgentResponse:
        if agent_id and agent_id in self.agents:
            return await self.agents[agent_id].execute_task(task_input, metadata)
        return await self._dispatch_task_to_appropriate_agent(task_input, metadata)

    def _find_agent(self, cls) -> Optional[Agent]:
        return next((a for a in self.agents.values() if type(a) is cls), None)

    @staticmethod
    def _looks_multistep(lowered: str) -> bool:
        markers = (" then ", "step by step", "research", "compare", "find out", " and then ", "first ")
        return any(m in f" {lowered} " for m in markers)

    def _heuristic_route(self, task_input: str) -> str:
        lowered = task_input.lower()
        if lowered.startswith(("plan", "decide")) or " plan " in f" {lowered} ":
            return "plan"
        return "agent" if self._looks_multistep(lowered) else "exec"

    async def route(self, task_input: str) -> str:
        """chat | plan | exec | agent. Uses the fast model when one is configured."""
        llm = self.router_llm
        if llm is not None and llm.fast_model:
            try:
                raw = await llm.chat([
                    {"role": "system", "content": (
                        "Classify the user's request. Reply with ONLY JSON {\"route\": \"chat\"|\"exec\"|\"agent\"|\"plan\"}. "
                        "chat = small talk or a question answerable from general knowledge; exec = needs one or two "
                        "tools (math, a web page, a file, the date); agent = a multi-step job needing research, "
                        "several files or code; plan = the user only wants a plan.")},
                    {"role": "user", "content": task_input}], role="fast", json_mode=True)
                r = str(extract_json(raw).get("route", "")).lower()
                if r in ("chat", "exec", "agent", "plan"):
                    return r
            except Exception as e:
                logger.warning(f"Router fell back to heuristics: {e}")
        return self._heuristic_route(task_input)

    async def _dispatch_task_to_appropriate_agent(self, task_input: str, metadata: Dict[str, Any]) -> AgentResponse:
        route = await self.route(task_input)
        self.last_route = route
        agent = self._find_agent({"chat": ChatAgent, "plan": PlanningAgent, "exec": ExecutionAgent,
                                  "agent": PlanExecuteAgent}[route]) or self._find_agent(ExecutionAgent)
        if agent is None and self.agents:
            agent = next(iter(self.agents.values()))
        if agent is None:
            raise RuntimeError("No agents available to process task")
        return await agent.execute_task(task_input, metadata)

    async def run(self):
        self.running = True
        while self.running:
            task = await self.task_queue.get_next_task()
            if task:
                try:
                    result = await self.submit_task(task["input"], task.get("agent_id"))
                    await self.task_queue.complete_task(task["task_id"], result)
                except Exception as e:
                    await self.task_queue.complete_task(task["task_id"], None, str(e))
            await asyncio.sleep(0.1)

    def stop(self):
        self.running = False


@dataclass
class AgentSet:
    harness: AgentHarness
    ids: Dict[str, str]          # mode -> agent id
    agents: Dict[str, Agent]     # mode -> agent


def build_agent_set(llm: LocalQwenClient, tools: List[Tool], *, tracer: Tracer = None, on_event=None,
                    on_token=None, confirm: ConfirmFn = None, auto_approve: bool = False,
                    cancel: threading.Event = None) -> AgentSet:
    """A fresh, independent set of agents (background jobs get their own)."""
    harness = AgentHarness(router_llm=llm)
    agents: Dict[str, Agent] = {
        "chat": ChatAgent(llm=llm), "plan": PlanningAgent(tools=tools, llm=llm),
        "exec": ExecutionAgent(tools=tools, llm=llm), "agent": PlanExecuteAgent(tools=tools, llm=llm)}
    ids = {}
    for mode, a in agents.items():
        a.on_event, a.on_token, a.tracer = on_event, on_token, tracer
        a.confirm, a.auto_approve, a.cancel = confirm, auto_approve, cancel
        ids[mode] = harness.register_agent(a)
    return AgentSet(harness, ids, agents)


# --------------------------------------------------------------------------- #
# Background jobs (run on their own agents; read-only tools, no prompts)
# --------------------------------------------------------------------------- #
class BackgroundRunner:
    def __init__(self, make_agents: Callable[[], AgentSet], on_done: Callable[[Dict[str, Any]], None] = None):
        self.make_agents = make_agents
        self.on_done = on_done
        self.queue = TaskQueue()
        self._worker: Optional[asyncio.Task] = None

    async def submit(self, prompt: str, mode: str = "agent") -> str:
        job_id = await self.queue.add_task({"input": prompt, "mode": mode})
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work())
        return job_id[:8]

    async def _work(self) -> None:
        while True:
            task = await self.queue.get_next_task()
            if task is None:
                return
            try:
                aset = self.make_agents()
                resp = await aset.harness.submit_task(task["input"], aset.ids.get(task["mode"]))
                await self.queue.complete_task(task["task_id"], resp, resp.error if resp.status == TaskStatus.FAILED else None)
            except Exception as e:
                await self.queue.complete_task(task["task_id"], None, str(e))
            if self.on_done:
                try:
                    self.on_done(task)
                except Exception:
                    pass

    def jobs(self) -> List[Dict[str, Any]]:
        return [*self.queue.completed_tasks, *self.queue.active_tasks.values(), *self.queue.queue]


# --------------------------------------------------------------------------- #
# Evals: run a fixed set of tasks against the real model and score them
# --------------------------------------------------------------------------- #
def builtin_eval_cases() -> List[Dict[str, Any]]:
    year = str(datetime.now().year)
    return [
        {"name": "date_year", "prompt": "What year is it?", "expect_contains": [year]},
        {"name": "math_simple", "prompt": "What is 1234 * 5678?", "expect_contains": ["7006652", "7,006,652"], "expect_any": True},
        {"name": "math_two_step", "prompt": "Compute (15 + 27) * 3 and then subtract 17.", "expect_contains": ["109"]},
        {"name": "word_count", "prompt": "How many words are in this sentence: 'the quick brown fox jumps over the lazy dog'?",
         "expect_contains": ["9", "nine"], "expect_any": True},
        {"name": "file_roundtrip", "prompt": "Create a file named notes.txt containing exactly: hello eval. Then read it back and confirm.",
         "file_contains": {"notes.txt": "hello eval"}, "expect_tools": ["WriteFile"]},
        {"name": "run_python", "prompt": "Use Python to compute the sum of the squares of 1 through 20 and tell me the result.",
         "expect_contains": ["2870", "2,870"], "expect_any": True, "expect_tools": ["RunPython"]},
        {"name": "search_files", "prompt": "Which file in the workspace mentions pineapple?",
         "seed": {"a.txt": "apple pie", "b.txt": "pineapple pizza"}, "expect_contains": ["b.txt"]},
        {"name": "edit_file", "prompt": "In config.txt change the port from 8080 to 9090.",
         "seed": {"config.txt": "host=localhost\nport=8080\n"}, "file_contains": {"config.txt": "port=9090"}},
        {"name": "remember_recall", "prompt": "Remember that my dog's name is Biscuit, then use Recall to tell me my dog's name.",
         "expect_contains": ["biscuit"], "expect_tools": ["Remember"]},
        {"name": "multi_step_agent", "mode": "agent",
         "prompt": "Create squares.txt in the workspace listing the squares of 1 to 5, one per line, then read it and tell me their sum.",
         "expect_contains": ["55"], "file_contains": {"squares.txt": "25"}},
        {"name": "web_fetch", "network": True, "prompt": "Fetch https://example.com and tell me the page title.",
         "expect_contains": ["example domain"]},
        {"name": "web_search", "network": True, "prompt": "Search the web for the Python programming language and tell me its official website domain.",
         "expect_contains": ["python.org"], "expect_tools": ["WebSearch"]},
    ]


async def run_evals(llm: LocalQwenClient, cases: List[Dict[str, Any]] = None, no_network: bool = False,
                    out_dir: Path = None, echo: Callable[[str], None] = print) -> Dict[str, Any]:
    cases = cases or builtin_eval_cases()
    rows = []
    for case in cases:
        if no_network and case.get("network"):
            rows.append({"name": case["name"], "status": "skipped", "reason": "network"})
            echo(f"  SKIP  {case['name']} (network)")
            continue
        tmp = Path(tempfile.mkdtemp(prefix="agent_eval_"))
        try:
            ws = Workspace(tmp / "ws")
            for fname, content in (case.get("seed") or {}).items():
                (ws.root / fname).write_text(content, encoding="utf-8")
            tools = build_tools(llm, FactStore(tmp / "facts.json"), ws)
            aset = build_agent_set(llm, tools, auto_approve=True, tracer=Tracer(TRACE_DIR / "eval.jsonl"))
            t0 = time.time()
            resp = await aset.harness.submit_task(case["prompt"], aset.ids[case.get("mode", "exec")])
            elapsed = time.time() - t0
            result = resp.result if isinstance(resp.result, dict) else {}
            answer = str(result.get("answer") or "")
            used = [t.get("tool") for t in result.get("tool_results", [])]
            fails: List[str] = []
            if resp.status != TaskStatus.COMPLETED:
                fails.append(f"run {resp.status.value}: {resp.error}")
            exp = [e.lower() for e in case.get("expect_contains", [])]
            if exp:
                hit = [e in answer.lower() for e in exp]
                if not (any(hit) if case.get("expect_any") else all(hit)):
                    fails.append(f"answer missing {exp}: {_brief(answer, 120)!r}")
            for t in case.get("expect_tools", []):
                if t not in used:
                    fails.append(f"tool {t} not used (used: {used})")
            for t in case.get("forbid_tools", []):
                if t in used:
                    fails.append(f"forbidden tool {t} used")
            for fname, needle in (case.get("file_contains") or {}).items():
                try:
                    if needle not in (ws.root / fname).read_text(encoding="utf-8"):
                        fails.append(f"{fname} lacks {needle!r}")
                except OSError:
                    fails.append(f"{fname} was not created")
            if case.get("max_seconds") and elapsed > case["max_seconds"]:
                fails.append(f"too slow ({elapsed:.0f}s)")
            rows.append({"name": case["name"], "status": "fail" if fails else "pass", "seconds": round(elapsed, 1),
                         "tools": used, "failures": fails, "answer": answer[:300]})
            echo(f"  {'FAIL' if fails else 'PASS'}  {case['name']:<18} {elapsed:5.1f}s  tools={used}"
                 + (f"\n        {'; '.join(fails)}" if fails else ""))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    ran = [r for r in rows if r["status"] != "skipped"]
    passed = sum(r["status"] == "pass" for r in ran)
    summary = {"model": llm.model, "native_tools": llm.native_tools, "passed": passed, "ran": len(ran),
               "skipped": len(rows) - len(ran), "rows": rows, "timestamp": datetime.now().isoformat(timespec="seconds")}
    echo(f"\n  {passed}/{len(ran)} passed" + (f" ({summary['skipped']} skipped)" if summary["skipped"] else ""))
    out_dir = Path(out_dir or BASE_DIR / "evals")
    try:
        _atomic_write(out_dir / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json", summary)
    except OSError:
        pass
    return summary


# --------------------------------------------------------------------------- #
# Interactive mode
# --------------------------------------------------------------------------- #
HELP_TEXT = """
Type a prompt and press Enter. Ctrl+C while the agent is working stops it at the next safe point
(use /resume to continue); Ctrl+C at the prompt exits.

Modes (/mode <n>):
  auto  - the harness routes each prompt to chat / plan / exec / agent (default)
  chat  - plain conversation          plan - just make a plan
  exec  - tool-using agent            agent - plan, execute, verify, re-plan, reflect, summarize

Commands:
  /mode <auto|chat|plan|exec|agent>      /resume            continue an interrupted task
  /bg <prompt>                           run in the background (read-only tools, no prompts)
  /jobs   /job <id>                      list / show background jobs
  /remember <fact>  /facts  /forget <id> saved facts shared by every session
  /feedback <correction>  /learn [list|stats|verify ID|reject ID|export] learning from mistakes
  /summary                               show the rolling summary of this session
  /history [n]  /reset  /session <n>  /sessions
  /tools  /mcp  /workspace               tools, MCP servers, sandbox folder
  /approve [on|off]                      auto-approve file writes / code execution (be careful)
  /trace [n]                             last n trace events    /help  /quit
"""


class Printer:
    """Streams tokens and keeps progress lines from colliding with them."""

    def __init__(self):
        self.started = False

    def token(self, t: str) -> None:
        if not self.started:
            print("\n", end="")
            self.started = True
        print(t, end="", flush=True)

    def end_line(self) -> None:
        if self.started:
            print()
            self.started = False


def _response_to_text(resp: AgentResponse) -> Optional[str]:
    """Turn an agent result into text worth remembering (None if it failed)."""
    if resp.status in (TaskStatus.FAILED, TaskStatus.INTERRUPTED):
        return None
    result = resp.result
    if isinstance(result, dict) and "planning_steps" in result:
        return "Plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(result["planning_steps"], 1))
    if isinstance(result, dict) and "answer" in result:
        text = str(result["answer"])
        used = [f"{t.get('tool')} -> {_brief(t.get('result'), 150)}" for t in result.get("tool_results", [])
                if t.get("status") == "success"]
        if used:
            text += "\n(tools used: " + "; ".join(used) + ")"
        return text
    return json.dumps(result, default=str)


def _print_response(resp: AgentResponse) -> None:
    if resp.status == TaskStatus.INTERRUPTED:
        print("\n[interrupted] stopped. Type /resume to continue where it left off.\n")
        return
    if resp.status == TaskStatus.FAILED:
        print(f"\n[error] {resp.error}\n")
        return
    result = resp.result
    if isinstance(result, dict) and "planning_steps" in result:
        print("\nPlan:")
        for i, step in enumerate(result["planning_steps"], 1):
            print(f"  {i}. {step}")
    elif isinstance(result, dict) and "answer" in result:
        print() if result.get("streamed") else print(f"\n{result['answer']}")
        if result.get("incomplete"):
            print("\n(note: hit the step limit; the answer may be incomplete)")
    else:
        print("\n" + json.dumps(result, indent=2, default=str))
    print(f"\n({resp.execution_time:.1f}s)\n")


def _install_sigint(cancel: threading.Event):
    def handler(signum, frame):
        cancel.set()
        print("\n  [interrupt] stopping at the next safe point...")
    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        return None


def _restore_sigint(old) -> None:
    if old is not None:
        try:
            signal.signal(signal.SIGINT, old)
        except ValueError:
            pass


async def interactive(session: str = "default", directory: str = None, auto_approve: bool = False,
                      workspace: str = None, no_mcp: bool = False) -> None:
    logging.getLogger().setLevel(logging.WARNING)  # keep the prompt clean
    llm = LocalQwenClient()
    ws = Workspace(workspace)
    facts = FactStore()
    mcp_servers, mcp_tools = ([], []) if no_mcp else await asyncio.to_thread(load_mcp)
    tools = build_tools(llm, facts, ws, mcp_tools)
    ro_tools = build_tools(llm, facts, ws, mcp_tools, read_only=True)
    cancel = threading.Event()
    printer = Printer()
    safe_session = re.sub(r"[^\w\-]", "_", session) or "default"
    tracer = Tracer(TRACE_DIR / f"{safe_session}.jsonl")
    approved_always: set = set()
    confirm_lock = asyncio.Lock()
    state = {"auto_approve": auto_approve}

    def show(kind: str, text: str) -> None:  # live view of the agent's progress
        printer.end_line()
        text = " ".join(str(text).split())
        print(f"  [{kind}] {text[:300]}{'...' if len(text) > 300 else ''}")

    async def confirm(tool_name: str, description: str) -> bool:
        async with confirm_lock:
            if tool_name in approved_always:
                return True
            if cancel.is_set():
                return False
            printer.end_line()
            print(f"\n  [confirm] {description}")
            ans = (await asyncio.to_thread(
                input, "  Allow? [y]es / [n]o / [a]lways this tool this session: ")).strip().lower()
            if ans.startswith("a"):
                approved_always.add(tool_name)
                return True
            return ans.startswith("y")

    def make_main() -> AgentSet:
        return build_agent_set(llm, tools, tracer=tracer, on_event=show, on_token=printer.token,
                               confirm=confirm, auto_approve=state["auto_approve"], cancel=cancel)

    main = make_main()
    harness = main.harness

    def job_done(task: Dict[str, Any]) -> None:
        printer.end_line()
        print(f"\n  [job {task['task_id'][:8]} {task['status'].value}]  /job {task['task_id'][:8]} to view")

    bg = BackgroundRunner(lambda: build_agent_set(llm, ro_tools, tracer=tracer), job_done)

    memory = ConversationMemory(session, directory)
    learning = LearningStore()
    last_turn: Dict[str, str] = {"task": "", "answer": ""}
    mode = "auto"

    print(f"Model: {llm.model} ({llm.backend} @ {llm.base_url}, context {llm.num_ctx} tokens)")
    extras = [f"fast={llm.fast_model}" if llm.fast_model else "", f"think={llm.think_model}" if llm.think_model else ""]
    print(f"Workspace: {ws.root}   Tools: {len(tools)}" + (f"   MCP servers: {len(mcp_servers)}" if mcp_servers else "")
          + "   " + " ".join(e for e in extras if e))
    print(f"Session '{memory.session}': " + (f"restored {len(memory.messages)} earlier messages." if memory.messages
                                             else "new conversation."))
    if memory.load_checkpoint():
        print("An interrupted task is saved. Type /resume to continue it.")
    print(f"Learning: {learning.path} | verified lessons: {learning.stats()['verified']}")
    print("Type /help for commands, /quit to exit.")

    async def run_prompt(line: str, run_mode: str, snapshot: Dict[str, Any] = None) -> None:
        await memory.maybe_summarize(llm)
        notes = await memory.build_notes(llm, facts, line)
        lesson_hits = learning.search(line, k=3)
        if lesson_hits:
            lesson_text = "\n".join(
                f"- Lesson {it['id']} (confidence {it['confidence']:.2f}): "
                f"{(it['lesson'] or it['correction'])[:500]}" for it in lesson_hits)
            notes = (notes + "\n\nVerified learning lessons (use when relevant; still check evidence):\n" + lesson_text).strip()
        eff = run_mode
        if eff == "auto":
            eff = await harness.route(line)
            print(f"  [route] {eff}")
        meta: Dict[str, Any] = {"history": memory.recent(), "notes": notes}
        if snapshot:
            meta["resume"] = snapshot
        cancel.clear()
        printer.started = False
        old = _install_sigint(cancel)
        try:
            resp = await harness.submit_task(line, main.ids[eff], meta)
        finally:
            _restore_sigint(old)
        printer.end_line()
        _print_response(resp)
        if resp.status == TaskStatus.INTERRUPTED:
            memory.save_checkpoint({"mode": eff, "task": line, "snapshot": resp.metadata.get("snapshot")})
            return
        memory.clear_checkpoint()
        response_text = _response_to_text(resp) or (resp.error or "")
        last_turn["task"], last_turn["answer"] = line, response_text
        text = _response_to_text(resp)
        if text:  # only remember turns that succeeded
            memory.add("user", line)
            memory.add("assistant", text)

        # Automatically capture execution failures. This path runs only after a failure,
        # so ordinary successful responses pay no extra model-call latency.
        failures: List[str] = []
        def collect_failures(obj: Any) -> None:
            if isinstance(obj, dict):
                if obj.get("status") == "failed" and obj.get("tool"):
                    err = str(obj.get("error") or obj.get("result") or "tool failed")
                    if "user denied this action" not in err.lower():
                        failures.append(f"{obj.get('tool')}: {err[:700]}")
                for value in obj.values():
                    collect_failures(value)
            elif isinstance(obj, list):
                for value in obj:
                    collect_failures(value)
        collect_failures(resp.result)
        if resp.status == TaskStatus.FAILED and resp.error:
            failures.append(f"Agent execution: {resp.error[:1000]}")
        failures = list(dict.fromkeys(failures))[:5]
        if failures:
            failure_text = "\n".join(failures)
            proposed = ("When this task pattern occurs, inspect the recorded error and tool schema before retrying; "
                        "do not repeat an identical failed call without changing the cause. Evidence: " + failure_text)
            lesson_id = learning.add(source="automatic_failure", task=line, failure=failure_text,
                                     lesson=proposed, status="uncertain", evidence="Captured from runtime result")
            review = await learning.review(llm, lesson_id)
            print(f"  [learning] captured failure #{lesson_id}: {review['status']} "
                  f"(review confidence {review['confidence']:.2f})")

    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, f"[{mode}] > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue

            if line.startswith("/"):
                cmd, _, arg = line.partition(" ")
                cmd, arg = cmd.lower(), arg.strip()
                if cmd in ("/quit", "/exit", "/q"):
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/tools":
                    for t in tools:
                        print(f"  {t.name}{' (asks first)' if t.dangerous else ''}: {t.description}")
                elif cmd == "/mcp":
                    if not mcp_servers:
                        print("  No MCP servers. Create mcp.json next to the script, e.g.\n"
                              '  {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}')
                    for s in mcp_servers:
                        print(f"  {s.name}: {', '.join(t['name'] for t in s.tools)}")
                elif cmd == "/workspace":
                    print(f"  {ws.root}")
                elif cmd == "/reset":
                    memory.clear()
                    print(f"History for session '{memory.session}' erased.")
                elif cmd == "/sessions":
                    names = ConversationMemory.list_sessions(directory)
                    print("Saved sessions: " + (", ".join(names) if names else "(none yet)"))
                elif cmd == "/session":
                    if arg:
                        memory = ConversationMemory(arg, directory)
                        print(f"Switched to session '{memory.session}' ({len(memory.messages)} messages).")
                    else:
                        print(f"Current session: {memory.session}. Usage: /session <n>")
                elif cmd == "/summary":
                    print(memory.summary or "(no summary yet; it builds up once the conversation outgrows the window)")
                elif cmd == "/history":
                    n = int(arg) if arg.isdigit() else 10
                    msgs = memory.messages[-n:]
                    if not msgs:
                        print("(no history yet)")
                    for m in msgs:
                        text = m["content"].replace("\n", " ")
                        print(f"  {m['role']:>9}: {text[:200]}{'...' if len(text) > 200 else ''}")
                elif cmd == "/facts":
                    if not facts.items:
                        print("  (no saved facts)")
                    for it in facts.items:
                        print(f"  {it['id']}  {it['text']}")
                elif cmd == "/remember":
                    print(f"  saved as {facts.add(arg)}" if arg else "Usage: /remember <fact>")
                elif cmd == "/forget":
                    print("  forgotten" if facts.remove(arg) else "  no such fact id")
                elif cmd == "/feedback":
                    if not arg:
                        print("Usage: /feedback <correct answer or reusable lesson>")
                    elif not last_turn.get("task"):
                        print("  No completed turn to attach feedback to yet.")
                    else:
                        lesson_id = learning.add(
                            source="user_feedback", task=last_turn["task"],
                            failure=last_turn.get("answer", ""), correction=arg,
                            lesson=arg, status="uncertain", evidence="Explicit user-provided correction")
                        print(f"  captured correction as learning record #{lesson_id}; checking consistency...")
                        review = await learning.review(llm, lesson_id)
                        print(f"  review: {review['status']} (confidence {review['confidence']:.2f}) — {review['reason']}")
                        if review["status"] != "verified":
                            print("  This lesson will not be injected into future prompts until verified.")
                elif cmd == "/learn":
                    sub, _, value = arg.partition(" ")
                    sub = sub.lower() or "list"
                    if sub == "stats":
                        stats = learning.stats()
                        print("  Learning records: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
                    elif sub == "list":
                        items = learning.list_recent(20)
                        if not items:
                            print("  No learning records yet. Use /feedback after a correction.")
                        for it in items:
                            summary = it["lesson"] or it["correction"] or it["failure"]
                            print(f"  #{it['id']} [{it['status']}] {it['source']}: {_brief(summary, 180)}")
                    elif sub in ("verify", "reject"):
                        try:
                            lesson_id = int(value.strip())
                        except ValueError:
                            print(f"Usage: /learn {sub} ID")
                        else:
                            status = "verified" if sub == "verify" else "rejected"
                            if learning.set_status(lesson_id, status, verifier="explicit user decision"):
                                print(f"  learning record #{lesson_id} marked {status}.")
                            else:
                                print(f"  no learning record #{lesson_id}.")
                    elif sub == "export":
                        path, count = learning.export_training()
                        print(f"  exported {count} verified correction examples to {path}")
                    else:
                        print("Usage: /learn [list|stats|verify ID|reject ID|export]")
                elif cmd == "/approve":
                    state["auto_approve"] = (arg.lower() != "off")
                    for a in main.agents.values():
                        a.auto_approve = state["auto_approve"]
                    print(f"  auto-approve {'ON: writes and code runs will NOT ask first' if state['auto_approve'] else 'off'}")
                elif cmd == "/bg":
                    if not arg:
                        print("Usage: /bg <prompt>")
                    else:
                        jid = await bg.submit(arg, "agent")
                        print(f"  job {jid} started (read-only tools; it can't ask for approval)")
                elif cmd == "/jobs":
                    jobs = bg.jobs()
                    if not jobs:
                        print("  (no jobs)")
                    for j in jobs:
                        print(f"  {j['task_id'][:8]}  {j['status'].value:<11} {_brief(j['input'], 70)}")
                elif cmd == "/job":
                    j = bg.queue.find(arg) if arg else None
                    if not j:
                        print("  no such job")
                    elif j["status"] in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                        print(f"  {j['status'].value}...")
                    elif j.get("result") is not None:
                        _print_response(j["result"])
                    else:
                        print(f"  failed: {j.get('error')}")
                elif cmd == "/trace":
                    n = int(arg) if arg.isdigit() else 15
                    try:
                        for ln in tracer.path.read_text(encoding="utf-8").splitlines()[-n:]:
                            r = json.loads(ln)
                            print(f"  {r.get('event'):<14} {_brief({k: v for k, v in r.items() if k not in ('ts', 'event', 'task')}, 160)}")
                    except OSError:
                        print("  (no trace yet)")
                elif cmd == "/resume":
                    ck = memory.load_checkpoint()
                    if not ck:
                        print("  nothing to resume")
                    else:
                        try:
                            await run_prompt(ck["task"], ck["mode"], ck.get("snapshot"))
                        except Exception as e:
                            print(f"\n[error] {e}\n")
                elif cmd == "/mode":
                    if arg.lower() in ("auto", "chat", "plan", "exec", "agent"):
                        mode = arg.lower()
                    else:
                        print("Usage: /mode <auto|chat|plan|exec|agent>")
                else:
                    print("Unknown command. Type /help.")
                continue

            try:
                print("thinking...")
                await run_prompt(line, mode)
            except Exception as e:
                printer.end_line()
                print(f"\n[error] {e}\n")
    finally:
        for s in mcp_servers:
            s.close()
    print("Bye!")


# --------------------------------------------------------------------------- #
# Scripted demo (--demo)
# --------------------------------------------------------------------------- #
async def main():
    llm = LocalQwenClient()
    print(f"Using {llm.backend} backend, model={llm.model}, url={llm.base_url}")
    ws = Workspace()
    tools = build_tools(llm, FactStore(), ws, read_only=True)
    aset = build_agent_set(llm, tools, on_event=lambda k, t: print(f"  [{k}] {_brief(t, 160)}"))
    print("=== Testing Agent Harness ===")
    for label, mode, prompt in [("Planning", "plan", "Plan a strategy to optimize a marketing campaign"),
                                ("Text", "exec", "Count words in 'Hello world, this is a test'"),
                                ("Math", "exec", "Calculate 25 * 4 + 10"),
                                ("Date", "exec", "What year is it?")]:
        r = await aset.harness.submit_task(prompt, aset.ids[mode])
        print(f"{label} response: {r.result or r.error}")
    print("=== Harness operation completed ===")



# --------------------------------------------------------------------------- #
# Local browser chat UI (--web; default launch mode in this edition)
# --------------------------------------------------------------------------- #
WEB_CHAT_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Workbench</title>
<style>
:root{color-scheme:dark;--bg:#080c14;--side:#0d1420;--panel:#111b2a;--line:#26364a;--text:#edf4ff;--muted:#91a4bd;--cyan:#65e6d4;--violet:#a99bff;--red:#ff8d9a;--green:#6de2a8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,Segoe UI,sans-serif;height:100vh;display:grid;grid-template-columns:265px minmax(0,1fr)}button,textarea{font:inherit}.side{background:var(--side);border-right:1px solid var(--line);padding:20px 14px;display:flex;flex-direction:column;gap:18px;min-height:0}.brand{display:flex;gap:10px;align-items:center;font-weight:750;font-size:17px}.mark{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(135deg,var(--cyan),var(--violet));color:#071018;font-weight:900}.sub{font-size:11px;color:var(--muted);margin-top:3px}.new{width:100%;padding:10px;border:1px solid #34475f;border-radius:10px;background:#152234;color:var(--text);cursor:pointer}.section{border-top:1px solid var(--line);padding-top:14px}.label{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#7f96b2;font-weight:800;margin-bottom:8px}.stat{background:#111d2c;border:1px solid var(--line);border-radius:10px;padding:10px;margin:6px 0}.stat b{display:block;font-size:12px}.stat span{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.command{display:flex;gap:8px;align-items:baseline;font-size:12px;margin:7px 0;color:var(--muted)}.command code{color:var(--cyan)}.sidefoot{margin-top:auto;font-size:11px;color:var(--muted)}main{min-width:0;display:flex;flex-direction:column;height:100vh}.top{padding:15px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:10px;background:#0b111c}.top strong{font-size:13px}.status{font-size:11px;color:var(--muted);display:flex;align-items:center;gap:7px}.dot{width:8px;height:8px;border-radius:50%;background:#e6b65f}.dot.ok{background:var(--green)}.dot.err{background:var(--red)}.tag{border:1px solid var(--line);border-radius:99px;padding:4px 9px;color:var(--muted)}.chat{width:min(940px,100%);margin:0 auto;flex:1;min-height:0;display:flex;flex-direction:column;padding:20px 22px 0}.messages{flex:1;overflow:auto;display:flex;flex-direction:column;gap:17px;padding:8px 2px 20px}.welcome{margin:auto;max-width:610px;text-align:center}.welcome .hero{width:58px;height:58px;margin:0 auto 15px;border-radius:19px;background:linear-gradient(135deg,#163b46,#282044);border:1px solid #35546a;display:grid;place-items:center;font-size:25px;color:var(--cyan)}h1{font-size:26px;letter-spacing:-.04em;margin:0 0 8px}.welcome p{color:var(--muted);margin:0 0 20px}.chips{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}.chip{border:1px solid var(--line);background:#101a28;color:#c8d7e9;border-radius:9px;padding:8px 10px;cursor:pointer}.msg{max-width:88%;display:flex;flex-direction:column;gap:5px}.msg.user{align-self:flex-end}.msg.assistant{align-self:flex-start}.who{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#7f96b2}.bubble{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px 15px;border-radius:13px;border:1px solid var(--line);background:#111b29}.user .bubble{background:#172a40;border-color:#2b4c67}.composer-wrap{padding:12px 0 18px}.error{color:var(--red);padding:8px 0;font-size:12px}.composer{display:flex;align-items:flex-end;gap:10px;border:1px solid #31475c;background:#0e1825;padding:9px;border-radius:15px}.composer:focus-within{border-color:#5a9b9d;box-shadow:0 0 0 3px #65e6d414}.composer textarea{flex:1;resize:none;max-height:180px;min-height:42px;border:0;outline:0;background:transparent;color:var(--text);padding:9px 7px}.send{border:0;border-radius:10px;background:var(--cyan);color:#061318;font-weight:800;padding:10px 14px;cursor:pointer}.send:disabled{opacity:.45}.under{display:flex;justify-content:space-between;gap:8px;padding:7px 4px;color:var(--muted);font-size:10px}.mini{font-size:11px;color:var(--muted);white-space:pre-wrap;overflow-wrap:anywhere}.activitybar{height:3px;background:#1c2938}.activitybar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width .3s}@media(max-width:760px){body{grid-template-columns:1fr}.side{display:none}.chat{padding:12px 12px 0}.top{padding:12px}.msg{max-width:96%}}
</style></head><body>
<aside class="side"><div class="brand"><div class="mark">Q</div><div>Qwen Workbench<div class="sub">LOCAL MODEL CONTROL</div></div></div><button class="new" id="clear">＋ New conversation</button>
<div class="section"><div class="label">Live system</div><div class="stat"><b id="clock">Reading system clock…</b><span id="zone">Timezone loading</span></div><div class="stat"><b id="statusText">Connecting to Ollama…</b><span id="model">Model: checking</span></div><div class="stat"><b>Computer resources</b><span id="resources">Waiting for telemetry…</span></div><div class="stat"><b>GPU / accelerator</b><span id="gpu">Checking GPU availability…</span></div></div>
<div class="section"><div class="label">Commands</div><div class="command"><code>/help</code><span>browser commands</span></div><div class="command"><code>/time</code><span>read live local clock</span></div><div class="command"><code>/resources</code><span>refresh machine stats</span></div><div class="command"><code>/harness</code><span>inspect this agent's code</span></div><div class="command"><code>/clear</code><span>new conversation</span></div><div class="mini" id="cliCommands"></div></div>
<div class="sidefoot">Local UI → Python agent → Ollama<br>Private internal chain-of-thought is not exposed; activity and source inspection are.</div></aside>
<main><div class="top"><strong>Chat / Agent workspace</strong><div class="status"><span id="dot" class="dot"></span><span id="status">Checking connection</span><span class="tag" id="thinking">Thinking: on</span></div></div><div class="activitybar"><i id="progress"></i></div><section class="chat"><section id="messages" class="messages"><div class="welcome" id="welcome"><div class="hero">✦</div><h1>What are we building today?</h1><p>This workspace talks to your configured local Qwen model. Ask it to inspect the Python harness, check the machine, or use the live clock.</p><div class="chips"><button class="chip" data-prompt="What time is it right now?">Read current time</button><button class="chip" data-prompt="Inspect your own Python model harness and explain how it works.">Inspect model harness</button><button class="chip" data-prompt="Show available commands and system resources.">Show commands & resources</button></div></div></section><div class="composer-wrap"><div class="error" id="error" hidden></div><form class="composer" id="form"><textarea id="input" rows="1" placeholder="Message Qwen… (Enter to send, Shift+Enter for newline)" autofocus></textarea><button class="send" id="send" type="submit">Send ↗</button></form><div class="under"><span id="activity">Ready when you are</span><span>Responses generated by the local model</span></div></div></section></main>
<script>
const $=id=>document.getElementById(id);let history=[];let busy=false;let started=0;
function status(s,k=''){$('status').textContent=s;$('dot').className='dot '+k;$('statusText').textContent=s}
async function clock(){try{const r=await fetch('/api/time?fresh='+Date.now(),{cache:'no-store'});const d=await r.json();$('clock').textContent=d.time+' '+d.timezone;$('zone').textContent=d.weekday+', '+d.date+' · '+d.timezone_name+' · UTC'+d.utc_offset;}catch(e){$('clock').textContent='Clock unavailable';$('zone').textContent=e.message}}
async function telemetry(){try{const r=await fetch('/api/resources?fresh='+Date.now(),{cache:'no-store'}),d=await r.json();$('resources').textContent='CPU '+(d.cpu_percent==null?'n/a':d.cpu_percent+'%')+' · RAM '+(d.ram_used_gb==null?'n/a':d.ram_used_gb+'/'+d.ram_total_gb+' GB')+' · Python '+(d.process_rss_mb==null?'n/a':d.process_rss_mb+' MB');$('gpu').textContent=(d.gpu_name||'No NVIDIA GPU detected')+' · '+(d.gpu_detail||'telemetry unavailable')}catch(e){$('resources').textContent='Unavailable: '+e.message}}
async function health(){try{const r=await fetch('/api/health?fresh='+Date.now(),{cache:'no-store'}),d=await r.json();if(!r.ok)throw Error(d.error||'Ollama unavailable');status('Model connected','ok');$('model').textContent=d.model+' · '+d.base_url;$('thinking').textContent='Thinking: '+(d.thinking_enabled?'on':'off');$('cliCommands').textContent='CLI commands:\n'+(d.cli_commands||'Run --cli and use /help')}catch(e){status('Model offline','err');$('model').textContent=e.message;$('error').hidden=false;$('error').textContent=e.message}}
function add(role,text){$('welcome')?.remove();let w=document.createElement('div');w.className='msg '+role;let who=document.createElement('div');who.className='who';who.textContent=role==='user'?'You':'Qwen · local model';let b=document.createElement('div');b.className='bubble';b.textContent=text;w.append(who,b);$('messages').append(w);$('messages').scrollTop=$('messages').scrollHeight;return b}
async function sendText(raw){const text=raw.trim();if(!text||busy)return;$('error').hidden=true;if(/^\/(time|clock)$/i.test(text)){add('user',text);await clock();add('assistant',$('clock').textContent+'\n'+$('zone').textContent);return}if(/^\/(help|commands)$/i.test(text)){add('user',text);add('assistant','Browser commands:\n/help or /commands — list commands\n/time or /clock — live time from Python\n/resources — machine telemetry\n/harness — inspect the Python agent harness\n/clear — new conversation\n\nCLI-only commands:\n'+$('cliCommands').textContent);return}if(/^\/resources$/i.test(text)){add('user',text);await telemetry();add('assistant',$('resources').textContent+'\nGPU: '+$('gpu').textContent);return}if(/^\/harness$/i.test(text)){add('user',text);return await askModel('Inspect your own Python model harness and explain its architecture, tools, model settings, and limitations. Use the source context provided by the server.');}if(/^\/clear$/i.test(text)){history=[];$('messages').innerHTML='<div class="welcome" id="welcome"><div class="hero">✦</div><h1>What are we building today?</h1><p>This workspace talks to your configured local Qwen model.</p><div class="chips"><button class="chip" data-prompt="What time is it right now?">Read current time</button><button class="chip" data-prompt="Inspect your own Python model harness and explain how it works.">Inspect model harness</button></div></div>';return}return askModel(text)}
async function askModel(text){add('user',text);history.push({role:'user',content:text});busy=true;started=Date.now();$('send').disabled=true;$('input').disabled=true;$('send').textContent='Working…';$('activity').textContent='Local model is processing your request…';$('progress').style.width='70%';const bubble=add('assistant','Connecting to local model…');try{const r=await fetch('/api/chat',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});const d=await r.json();if(!r.ok)throw Error(d.error||'Model request failed');bubble.textContent=d.reply||'(Empty model response)';history.push({role:'assistant',content:d.reply||''});$('activity').textContent='Completed in '+((Date.now()-started)/1000).toFixed(1)+'s · '+(d.model||'Qwen')+' · '+(d.thinking_enabled?'thinking enabled':'thinking disabled');$('thinking').textContent='Thinking: '+(d.thinking_enabled?'on':'off');await clock()}catch(e){bubble.textContent='Model error: '+e.message;$('error').hidden=false;$('error').textContent=e.message;history.pop()}finally{busy=false;$('send').disabled=false;$('input').disabled=false;$('send').textContent='Send ↗';$('progress').style.width='0';$('messages').scrollTop=$('messages').scrollHeight;$('input').focus()}}
$('form').addEventListener('submit',e=>{e.preventDefault();const t=$('input').value;$('input').value='';$('input').style.height='auto';sendText(t)});$('input').addEventListener('input',()=>{$('input').style.height='auto';$('input').style.height=Math.min($('input').scrollHeight,180)+'px'});$('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('form').requestSubmit()}});$('clear').addEventListener('click',()=>sendText('/clear'));document.addEventListener('click',e=>{if(e.target.matches('[data-prompt]'))sendText(e.target.dataset.prompt)});clock();telemetry();health();setInterval(clock,1000);setInterval(telemetry,2500);setInterval(health,15000);
</script></body></html>"""


def _local_resource_snapshot() -> Dict[str, Any]:
    """Best-effort local CPU/RAM/process/GPU telemetry; no third-party package required."""
    import platform

    result: Dict[str, Any] = {"platform": platform.system(), "cpu_percent": None,
                              "ram_used_gb": None, "ram_total_gb": None, "ram_percent": None,
                              "process_rss_mb": None, "gpu_name": None, "gpu_detail": None}
    # psutil gives accurate cross-platform readings when installed.
    try:
        import psutil  # type: ignore
        result["cpu_percent"] = round(psutil.cpu_percent(interval=0.05), 1)
        vm = psutil.virtual_memory()
        result["ram_used_gb"] = round(vm.used / (1024 ** 3), 1)
        result["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
        result["ram_percent"] = round(vm.percent, 1)
        result["process_rss_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
    except Exception:
        # Lightweight fallbacks for systems without psutil.
        try:
            if sys.platform == "win32":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                m = MEMORYSTATUSEX()
                m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                    total = m.ullTotalPhys
                    used = total - m.ullAvailPhys
                    result.update(ram_used_gb=round(used / (1024 ** 3), 1),
                                  ram_total_gb=round(total / (1024 ** 3), 1),
                                  ram_percent=round(m.dwMemoryLoad, 1))
            elif sys.platform.startswith("linux"):
                vals = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        k, v = line.split(":", 1)
                        vals[k] = int(v.strip().split()[0]) * 1024
                total = vals.get("MemTotal", 0)
                avail = vals.get("MemAvailable", 0)
                if total:
                    used = total - avail
                    result.update(ram_used_gb=round(used / (1024 ** 3), 1),
                                  ram_total_gb=round(total / (1024 ** 3), 1),
                                  ram_percent=round(100 * used / total, 1))
        except Exception:
            pass
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=1.5)
        if p.returncode == 0 and p.stdout.strip():
            first = p.stdout.strip().splitlines()[0]
            parts = [x.strip() for x in first.split(",")]
            if len(parts) >= 4:
                result["gpu_name"] = parts[0]
                result["gpu_detail"] = f"GPU utilization {parts[1]}% · VRAM {parts[2]} / {parts[3]} MiB"
    except Exception:
        pass
    if not result["gpu_detail"]:
        result["gpu_detail"] = "nvidia-smi not found or GPU telemetry unavailable. Ollama may run in Docker or use another backend."
    return result


def start_web_ui(host: str = "127.0.0.1", port: int = 8001, open_browser: bool = True) -> None:
    """Serve a local browser chat UI that calls the configured Ollama model."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    llm = LocalQwenClient(think=os.getenv("QWEN_THINK", "1").strip().lower() not in ("0", "false", "off", "no"))
    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "LocalQwenWeb/1.0"
        def log_message(self, fmt, *args):
            logger.info("web %s - %s", self.address_string(), fmt % args)

        def _json(self, status: int, payload: Dict[str, Any]):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                raw = WEB_CHAT_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/api/health":
                try:
                    with urllib.request.urlopen(f"{llm.base_url}/api/tags", timeout=4) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    self._json(200, {"ok": True, "model": llm.model, "base_url": llm.base_url, "available_models": models, "thinking_enabled": llm.think, "cli_commands": HELP_TEXT.strip()})
                except Exception as exc:
                    self._json(503, {"ok": False, "model": llm.model, "base_url": llm.base_url,
                                     "error": f"Cannot reach Ollama at {llm.base_url}: {exc}"})
            elif self.path == "/api/resources":
                self._json(200, _local_resource_snapshot())
            elif self.path.startswith("/api/time"):
                self._json(200, system_clock())
            elif self.path == "/api/harness":
                try:
                    source_path = os.path.abspath(__file__)
                    with open(source_path, "r", encoding="utf-8") as source_file:
                        source_text = source_file.read()
                    self._json(200, {"path": source_path, "characters": len(source_text), "source": source_text[:60000]})
                except Exception as exc:
                    self._json(500, {"error": f"Cannot read harness source: {exc}"})
            elif self.path == "/api/commands":
                self._json(200, {"browser_commands": [
                    {"command": "/help", "description": "Show commands supported in the browser chat"},
                    {"command": "/time", "description": "Show the current local date, time, and timezone"},
                    {"command": "/resources", "description": "Refresh CPU, RAM, process, and GPU readings"},
                    {"command": "/clear", "description": "Start a fresh conversation"}],
                    "cli_commands": HELP_TEXT.strip()})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    self._json(413, {"error": "Request body is empty or too large (1 MB limit)."})
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                incoming = body.get("messages", [])
                if not isinstance(incoming, list) or not incoming:
                    self._json(400, {"error": "Send a non-empty messages array."})
                    return
                # Only accept simple chat roles; never accept tool calls from the browser.
                messages = []
                for item in incoming[-40:]:
                    if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                        continue
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        messages.append({"role": item["role"], "content": content[:20000]})
                if not messages or messages[-1]["role"] != "user":
                    self._json(400, {"error": "The last message must be a user message."})
                    return
                # Answer direct clock/date questions in Python instead of relying on
                # the model to obey the timestamp. This makes time answers deterministic.
                latest_text = messages[-1]["content"].strip()
                time_question = re.search(
                    r"(?:\bwhat(?:'s| is)\s+(?:the\s+)?time\b|\bwhat\s+time\s+is\s+it\b|\bcurrent\s+(?:local\s+)?time\b|\btime\s+right\s+now\b|\b(?:today's\s+date|what\s+date\s+is\s+it|current\s+date|what\s+day\s+is\s+it)\b|^/time$|^/clock$)",
                    latest_text, re.IGNORECASE,
                )
                if time_question:
                    c = system_clock()
                    reply = (f"It is {c['time']} {c['timezone']} on {c['weekday']}, "
                             f"{c['date']} ({c['timezone_name']}, UTC{c['utc_offset']}).")
                    self._json(200, {"reply": reply, "model": llm.model,
                                     "time": c, "thinking_enabled": llm.think})
                    return

                # Explicitly inject a fresh timestamp into every request so the model
                # does not rely on stale conversation context or its training cutoff.
                system_content = (clock_line() + " You are a helpful local assistant. "
                    "You may explain your approach with a concise reasoning summary, but do not expose private hidden chain-of-thought. "
                    "When asked about current time, use the timestamp above. ")
                # When the user asks about the harness itself, read this exact running
                # Python file and provide relevant source context to the local model.
                harness_question = re.search(r"\b(harness|this script|your own code|agent architecture|how are you built)\b", latest_text, re.IGNORECASE)
                if harness_question:
                    try:
                        with open(os.path.abspath(__file__), "r", encoding="utf-8") as source_file:
                            harness_source = source_file.read()
                        system_content += ("\n\nSOURCE CONTEXT: The following is the actual Python file running this chat. "
                            "Use it to answer questions about this agent harness; distinguish facts from inference.\n"
                            + harness_source[:60000])
                    except Exception as source_exc:
                        system_content += f"\nCould not read own source file: {source_exc}"
                system_message = {"role": "system", "content": system_content}
                reply = asyncio.run(llm.chat([system_message, *messages], role="main"))
                self._json(200, {"reply": reply, "model": llm.model, "time": system_clock(),
                                 "thinking_enabled": llm.think})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "Invalid JSON request."})
            except Exception as exc:
                logger.exception("Browser chat request failed")
                self._json(502, {"error": str(exc)})

    try:
        server = ThreadingHTTPServer((host, port), ChatHandler)
    except OSError as exc:
        raise RuntimeError(f"Could not start web UI on http://{host}:{port}: {exc}") from exc
    url_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{url_host}:{port}"
    print(f"\nLocal Qwen Workbench (new interface): {url}")
    print(f"Ollama endpoint: {llm.base_url}")
    print(f"Configured model: {llm.model}")
    print("Keep this terminal open while chatting. Press Ctrl+C to stop the web server.\n")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local web UI...")
    finally:
        server.server_close()


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local Qwen3 agent harness v2")
    parser.add_argument("--web", action="store_true", help="launch the browser chat UI")
    parser.add_argument("--cli", action="store_true", help="use the original interactive command-line agent")
    parser.add_argument("--host", default="127.0.0.1", help="web UI bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8001, help="web UI port (default: 8001; separate from older server)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    parser.add_argument("--demo", action="store_true", help="run the scripted test instead of the prompt")
    parser.add_argument("--eval", nargs="?", const="builtin", metavar="FILE.json",
                        help="run the built-in evals (or cases from a JSON file) and exit")
    parser.add_argument("--no-network", action="store_true", help="skip eval cases that need internet")
    parser.add_argument("--session", default="default", help="conversation session name (default: default)")
    parser.add_argument("--workspace", default=None, help="sandbox folder for file/code tools")
    parser.add_argument("--yes", action="store_true", help="auto-approve file writes and code execution")
    parser.add_argument("--no-mcp", action="store_true", help="don't start MCP servers from mcp.json")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(main())
    elif args.web:
        start_web_ui(host=args.host, port=args.port, open_browser=not args.no_browser)
    elif args.eval:
        cases = None if args.eval == "builtin" else json.loads(Path(args.eval).read_text(encoding="utf-8"))
        print("Running evals (this calls your model many times)...")
        asyncio.run(run_evals(LocalQwenClient(), cases, no_network=args.no_network))
    elif args.cli:
        asyncio.run(interactive(args.session, auto_approve=args.yes, workspace=args.workspace, no_mcp=args.no_mcp))
    else:
        # This web-enabled edition opens the browser by default. Use --cli for the
        # original terminal agent, or --demo / --eval for the existing test modes.
        start_web_ui(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    cli()
