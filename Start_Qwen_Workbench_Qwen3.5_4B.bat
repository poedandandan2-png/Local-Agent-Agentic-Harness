@echo off
setlocal
cd /d "%~dp0"
title Local Qwen Workbench - Qwen3.5 4B

set "OLLAMA_HOST=http://127.0.0.1:11434"
set "LLM_BACKEND=ollama"
set "QWEN_MODEL=qwen3.5:4b"
set "LOCAL_TIMEZONE=America/Denver"

 echo Checking Ollama connection...
curl -sS --max-time 5 "%OLLAMA_HOST%/api/tags" >nul
if errorlevel 1 (
  echo.
  echo ERROR: Cannot connect to Ollama at %OLLAMA_HOST%.
  echo Check that the Docker container is running and port 11434 is published.
  echo.
  pause
  exit /b 1
)

echo Checking that qwen3.5:4b is installed...
curl -sS "%OLLAMA_HOST%/api/tags" | findstr /i "qwen3.5:4b" >nul
if errorlevel 1 (
  echo Model tag not found in Ollama response.
  echo Run: docker exec -it ollama ollama pull qwen3.5:4b
  echo.
  pause
  exit /b 1
)

echo Starting Qwen Workbench...
python "%~dp0Local_Qwen_Workbench_Qwen3.5_4B.py"
if errorlevel 1 (
  echo.
  echo Workbench stopped with an error.
  pause
)
endlocal
