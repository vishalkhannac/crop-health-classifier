@echo off
title Plant & Vegetable Health Analyzer (Offline Mode)
echo ============================================================
echo   Starting Plant & Vegetable Health Analyzer (Offline)
echo ============================================================

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

if exist "C:\Users\kurtz\Downloads\python\Python\python.exe" (
    set "PY_EXE=C:\Users\kurtz\Downloads\python\Python\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo Using Python: %PY_EXE%
echo Opening browser at http://localhost:8501 ...
echo.

"%PY_EXE%" "%~dp0run_app.py"

pause
