# start.ps1 — Offline Launcher for Plant & Vegetable Health Analyzer
$ErrorActionPreference = "Stop"

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_HEADLESS = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (Test-Path "C:\Users\kurtz\Downloads\python\Python\python.exe") {
    $pythonExe = "C:\Users\kurtz\Downloads\python\Python\python.exe"
} elseif (Test-Path "$scriptDir\venv\Scripts\python.exe") {
    $pythonExe = "$scriptDir\venv\Scripts\python.exe"
} else {
    $pythonExe = "python"
}

Write-Host "Starting Plant & Vegetable Health Analyzer (Offline Mode)..." -ForegroundColor Green
Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan
Write-Host "Opening Streamlit app..." -ForegroundColor Yellow

& $pythonExe "$scriptDir\run_app.py"
