# start.ps1 — Offline Launcher for Plant and Vegetable Health Analyzer
$ErrorActionPreference = "Stop"

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_HEADLESS = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$candidateSite = @(
    "$scriptDir\venv\Lib\site-packages",
    "D:\plant-veg-health\venv\Lib\site-packages"
)
foreach ($p in $candidateSite) {
    if (Test-Path $p) {
        $env:PYTHONPATH = "$p;$scriptDir;$scriptDir\src;D:\plant-veg-health\src;" + $env:PYTHONPATH
    }
}

if (Test-Path "C:\Users\kurtz\Downloads\python\Python\python.exe") {
    $pythonExe = "C:\Users\kurtz\Downloads\python\Python\python.exe"
} elseif (Test-Path "$scriptDir\venv\Scripts\python.exe") {
    $pythonExe = "$scriptDir\venv\Scripts\python.exe"
} elseif (Test-Path "D:\plant-veg-health\venv\Scripts\python.exe") {
    $pythonExe = "D:\plant-veg-health\venv\Scripts\python.exe"
} else {
    $pythonExe = "python"
}

Write-Host "Starting Plant and Vegetable Health Analyzer (Offline Mode)..." -ForegroundColor Green
Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan
Write-Host "Opening Streamlit app..." -ForegroundColor Yellow

& $pythonExe "$scriptDir\run_app.py"
