$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Starting Konnon Drawing Model Live Studio..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts\start_webview.py --port 8080
