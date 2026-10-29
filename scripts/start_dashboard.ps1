# Start the project dashboard (Streamlit) in its own window: http://localhost:8501
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start_dashboard.ps1 [-Port 8501] [-Lan]
# -Lan serves on all interfaces so a phone on the same network can open http://<pc-ip>:<port>
# (Windows may ask to allow it through the firewall).
param(
    [int]$Port = 8501,
    [switch]$Lan,
    [string]$Python = "D:\venvs\dash\Scripts\python.exe"
)
$Repo = Split-Path -Parent $PSScriptRoot
$Address = if ($Lan) { "0.0.0.0" } else { "localhost" }
$Args = @("-m", "streamlit", "run", (Join-Path $Repo "scripts\dashboard.py"),
    "--server.port", $Port, "--server.address", $Address,
    "--server.headless", "true", "--browser.gatherUsageStats", "false")
Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Repo
Write-Host "Dashboard starting on http://${Address}:$Port"
