# ==============================================================================
# LAAX Component 1: Automated Windows Installation Script
# Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Installing LAAX Adaptive Endpoint Agent (Component 1)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[-] Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}
Write-Host "[*] Python executable found: $($pythonCmd.Source)" -ForegroundColor Green

# 2. Pip Install in editable mode
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[*] Installing laax-agent package..." -ForegroundColor Yellow
python -m pip install -e .

# 3. Create local data directory
$laaxDir = "$env:LOCALAPPDATA\LAAX"
if (-not (Test-Path $laaxDir)) {
    New-Item -ItemType Directory -Path $laaxDir | Out-Null
    Write-Host "[+] Created local directory: $laaxDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "  Commands available:" -ForegroundColor White
Write-Host "    python -m laax_agent --help" -ForegroundColor Yellow
Write-Host "    python -m laax_agent --interactive   (Interactive live demo)" -ForegroundColor Yellow
Write-Host "    python benchmark_overhead.py         (Overhead profiling)" -ForegroundColor Yellow
Write-Host "    python run_broker.py --insecure      (message broker, development only)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
