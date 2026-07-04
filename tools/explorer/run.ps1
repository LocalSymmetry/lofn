# Lofn Prompt Explorer — one-command launch (Windows).
#   pwsh -File tools/explorer/run.ps1        # or right-click > Run with PowerShell
# Creates the venv + installs deps + builds the web UI on first run, then serves
# the whole thing from one local process at http://127.0.0.1:8765.
$ErrorActionPreference = "Stop"
$explorer = $PSScriptRoot
$root = (Resolve-Path (Join-Path $explorer "..\..")).Path
$venv = Join-Path $explorer "server\.venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
  Write-Host "[explorer] creating venv + installing backend deps..." -ForegroundColor Cyan
  python -m venv $venv
  & $py -m pip install --quiet --upgrade pip
  & $py -m pip install --quiet -r (Join-Path $explorer "server\requirements.txt")
}

$dist = Join-Path $explorer "web\dist\index.html"
if ((-not (Test-Path $dist)) -or ($args -contains "--build")) {
  Write-Host "[explorer] building web UI..." -ForegroundColor Cyan
  Push-Location (Join-Path $explorer "web")
  if (-not (Test-Path "node_modules")) { npm install --no-fund --no-audit }
  npm run build
  Pop-Location
}

Write-Host "[explorer] serving http://127.0.0.1:8765  (Ctrl+C to stop)" -ForegroundColor Green
Start-Process "http://127.0.0.1:8765"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
& $py -m tools.explorer.server
