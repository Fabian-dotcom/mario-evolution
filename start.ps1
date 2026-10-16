# Ein-Klick-Start fuer das Mario Neuroevolution Lab (Windows PowerShell).
#
# Erstaufruf:  .\start.ps1
#   - erstellt bei Bedarf die virtuelle Umgebung .venv
#   - installiert alle Abhaengigkeiten aus requirements.txt
#   - startet das Dashboard
#
# Danach reicht .\start.ps1 immer wieder (Installationspruefung ist schnell).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Erstelle virtuelle Umgebung .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Pruefe/Installiere Abhaengigkeiten ..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -q -e .

Write-Host "Starte Dashboard (python -m mario_evolution) ..." -ForegroundColor Green
$env:PYTHONPATH = "$PSScriptRoot\src;$env:PYTHONPATH"
.\.venv\Scripts\python.exe -m mario_evolution
