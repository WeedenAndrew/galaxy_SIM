# Build the one-click executable.
#
#   .\build.ps1
#
# Uses the project venv, so the build sees exactly the packages the simulation
# was measured against. Installing PyInstaller globally would build against
# whatever happens to be on PATH — which here is a 32-bit Python 3.8 with
# neither NumPy nor pygame.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "No venv found. Create one first:
    C:\Python314\python.exe -m venv .venv
    .\.venv\Scripts\python.exe -m pip install numpy scipy pygame-ce"
}

Write-Host "`n== installing PyInstaller into the venv ==" -ForegroundColor Cyan
& $py -m pip install --quiet --upgrade pyinstaller

Write-Host "`n== building ==" -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --clean GalaxySim.spec

$exe = if (Test-Path ".\dist\GalaxySim\GalaxySim.exe") {
    ".\dist\GalaxySim\GalaxySim.exe"
} else {
    ".\dist\GalaxySim.exe"
}

if (Test-Path $exe) {
    $size = (Get-ChildItem $exe).Length / 1MB
    $total = (Get-ChildItem .\dist -Recurse | Measure-Object Length -Sum).Sum / 1MB
    Write-Host "`n  built: $exe" -ForegroundColor Green
    Write-Host ("  exe {0:N1} MB, dist total {1:N1} MB" -f $size, $total)
    Write-Host "`n  Double-click it, or:  $exe"
    Write-Host "  If it fails, look for galaxy-sim-crash-*.log beside the exe."
} else {
    Write-Error "Build reported success but no executable was produced."
}
