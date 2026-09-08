@echo off
REM Build the executable. Double-click this, or run it from any shell.
REM
REM Wraps build.ps1 because Windows blocks .ps1 files by default under the
REM Restricted execution policy. The bypass below applies to this one process
REM only -- it does not change the machine's policy, and nothing about your
REM system is different after it exits. Same wrapper CAPTURE.cmd uses in
REM desk_Puff.

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\build.ps1"

echo.
pause
