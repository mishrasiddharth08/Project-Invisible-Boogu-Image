@echo off
setlocal
cd /d "%~dp0"
set "BOOGU_PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not exist "%BOOGU_PYTHON%" (
  echo Forge Python was not found. Edit BOOGU_PYTHON in this file to your existing Forge Python.
  pause
  exit /b 1
)
if "%~1"=="" echo Boogu manual setup: nothing downloads until you select files and type DOWNLOAD.
"%BOOGU_PYTHON%" -m pi_boogu_image.download.setup %*
pause
