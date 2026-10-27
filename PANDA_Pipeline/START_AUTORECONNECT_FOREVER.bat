@echo off
setlocal
set "PIPELINE_DIR=%~dp0"
set "LOCAL_WATCHDOG=%LOCALAPPDATA%\PANDA_WATCHDOG"
if not exist "%LOCAL_WATCHDOG%" mkdir "%LOCAL_WATCHDOG%"

copy /Y "%PIPELINE_DIR%PANDA_AUTORECONNECT_WATCHDOG.ps1" "%LOCAL_WATCHDOG%\PANDA_AUTORECONNECT_WATCHDOG.ps1" >nul

echo ===============================================================
echo PANDA AUTO-RECONNECT
echo The watchdog is copied to your internal Windows drive.
echo If the external SSD disconnects, leave the watchdog window open.
echo It will wait for the SSD and restart/resume automatically.
echo ===============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%LOCAL_WATCHDOG%\PANDA_AUTORECONNECT_WATCHDOG.ps1" -PipelineDir "%PIPELINE_DIR%"
