@echo off
setlocal
set "PIPELINE_DIR=%~dp0"
set "LOCAL_WATCHDOG=%LOCALAPPDATA%\PANDA_SICAPV2_WATCHDOG"
if not exist "%LOCAL_WATCHDOG%" mkdir "%LOCAL_WATCHDOG%"
copy /Y "%PIPELINE_DIR%PANDA_EXTERNAL_SICAPV2_WATCHDOG.ps1" "%LOCAL_WATCHDOG%\PANDA_EXTERNAL_SICAPV2_WATCHDOG.ps1" >nul

echo ===============================================================
echo SICAPv2 EXTERNAL TEST - AUTO DOWNLOAD + FULL RESUME
echo ===============================================================
echo This is independent from run_full_panda_paper.py.
echo Keep this window open. SSD/network interruptions are resumed automatically.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%LOCAL_WATCHDOG%\PANDA_EXTERNAL_SICAPV2_WATCHDOG.ps1" -PipelineDir "%PIPELINE_DIR%"
