@echo off
setlocal
cd /d "%~dp0"
set NO_ALBUMENTATIONS_UPDATE=1
:restart
python run_full_panda_paper.py
set RC=%ERRORLEVEL%
if %RC%==0 goto done
echo.
echo [WATCHDOG] Python exited with code %RC%. Waiting 10 seconds, then restarting from durable state...
timeout /t 10 /nobreak >nul
goto restart
:done
echo.
echo [DONE] Final model/results and all resumed pilots completed successfully.
pause
