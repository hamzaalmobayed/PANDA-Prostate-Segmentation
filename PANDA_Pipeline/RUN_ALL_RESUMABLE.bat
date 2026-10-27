@echo off
setlocal
cd /d "%~dp0"

:LOOP
echo ============================================================
echo PANDA PROSTATE - GLOBAL SSD WAIT / RESUME
echo ============================================================
echo If the SSD disconnects: reconnect it and KEEP THIS WINDOW OPEN.
echo If Python crashes: this launcher restarts it automatically.
echo.

python run_full_panda_paper.py
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% EQU 0 goto DONE

echo.
echo [WATCHDOG] Python exited with code %EXITCODE%.
echo [WATCHDOG] Restarting in 15 seconds from saved checkpoints...
timeout /t 15 /nobreak >nul
goto LOOP

:DONE
echo.
echo ALL PIPELINE STAGES COMPLETED SUCCESSFULLY.
pause
