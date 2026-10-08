@echo off
setlocal
cd /d "%~dp0"
set NO_ALBUMENTATIONS_UPDATE=1
python run_external_sicapv2_test.py
pause
