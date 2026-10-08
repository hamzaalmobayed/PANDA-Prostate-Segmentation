@echo off
python -c "import torch; print('torch=',torch.__version__); print('cuda_available=',torch.cuda.is_available()); print('cuda_runtime=',torch.version.cuda); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if exist "%SystemRoot%\System32\nvidia-smi.exe" nvidia-smi
pause
