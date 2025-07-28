@echo off
REM 全BMPファイルに対してガウス・ニュートン法を実行

set SCRIPT=experiment_gauss_newton_method.py
set INPUT_DIR=input/color
set SCALE=1.2
set THETA=45
set SCALE_INIT=1
set THETA_INIT=50
set THRESHOLD=1e-4

for %%I in (%INPUT_DIR%\*.bmp) do (
    echo Running on %%~nxI ...
    python %SCRIPT% %%I %SCALE% %THETA% --scale_init %SCALE_INIT% --theta_init %THETA_INIT% --threshold %THRESHOLD%
)

pause
