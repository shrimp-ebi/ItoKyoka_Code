@echo off
setlocal enabledelayedexpansion

REM 実行対象のPythonスクリプト
set SCRIPT=experiment_gauss_newton_method.py

REM 入力画像と固定パラメータ
set IMAGE=input/color/Airplane.bmp
set SCALE_TRUE=1.2
set THETA_TRUE=45
set THRESHOLD=1e-4

REM scale_init のリスト
for %%S in (1 1.1 1.2 1.3 1.4) do (
    REM theta_init のリスト
    for %%T in (35 40 45 50 55) do (
        echo 実行中: scale_init=%%S, theta_init=%%T
        python %SCRIPT% %IMAGE% %SCALE_TRUE% %THETA_TRUE% --scale_init %%S --theta_init %%T --threshold %THRESHOLD%
    )
)

echo 実験が完了しました
pause
