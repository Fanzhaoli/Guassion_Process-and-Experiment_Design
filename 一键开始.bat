@echo off
chcp 65001 > nul
title SPE 实验自动化 - 超简单一键开始

set "PROJECT_ROOT=E:\Github\Guassion_Process-and-Experiment_Design"
cd /d "%PROJECT_ROOT%"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║   SPE 实验设计自动化迭代环境 - 超简单一键开始             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

echo [1/2] 检查并安装依赖...
call install_dependencies.bat
if errorlevel 1 (
    echo.
    echo [依赖安装失败，请重试]
    pause
    exit /b 1
)

echo.
echo.
echo [2/2] 启动实验...
call run_experiment.bat

:end
echo.
