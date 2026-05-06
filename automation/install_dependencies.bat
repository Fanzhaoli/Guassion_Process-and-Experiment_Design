@echo off
chcp 65001 > nul
title SPE 实验自动化 - 依赖自动安装

echo ============================================================
echo   SPE 实验自动化环境 - 依赖自动安装程序
echo ============================================================
echo.

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: 1. 查找 Python
echo [步骤1/4] 查找可用的 Python 环境...
set PYTHON_CMD=
for %%p in (
    "python3"
    "python"
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
    "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python.exe"
) do (
    %%p --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=%%p"
        set "PYTHON_PATH=%%~p"
        echo [找到] %%p
        goto :found_python
    )
)

echo [错误] 未找到 Python！请检查 Python 是否已安装。
pause
exit /b 1

:found_python
echo.
echo [步骤2/4] 检查 Python 版本...
%PYTHON_CMD% --version

:: 2. 检查是否已安装依赖
echo.
echo [步骤3/4] 检查现有依赖包...
set "NEED_INSTALL=0"

%PYTHON_CMD% -c "import numpy; print(' numpy: OK')" 2>nul
if errorlevel 1 (
    echo numpy: [缺失]
    set "NEED_INSTALL=1"
)

%PYTHON_CMD% -c "import pandas; print(' pandas: OK')" 2>nul
if errorlevel 1 (
    echo pandas: [缺失]
    set "NEED_INSTALL=1"
)

%PYTHON_CMD% -c "import sklearn; print(' sklearn: OK')" 2>nul
if errorlevel 1 (
    echo sklearn: [缺失]
    set "NEED_INSTALL=1"
)

%PYTHON_CMD% -c "import scipy; print(' scipy: OK')" 2>nul
if errorlevel 1 (
    echo scipy: [缺失]
    set "NEED_INSTALL=1"
)

%PYTHON_CMD% -c "import matplotlib; print(' matplotlib: OK')" 2>nul
if errorlevel 1 (
    echo matplotlib: [缺失]
    set "NEED_INSTALL=1"
)

echo.

if %NEED_INSTALL% equ 0 (
    echo [√] 所有依赖已安装完成！无需再安装。
    goto :verify_and_done
)

echo [!] 部分依赖未安装，开始自动安装...
echo.

:: 3. 尝试安装依赖
echo [步骤4/4] 尝试安装依赖包...
echo   (可能需要等待几分钟，取决于网络)
echo.

:: 尝试方式1: 使用国内镜像加速
echo [方式1/4] 尝试使用国内清华镜像安装...
%PYTHON_CMD% -m pip install numpy pandas scikit-learn scipy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if errorlevel 1 (
    echo [×] 镜像安装失败
) else (
    goto :verify_and_done
)

:: 尝试方式2: 阿里云镜像
echo.
echo [方式2/4] 尝试阿里云镜像...
%PYTHON_CMD% -m pip install numpy pandas scikit-learn scipy matplotlib -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
if errorlevel 0 (
    goto :verify_and_done
)

:: 尝试方式3: 豆瓣镜像
echo.
echo [方式3/4] 尝试豆瓣镜像...
%PYTHON_CMD% -m pip install numpy pandas scikit-learn scipy matplotlib -i https://pypi.douban.com/simple/ --trusted-host pypi.douban.com
if errorlevel 0 (
    goto :verify_and_done
)

:: 尝试方式4: 官方源
echo.
echo [方式4/4] 尝试官方 PyPI 源...
%PYTHON_CMD% -m pip install numpy pandas scikit-learn scipy matplotlib
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [×] 所有安装方式均失败！
    echo.
    echo 请尝试以下方法手动解决：
    echo   1. 检查网络连接是否正常
    echo   2. 检查是否需要设置代理
    echo   3. 请手动运行以下命令：
    echo      pip install numpy pandas scikit-learn scipy matplotlib
    echo ============================================================
    echo.
    pause
    exit /b 1
)

:verify_and_done
echo.
echo ============================================================
echo   正在验证依赖安装结果...
echo ============================================================

%PYTHON_CMD% -c "
import sys
print('=== 模块验证 ===')
try:
    import numpy; print(' [√] numpy')
    import pandas; print(' [√] pandas')
    import sklearn; print(' [√] scikit-learn')
    import scipy; print(' [√] scipy')
    import matplotlib; print(' [√] matplotlib')
    print()
    print('所有依赖安装成功！')
except ImportError as e:
    print(f' [×] 缺少模块: {e}')
    print()
    print('请重新运行此脚本，或手动 pip install')
    sys.exit(1)
" 2>&1

if errorlevel 1 (
    echo.
    echo [×] 仍有部分依赖缺失，安装可能未完成。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [√] 依赖安装完成！
echo ============================================================
echo.
echo 现在你可以运行实验了！
echo   - 双击 run_experiment.bat 启动
echo   - 或在命令行运行: python -m automation.cli --profile quick
echo.
pause
