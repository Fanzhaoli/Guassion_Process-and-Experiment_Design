@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ┌──────────────────────────────────────────────────────────┐
:: │  SPE 实验设计自动化迭代环境 - 一键启动脚本               │
:: │  Guassion Process & Experiment Design Automation         │
:: │  Gaussian Process + DDM Pipeline Launcher                │
:: └──────────────────────────────────────────────────────────┘

set "PROJECT_ROOT=E:\Github\Guassion_Process-and-Experiment_Design"

cd /d "%PROJECT_ROOT%" 2>nul
if errorlevel 1 (
    echo [错误] 无法访问项目目录: %PROJECT_ROOT%
    echo 请检查路径是否正确。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   SPE 实验设计自动化迭代环境
echo   Guassion Process ^& Experiment Design Automation
echo ============================================================
echo.
echo 项目路径: %PROJECT_ROOT%
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保 Python 已安装并添加到 PATH
    pause
    exit /b 1
)

echo Python 环境检查通过
python --version 2>&1
echo.

:: 检查必要包
echo 检查 Python 依赖包...
python -c "import numpy, pandas, sklearn, scipy; print('numpy, pandas, sklearn, scipy: OK')" 2>nul
if errorlevel 1 (
    echo [警告] 部分依赖包未安装
    echo 正在安装必要依赖...
    pip install numpy pandas scikit-learn scipy matplotlib -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动安装:
        echo   pip install numpy pandas scikit-learn scipy matplotlib
        pause
        exit /b 1
    )
)
echo.

:: ── 主菜单 ──
:menu
echo ============================================================
echo   请选择运行模式:
echo.
echo   [1] 快速验证    (Quick:  8被试,  3x3x3 设计, 2轮迭代)
echo   [2] 标准运行    (Standard: 30被试, 8x8x8 设计, 5轮迭代)
echo   [3] 研究级运行  (Research: 50被试, 14x13x9 设计, 10轮迭代)
echo   [4] 自定义运行  (通过命令行参数指定)
echo   [5] 仅测试导入  (验证所有模块是否能正确加载)
echo   [6] 查看最近报告
echo   [Q] 退出
echo.
echo ============================================================
set /p choice="请输入选择 [1-6/Q]: "

if /i "%choice%"=="Q" goto end
if "%choice%"=="1" goto quick
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto research
if "%choice%"=="4" goto custom
if "%choice%"=="5" goto verify
if "%choice%"=="6" goto reports
echo 无效选择，请重试
goto menu

:: ── 快速验证模式 ──
:quick
echo.
echo [启动] 快速验证模式...
echo.
python -m automation.cli --profile quick --name quick_validation
if errorlevel 1 (
    echo.
    echo [错误] 运行失败！请检查错误信息。
    echo 尝试使用备用启动方式...
    echo.
    python automation/cli.py --profile quick --name quick_validation
)
goto menu

:: ── 标准运行模式 ──
:standard
echo.
echo [启动] 标准运行模式...
echo.
python -m automation.cli --profile standard --name standard_run
if errorlevel 1 (
    echo.
    echo [错误] 运行失败！
    echo 尝试使用备用启动方式...
    echo.
    python automation/cli.py --profile standard --name standard_run
)
goto menu

:: ── 研究级运行模式 ──
:research
echo.
echo [启动] 研究级运行模式...
echo.
python -m automation.cli --profile research --name research_run
if errorlevel 1 (
    echo.
    echo [错误] 运行失败！
    echo 尝试使用备用启动方式...
    echo.
    python automation/cli.py --profile research --name research_run
)
goto menu

:: ── 自定义运行 ──
:custom
echo.
echo === 自定义运行配置 ===
echo.
set /p rounds="迭代轮数 (默认: 5): "
if "%rounds%"=="" set rounds=5
set /p subjects="被试数量 (默认: 30): "
if "%subjects%"=="" set subjects=30
set /p name="实验名称 (默认: custom_run): "
if "%name%"=="" set name=custom_run
echo.
echo 配置: %rounds% 轮, %subjects% 被试, 名称: %name%
echo.

python -m automation.cli --profile standard --rounds %rounds% --subjects %subjects% --name %name%
if errorlevel 1 (
    echo [错误] 运行失败！
    echo 尝试使用备用启动方式...
    echo.
    python automation/cli.py --profile standard --rounds %rounds% --subjects %subjects% --name %name%
)
goto menu

:: ── 验证模式 ──
:verify
echo.
echo === 模块验证 ===
echo.
echo [1/6] 测试核心模块导入...
python -c "import sys; sys.path.insert(0,'.'); from automation.core.design_space import DesignSpace; print('  design_space: OK')" 2>&1
python -c "import sys; sys.path.insert(0,'.'); from automation.core.experiment_sequence import ExperimentSequence; print('  experiment_sequence: OK')" 2>&1
python -c "import sys; sys.path.insert(0,'.'); from automation.core.sigmoid_model import PureSigmoidModel; print('  sigmoid_model: OK')" 2>&1
python -c "import sys; sys.path.insert(0,'.'); from automation.core.gp_model import GPHybridModel; print('  gp_model: OK')" 2>&1
python -c "import sys; sys.path.insert(0,'.'); from automation.core.ez_diffusion import ez_diffusion; print('  ez_diffusion: OK')" 2>&1
python -c "import sys; sys.path.insert(0,'.'); from automation.core.effect_size import cohens_d_paired; print('  effect_size: OK')" 2>&1
echo.
echo [2/6] 测试 Pipeline 导入...
python -c "import sys; sys.path.insert(0,'.'); from automation.pipeline import ExperimentPipeline; print('  pipeline: OK')" 2>&1
echo.
echo [3/6] 测试设计空间生成...
python -c "from automation.core.design_space import DesignSpace; ds=DesignSpace(); df=ds.generate_grid(); print(f'  生成 {len(df)} 个设计点')" 2>&1
echo.
echo [4/6] 测试 Sigmoid 模型...
python -c "from automation.core.sigmoid_model import compute_v_s2, compute_a_s2; v=compute_v_s2(200,64,1); print(f'  v_self(200,64): {float(v):.4f}')" 2>&1
echo.
echo [5/6] 测试 EZ-Diffusion...
python -c "from automation.core.ez_diffusion import ez_diffusion; p=ez_diffusion(0.6, 0.01, 0.85); print(f'  ez: v={p[\"v\"]:.3f}, a={p[\"a\"]:.3f}, ter={p[\"ter\"]:.3f}')" 2>&1
echo.
echo [6/6] 测试完整生成模型...
python -c "from automation.core.generative_model import GenerativeModel; import pandas as pd; gm=GenerativeModel(seed=99); ds=gm.design_space; df=ds.generate_grid(['P','T','W'],[0,64,120],[30,200],[300,800]); syn=gm.generate_dataset(df,n_subjects=3,trials_per_condition=5); print(f'  生成 {len(syn)} 试次, {syn.subject.nunique()} 被试'); print(f'  RT均值: {syn[syn.RT.notna()].RT.mean()*1000:.1f}ms, omission率: {syn.omission.mean():.3f}')" 2>&1
echo.
echo === 验证完成 ===
pause
goto menu

:: ── 查看最近报告 ──
:reports
echo.
echo === 最近运行报告 ===
echo.
set "LOG_DIR=%PROJECT_ROOT%\automation\logs"
if not exist "%LOG_DIR%" (
    echo 暂无报告 (日志目录不存在)
    goto menu
)

:: 列出最近的运行目录
set count=0
for /f "tokens=*" %%d in ('dir /b /ad /o-d "%LOG_DIR%" 2^>nul') do (
    set /a count+=1
    if !count! leq 5 (
        echo !count!. %%d
        if exist "%LOG_DIR%\%%d\results_summary.json" (
            python -c "import json; f=open(r'%LOG_DIR%\%%d\results_summary.json','r',encoding='utf-8'); d=json.load(f); print(f'   耗时: {d.get(\"total_elapsed_s\",\"?\")}s, 成功: {d.get(\"success\",\"?\")}')" 2>nul
        )
        echo.
    )
)

if %count% equ 0 (
    echo 暂无报告
)
echo.
pause
goto menu

:end
echo.
echo 自动化环境已关闭。
exit /b 0
