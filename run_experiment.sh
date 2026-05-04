#!/bin/bash
# SPE 实验设计自动化迭代环境 - Linux/macOS 启动脚本

set -e

PROJECT_ROOT="E:/Github/Guassion_Process-and-Experiment_Design"

cd "$PROJECT_ROOT" 2>/dev/null || {
    echo "[错误] 无法访问项目目录: $PROJECT_ROOT"
    exit 1
}

echo "============================================================"
echo "  SPE 实验设计自动化迭代环境"
echo "  Gaussian Process + DDM Pipeline Launcher"
echo "============================================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[错误] 未找到 Python"
        exit 1
    fi
    PYTHON=python
else
    PYTHON=python3
fi

echo "Python: $($PYTHON --version)"

# 快速验证模式
run_quick() {
    echo "[快速验证模式]"
    $PYTHON -m automation.cli --profile quick --name quick_validation
}

# 标准运行
run_standard() {
    echo "[标准运行模式]"
    $PYTHON -m automation.cli --profile standard --name standard_run
}

# 研究级运行
run_research() {
    echo "[研究级运行模式]"
    $PYTHON -m automation.cli --profile research --name research_run
}

# 验证导入
verify() {
    echo "[模块验证]"
    $PYTHON -c "
import sys; sys.path.insert(0, '.')
from automation.core.design_space import DesignSpace
from automation.core.sigmoid_model import PureSigmoidModel, compute_v_s2, compute_a_s2
from automation.core.gp_model import GPHybridModel
from automation.core.ez_diffusion import ez_diffusion
from automation.core.effect_size import cohens_d_paired, g_power_analysis, bayes_factor_paired
from automation.core.generative_model import GenerativeModel
from automation.pipeline import ExperimentPipeline
print('所有模块导入成功!')
    "
}

# 菜单
show_menu() {
    echo ""
    echo "请选择运行模式:"
    echo "  [1] 快速验证"
    echo "  [2] 标准运行"
    echo "  [3] 研究级运行"
    echo "  [4] 模块验证"
    echo "  [Q] 退出"
    echo ""
}

show_menu
read -p "选择 [1-4/Q]: " choice

case $choice in
    1) run_quick ;;
    2) run_standard ;;
    3) run_research ;;
    4) verify ;;
    Q|q) echo "退出"; exit 0 ;;
    *) echo "无效选择"; exit 1 ;;
esac

echo ""
echo "流程完成!"
