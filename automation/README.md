# SPE 实验设计自动化迭代环境

> Gaussian Process + DDM 实验优化 Pipeline · 一键运行 · 全自动报告生成

## 📖 功能概述

本模块提供完整的 **Self-Prioritization Effect (SPE)** 实验设计自动化流程，将以下环节整合为一键运行的 Pipeline：

| 阶段 | 功能 | 对应模块 |
|------|------|----------|
| **Stage 1** | 生成实验设计空间 Ω = (P, T, W, M) | `design_space.py` |
| **Stage 2** | 生成 3×2×2 混合设计实验序列 | `experiment_sequence.py` |
| **Stage 3** | Sigmoid S2 机制 + GP 残差修正 + DDM 仿真 | `sigmoid_model.py` / `gp_model.py` / `ddm_engine.py` |
| **Stage 4** | EZ-Diffusion 参数估计 (v, a, t0) | `ez_diffusion.py` |
| **Stage 5** | 效应量分析 (Cohen's d, G-power, BF01) | `effect_size.py` |
| **Stage 6** | GP 边界探索 & 模型比较 | `generative_model.py` |
| **Stage 7** | JSON + Markdown 实验报告生成 | `pipeline.py` |

## 📁 目录结构

```
automation/
├── README.md                  # 本文件
├── __init__.py               # 包初始化
├── cli.py                    # 命令行入口（支持多种配置级别）
├── pipeline.py               # 核心 Pipeline 编排引擎
├── requirements.txt          # Python 依赖列表
├── install_dependencies.bat  # 自动安装依赖脚本
├── run_experiment.bat        # 交互式菜单启动脚本
├── 一键开始.bat              # 最简单的一键启动（安装+运行）
│
└── core/                    # 核心模块
    ├── __init__.py
    ├── design_space.py       # 实验设计空间 (P×T×W 网格 / LHS 采样)
    ├── experiment_sequence.py # 实验序列生成 (混合/区组/拉丁方 3种模式)
    ├── sigmoid_model.py      # Sigmoid S2 机制函数 (理论先验)
    ├── gp_model.py           # 高斯过程模型 (GP + S2 混合架构)
    ├── ddm_engine.py         # DDM 仿真引擎 (Euler + Deadline + Omission)
    ├── generative_model.py    # 整合生成模型 (Sigmoid → GP → DDM → 行为数据)
    ├── ez_diffusion.py       # EZ-Diffusion 参数反推 (RT/ACC → v, a, t0)
    ├── effect_size.py        # 效应量指标 (Cohen's d, G-power, BF01)
    └── logger.py             # 实验日志与可重复性追溯系统
```

## 🔧 环境要求

| 软件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.9+ | 推荐 3.11+ |
| numpy | 1.20.0+ | 数值计算 |
| pandas | 1.3.0+ | 数据处理 |
| scikit-learn | 1.0.0+ | 高斯过程模型 |
| scipy | 1.7.0+ | 统计检验 |
| matplotlib | 3.4.0+ | 图表输出 (可选) |

## 🚀 安装与运行

### 第一步：安装依赖

**方式 A（推荐）：** 双击 `install_dependencies.bat`，脚本会自动查找 Python 并安装所有依赖。

**方式 B：** 手动安装
```bash
cd automation
pip install -r requirements.txt
```

### 第二步：运行

> **注意：** 请在 `automation/` 目录下运行（不要切换到其他目录）

```bash
# 方式 1: 直接运行脚本（最简单）
python cli.py

# 方式 2: 交互式菜单
run_experiment.bat

# 方式 3: 一键启动（自动安装依赖 + 运行）
一键开始.bat

# 方式 4: 带参数运行
python cli.py --profile quick
python cli.py --profile standard
python cli.py --profile research
```

## ⚙️ 配置方法

### 三种预设配置级别

| 级别 | 命令参数 | 设计空间 | 被试数 | 试次/条件 | 迭代轮数 |
|------|---------|---------|--------|----------|---------|
| **快速** | `--profile quick` | 3×3×3 = 27 点 | 8 | 8 | 2 |
| **标准** | `--profile standard` | 8×8×8 = 512 点 | 30 | 20 | 5 |
| **研究** | `--profile research` | 14×13×9 点 | 50 | 30 | 10 |

### 自定义命令行参数

```bash
python cli.py --rounds 8 --subjects 40 --name my_experiment
python cli.py --no-real-data           # 仅模拟，不加载真实数据
python cli.py --seed 123              # 固定随机种子
```

### 通过配置文件运行

创建 JSON 配置文件，然后：

```bash
python cli.py --config my_config.json
```

配置 JSON 示例：

```json
{
  "design": {
    "P_values": [0, 32, 64, 120],
    "T_values": [30, 100, 200, 500],
    "W_values": [300, 600, 1000, 1500],
    "lhs_points": 30
  },
  "experiment": {
    "n_subjects": 40,
    "trials_per_condition": 25,
    "design_type": "mixed"
  },
  "iteration": {
    "n_rounds": 5,
    "use_real_data": true
  },
  "seed": 42
}
```

### Python API 编程调用

```python
# 从项目根目录运行 Python
import sys
sys.path.insert(0, '.')
from automation.pipeline import ExperimentPipeline

pipeline = ExperimentPipeline(config={
    'experiment': {'n_subjects': 30, 'trials_per_condition': 20},
    'iteration': {'n_rounds': 5},
})

results = pipeline.run()
print(f"SPE: {results['effect_analysis']['synthetic_SPE']['SPE_ms_mean']:.1f} ms")
```

## 📝 使用示例

### 1. 快速验证（推荐首次使用）

```bash
# 在 automation/ 目录下运行
python cli.py --profile quick --name quick_test
```

运行日志会显示：

```
============================================================
  实验优化自动化流程启动
============================================================
[INFO] 生成实验设计空间...
[INFO] 设计空间生成完成: 27 个网格点 (耗时 0.1s)
[INFO] 生成实验序列...
[INFO] 实验序列生成完成: 768 试次, 8 被试 (耗时 0.2s)
[INFO] 加载真实实验数据...
[INFO] 真实数据加载完成: 26616 行 (耗时 0.3s)
...
[INFO] 效应量分析: SPE=XXms, Cohen's d=XX, BF01=XX
============================================================
  实验完成!
  报告位置: automation/logs/{run_id}/final_report.json
============================================================
```

### 2. 标准研究运行

```bash
python cli.py --profile standard --name study_01
```

### 3. 完整研究级运行

```bash
python cli.py --profile research --name final_study
```

### 4. 交互式菜单

双击 `run_experiment.bat`，交互式菜单支持：

```
[1] 快速验证    (Quick:  8被试, 3×3×3 设计, 2轮迭代)
[2] 标准运行    (Standard: 30被试, 8×8×8 设计, 5轮迭代)
[3] 研究级运行  (Research: 50被试, 14×13×9 设计, 10轮迭代)
[4] 自定义运行  (通过命令行参数指定)
[5] 仅测试导入  (验证所有模块是否能正确加载)
[6] 查看最近报告
[Q] 退出
```

### 5. 仅测试模块导入（不生成数据）

```bash
python cli.py --profile quick --no-real-data --name import_test
# 或双击 run_experiment.bat，选择 [5] 仅测试导入
```

## 📊 输出文件

运行完成后，所有输出保存到 `automation/logs/{run_id}/`：

| 文件 | 说明 |
|------|------|
| `final_report.json` | 完整实验报告 (JSON) |
| `final_report.md` | Markdown 格式报告 |
| `design_grid.csv` | 设计空间网格 |
| `design_lhs.csv` | LHS 采样点 |
| `experiment_sequence.csv` | 实验试次序列 |
| `synthetic_behavior_data.csv` | 生成的行为数据 |
| `model_comparison.csv` | 模型比较结果 |
| `effect_analysis.json` | 效应量分析结果 |
| `experiment_log.txt` | 完整运行日志 |

## 🧩 核心模块说明

### `design_space.py` — 设计空间管理

```python
from automation.core.design_space import DesignSpace

ds = DesignSpace(seed=42)
grid = ds.generate_grid(
    P_values=[0, 64, 120],
    T_values=[30, 200],
    W_values=[300, 800],
)  # 3×2×2 = 12 个条件
```

### `experiment_sequence.py` — 实验序列生成

```python
from automation.core.experiment_sequence import ExperimentSequence

seq = ExperimentSequence(seed=42)
trials = seq.generate_mixed_design(design_df, n_subjects=30, trials_per_condition=20)
# 生成: 48 condition×40 trials = 1920 试次/被试 → 57600 总试次
```

### `sigmoid_model.py` — S2 机制函数

```python
from automation.core.sigmoid_model import compute_v_s2, compute_a_s2

v_self = compute_v_s2(T=200, P=64, condition_key=1)     # Self 条件漂移率
v_str  = compute_v_s2(T=200, P=64, condition_key=0)     # Stranger 漂移率
a      = compute_a_s2(M=800)                              # 决策边界
```

### `ddm_engine.py` — DDM 仿真

```python
from automation.core.ddm_engine import simulate_ddm_euler, simulate_ddm_with_deadline

RT, response = simulate_ddm_euler(v=2.0, a=1.5, z=0.75, t0=0.2)
RT, resp, omission, source = simulate_ddm_with_deadline(v=2.0, a=1.5, z=0.75, t0=0.2, deadline_s=0.8)
```

### `ez_diffusion.py` — 参数反推

```python
from automation.core.ez_diffusion import ez_diffusion, subject_level_ez_diffusion

params = ez_diffusion(rt_mean=0.45, rt_var=0.02, p_correct=0.85)
# 返回: {'v': 1.2, 'a': 0.15, 'ter': 0.3}
```

### `effect_size.py` — 效应量分析

```python
from automation.core.effect_size import cohens_d_paired, bayes_factor_paired, g_power_analysis

d_result = cohens_d_paired(self_rt, stranger_rt)
bf_result = bayes_factor_paired(self_rt, stranger_rt)
power_info = g_power_analysis(d_result['d'], n=40)
```

## ❓ 常见问题解答

### Q1: 运行时提示 `ModuleNotFoundError: No module named 'automation'`

**原因：** 从错误的目录运行了命令。

**解决方法：** 请务必在 `automation/` 目录下运行：

```bash
cd automation                      # 先进入 automation 目录
python cli.py                       # 再运行脚本
```

> `cli.py` 会自动将项目根目录添加到 `sys.path`，不需要任何额外配置。

### Q2: 运行时提示 `ModuleNotFoundError: No module named 'core'`

**原因：** 同 Q1。

**解决方法：** 同 Q1。请确保在 `automation/` 目录下运行 `python cli.py`。

### Q3: 找不到真实数据文件

**A:** Pipeline 默认查找 `2_Data/Real_Data/EXP_data_combined.csv`。如果文件不存在，使用 `--no-real-data` 跳过真实数据：

```bash
python cli.py --no-real-data
```

### Q4: 安装依赖失败 / 网络问题

**A:** `install_dependencies.bat` 内置了多重镜像源（清华、阿里云、豆瓣、官方 PyPI），会自动尝试。如果全部失败，请手动安装：

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### Q5: "快速"、"标准"、"研究" 配置有什么区别？

**A:** 区别在于数据规模和运行时间：

| 配置 | 设计空间 | 运行时间估计 | 适用场景 |
|------|---------|-------------|---------|
| quick | 27 条件 | ~30s | 验证代码是否正确 |
| standard | 512 条件 | ~2min | 常规研究分析 |
| research | 1638 条件 | ~8min | 完整研究报告 |

### Q6: 如何查看历史运行结果？

**A:** 所有运行结果按时间戳保存在 `automation/logs/` 下。运行 `run_experiment.bat` 选择 `[6] 查看最近报告` 可快速查看最近的运行。

### Q7: GP 模型训练需要额外配置吗？

**A:** 不需要。Pipeline 会自动根据配置初始化 GP 模型。如需自定义 GP 超参数，修改配置文件或代码中的 `kernel_type` 字段。

### Q8: 支持哪些实验设计类型？

**A:** 支持三种设计类型：
- `mixed`：混合设计（默认），每个被试经历所有条件
- `blocked`：区组设计，按 block 分组
- `latin_square`：拉丁方平衡设计

在配置中通过 `design_type` 字段指定。

### Q9: 生成的数据格式是什么？

**A:** 生成的 CSV 数据与真实实验数据格式兼容，包含以下核心列：
`subject`, `trial`, `P`, `T`, `W`, `M`, `label` (self/stranger), `matching` (Matching/NonMatching), `v`, `a`, `t0`, `z`, `RT`, `response`, `correct`, `omission`, `omission_source`

---

*最后更新: 2026-05*
