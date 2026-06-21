# CRF 分析与仿真数据生成对比

## 文件概览

本文档比较两组代码的差异与关联：

| 组别 | 文件 | 目录 |
|------|------|------|
| **新 (CRF 分析)** | `plot_CRF_zbias_HDDM.ipynb`, `plot_CRF_zbias_Wiener.ipynb`, `plot_CRF_zbias.py` | `HDDM_Stim-Coding_Simulation/` |
| **旧 (模型比较)** | `Response_Bias/simulation/simulation_all.ipynb`, `Stimulus_Bias/simulation/simuation_all.ipynb` | `ddm_stim_coding/` |

---

## 一、`plot_CRF_zbias_HDDM.ipynb` vs `plot_CRF_zbias_Wiener.ipynb`

> 两者位于 `HDDM_Stim-Coding_Simulation/`，结构几乎相同，核心区别在于**数据生成引擎**。

| 对比维度 | Wiener 模式 | HDDM 模式 |
|----------|-------------|------------|
| **数据生成引擎** | Euler-Maruyama 数值积分模拟 Wiener 扩散过程 (`dt=0.001`) | `hddm.generate.gen_rand_data()` 从 Wald/逆高斯分布直接采样 |
| **代码标识** | `USE_HDDM = False` | `USE_HDDM = True` |
| **速度** | 较慢 (1-2 分钟) | 很快 (< 1 秒) |
| **精度** | 数值近似，受 dt 步长影响 | 解析精确 (首通时间理论分布) |
| **功能完整性** | 仅 Wiener 仿真函数 | 仅 HDDM 内置生成器 |
| **输出目录** | `2_Data/.../Wiener/`, `3_Figures/.../Wiener/` | `2_Data/.../HDDM/`, `3_Figures/.../HDDM/` |

**共同点**:
- 均为 **Stim Coding** 范式，操控 DDM **起始点 z** 参数
- 4 个 z 水平: `neutral (0.50)`, `small (0.55)`, `medium (0.60)`, `large (0.65)`
- 30 被试 × 4 条件 × 150 试次 = 18,000 试次
- 输出: 仿真数据 CSV → CRF 计算 → 3 张图 (主 CRF 图, RT 分布, SPE 条形图)
- 对应实验范式: Self Matching Task (Sui, 2012)

---

## 二、`plot_CRF_zbias.py` — 合并版

位于同目录，是前两者的**统一版本**，包含自动回退逻辑：

```python
# 优先 HDDM，不可用时回退到 Wiener
try:
    import hddm
    HAS_HDDM = True
except ImportError:
    HAS_HDDM = False  # 自动使用函数内的 Wiener 回退分支
```

可在 Docker 或本地无 HDDM 环境下运行，输出路径为同一位置。

---

## 三、旧 `ddm_stim_coding` 仿真 Notebook 与新 CRF 分析的核心差异

### 3.1 目的不同

| 维度 | 旧 `simulation_all.ipynb` | 新 CRF 系列 |
|------|---------------------------|-------------|
| **核心目标** | 比较不同模型编码方案对偏差效应的捕捉能力 | 可视化 CRF 曲线，展示决策倾向随 RT 的变化 |
| **工作流** | 生成数据 → 运行 9 种模型变体拟合 → 模型比较 | 生成数据 → 计算 CRF → 绘制 3 张图 |
| **输出** | DIC 比较表、参数恢复图、模式对比 SVG | CRF 曲线图、RT 分布图、SPE 条形图 |

### 3.2 偏差操控方式不同

| 维度 | `Response_Bias/simulation/` | `Stimulus_Bias/simulation/` | 新 CRF 系列 |
|------|----------------------------|----------------------------|-------------|
| **偏差类型** | **z-bias** (Response Bias, 起点偏差) | **dc-bias** (Stimulus Bias, 漂移偏差) | **z-bias** (起点偏差) |
| **条件数量** | 3: `neutral`, `big_bias`, `small_bias` | 3: `neutral`, `big_bias`, `small_bias` | 4: `neutral`, `z_bias_small`, `z_bias_medium`, `z_bias_large` |
| **条件命名** | `neutral` / `big_bias` / `small_bias` | `neutral` / `big_bias` / `small_bias` | `neutral` / `z_bias_small(0.55)` / `z_bias_medium(0.60)` / `z_bias_large(0.65)` |
| **z 操控方式** | `subj_z_base +0.1 / -0.1` (个体基线 ±0.1) | z 固定 0.5，dc 操控 | 固定 z 水平 `[0.50, 0.55, 0.60, 0.65]` + 个体噪声 |
| **subject-level** | 每个被试 z 有组水平随机截距 | 每个被试 dc 有组水平随机截距 | 每个被试 z 有组水平随机截距 |
| **dc 处理** | dc=0 (无漂移偏差) | `dc_mean=0.3` (大), `dc_mean=-0.3` (小) | `dc_mean=0.0` (含个体噪声) |

### 3.3 数据生成引擎差异

| 维度 | 旧 Simulation | 新 CRF 系列 |
|------|---------------|-------------|
| **代码架构** | 独立模块 `data_generation.py`，由 notebook 调用 | 所有代码内嵌在 notebook/脚本中 |
| **HDDM 调用** | `hddm.generate.gen_rand_data()` 唯一方式 | 提供 Wiener 数值仿真 和 HDDM 两种选项 |
| **种子管理** | `seed = 420 + subject_id` | `seed = 420 + subj_id * 1000` |
| **参数采样** | v 在 Stimulus_Bias 有 between-subject 变异 | a, v, t 均有 between-subject 变异 |

### 3.4 旧 Simulation 的数据后处理

旧 notebook 在生成数据后执行**模型比较流水线**:

1. **定义 9 种模型结构** (M1–M9)，对应不同的 HDDMStimCoding 编码方案
2. 对每个模型运行 MCMC 采样
3. **比较 DIC** (Deviance Information Criterion)
4. 提取参数，绘制参数恢复图和模式对比图
5. 输出: `Model_Raw_DIC_Comparison.png`, `Parameter recovery of z bias.png`, `Fig/Mode_Comparison.svg`

新 CRF 系列不做任何模型拟合，只做描述性 CRF 分析和可视化。

### 3.5 HDDM 版本

| 文件 | HDDM 版本 |
|------|-----------|
| 旧 `simulation_all.ipynb` (Response_Bias) | **1.0.1rc0** (最新 master) |
| 旧 `simulation_0.8.ipynb` (Response_Bias) | **0.8.0** (因 z-flipping 实现差异) |
| 旧 `simuation_all.ipynb` (Stimulus_Bias) | **1.0.1rc0** |
| 旧 `simulation_0.8.ipynb` (Stimulus_Bias) | **0.8.0** |
| 新 `plot_CRF_zbias_HDDM.ipynb` | **1.0.1RC** |
| 新 `plot_CRF_zbias_Wiener.ipynb` | **1.0.1RC** (仅用于路径，数据生成不用 HDDM) |

---

## 四、数据流关系图

```
旧 ddm_stim_coding (模型比较)
════════════════════════════════════════════════════════════
Response_Bias/simulation/
  data_generation.py ──→ simulation_all.ipynb ──→ simulated_data.csv
                           │                        ├── 9 模型拟合
                           │                        ├── DIC 比较
                           │                        └── 参数恢复图
                           │
Stimulus_Bias/simulation/
  data_generation.py ──→ simuation_all.ipynb ──→ simulated_data.csv
                                                    ├── 9 模型拟合
                                                    ├── DIC 比较
                                                    └── 参数恢复图

新 HDDM_Stim-Coding_Simulation (CRF 分析)
════════════════════════════════════════════════════════════
plot_CRF_zbias_HDDM.ipynb ──→ simulation_zbias_crf_data.csv ──→ CRF 图 + RT 分布 + SPE 条形图
plot_CRF_zbias_Wiener.ipynb ──→ simulation_zbias_crf_data.csv ──→ CRF 图 + RT 分布 + SPE 条形图
plot_CRF_zbias.py           ──→ simulation_zbias_crf_data.csv ──→ CRF 图 + RT 分布 + SPE 条形图
```

**关联点**:
- 新 CRF 的 `plot_CRF_zbias.py` 文件头部注释明确引用了旧代码:
  - `ddm_stim_coding/Response_Bias/simulation/data_generation.py` (z-bias 数据生成)
  - `ddm_stim_coding/plot_simulation_CRF.ipynb` (CRF 绘图逻辑)
- 旧代码的 CRF 绘图逻辑 (在 `plot_simulation_CRF.ipynb`) 被新代码继承并独立化为可复现流水线

---

## 五、总结

| 项目 | 旧 (ddm_stim_coding) | 新 (HDDM_Stim-Coding_Simulation) |
|------|----------------------|----------------------------------|
| **目标** | 回答"哪种建模方式更好" | 回答"决策倾向如何随 RT 变化" |
| **产出** | 模型比较统计证据 | CRF 可视化曲线 |
| **偏差类型** | z-bias + dc-bias 各一子目录 | 仅 z-bias |
| **条件梯度** | 粗 (3 条件: neutral/small/large) | 细 (4 条件: neutral/small/medium/large) |
| **引擎选择** | 仅 HDDM | Wiener 数值仿真 vs HDDM 解析对比 |
| **架构** | notebook + 独立 .py 模块 | 自包含 notebook，或单文件 .py 脚本 |
| **可复现性** | 需 HDDM + PyMC 环境 | .py 可无 HDDM 运行 (Wiener 回退) |
