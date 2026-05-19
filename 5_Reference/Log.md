# 项目更新日志 (Log.md)

> **项目**: GP-SPE 实验设计优化  
> **目录**: `D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design`

---

## v0.1 — HDDM Docker 工作流初始版本

**日期**: 2026-05-06  
**责任人**: AI Assistant (Trae Agent)

### 新增功能

- **Step 1: 数据预处理** (`step1_prepare_data.py`)
  - 从 88 个原始 CSV (`UnExtact/raw/`) 读取数据
  - 按 groupID 分发至 8 个实验条件
  - 过滤 Matching 试次 (circle+self / square+stranger)
  - 遗漏试次 RT 设为 T+W 截止时间，标记 omission=1
  - 输出 HDDM 就绪 CSV 至 `2_Data/Real_Data/HDDM_Ready/`

- **Step 2: Docker HDDM 层级模型拟合** (`step2_hddm_fit.py` + `Docker_Run.ipynb`)
  - 使用 `hcp4715/hddm` Docker 镜像
  - `depends_on={"v": "identity"}` 区分 Self/Stranger 漂移率
  - 每条件独立拟合层级 DDM (3000 draws, 500 burn)
  - 遗漏试次以 RT=deadline, response=0 参与拟合
  - 双格式保存迹线: pickle (.pkl) + numpy (.npz)
  - 输出统计摘要至 `2_Data/Real_Data/HDDM_Traces/*_stats.csv`

- **Step 3: 参数提取与可视化** (`step3_extract_params.py`)
  - 从 `*_stats.csv` 提取各条件 DDM 参数 (v_self, v_stranger, a, t, z)
  - 计算 SPE_v = v_self - v_stranger
  - 生成两张图表: 参数柱状图 + SPE vs 设计参数散点图
  - 输出至 `3_Figures/HDDM_Results/`

### 功能完善

- 修正了 Docker 挂载路径: 挂载项目根目录而非 Python_HDDM 子目录
- 移除了 `Python_HDDM/2_Data/` 重复文件夹，统一使用项目根目录 `2_Data/`
- 参数提取改为从 stats.csv 直接读取（不再依赖脆弱的 .npz 迹线文件）

### 问题修复

- **修复 .npz 迹线文件为空的问题**: `model.get_traces()` 返回的对象无法直接序列化为 .npz。改为使用 `np.asarray(val, dtype=float).flatten()` 显式转换，同时保存 pickle 格式作为备用
- **修复 all_groups_ddm_params.csv 缺少 DDM 参数**: 之前因 trace 加载失败导致参数列为空。改为从 stats.csv 直接读取
- **修复文件夹结构混乱**: Docker 挂载 `Python_HDDM` 导致 `2_Data` 出现在代码目录下。Docker 改为挂载项目根目录

### 已知问题

- **Group 1 (W=300ms) 遗漏率约 72%**，SPE_v 接近零 (-0.09)，DDM 参数估计可能不可靠
- **Group 2 (W=600ms) SPE_v 异常 (+1.78)**，远高于其他组，可能是高遗漏率导致的拟合偏差
- **迹线保存仍可能不稳定**: 如果 `get_traces()` 返回空或非数组对象，仅保存 stats.csv。Step 3 可以仅依赖 stats.csv 运行

### 使用方式

```

---

## v0.4 — GP+Sigmoid Cleaned 验证管线 + 项目路线图

**日期**: 2026-05-09  
**责任人**: AI Assistant (Trae Agent)

### 新增功能

#### A. 完整 6 步 Cleaned 验证管线 (`run_cleaned_validation_pipeline.py`)

- **路径**: `1_Code/Python_HDDM/GP+Sigmoid/run_cleaned_validation_pipeline.py`
- **设计目标**: 纠正初版元数据问题，建立从诊断到候选推荐的全流程自动化

**Step 0: 真实数据行为摘要**
- 加载 `EXP_data_combined.csv`，仅保留 Matching 试次
- 按 group × identity (self/stranger) 计算 RT/ACC/遗漏率
- 输出: `step0_real_matching_summary_by_condition.csv`, `step0_real_matching_summary_by_identity.csv`

**Step 1: HDDM 参数诊断**
- 交叉验证 HDDM 参数表中的 (P, T, W) 与实际行为数据元数据
- 发现: G7 元数据 T=30ms 与行为 T=80ms 不匹配
- 标记: G1/G2 高遗漏率 (>50%)，G1-G3 v CI 超宽 (>11)
- 输出: `step1_hddm_parameter_diagnostics.csv`

**Step 2: 敏感性分析（6 种策略）**
- `original_keep_all` — 原始 HDDM 参数不作修改
- `original_merge_duplicate_designs` — 合并重复设计 (G7+G8)
- `metadata_corrected_keep_all` — 仅纠正元数据
- `metadata_corrected_aggregated` — 纠正+合并
- `metadata_corrected_drop_high_omission` — 纠正+丢弃高遗漏组 (>50%)
- `metadata_corrected_drop_any_flagged` — 纠正+丢弃所有标记组
- 输出: `step2_sensitivity_*.csv` (6 个文件)

**Step 3: 清洗后参数表**
- 将行为元数据的 (P, T, W) 覆盖到 HDDM 参数表
- 按设计条件聚合（合并重复设计的加权平均）
- 输出: `step3_cleaned_hddm_params_main.csv`

**Step 4: GP+Sigmoid 训练 + LOCV**
- 使用清洗后参数重新校准 Sigmoid（差分进化）
- 校准结果: alaph1=0.199, alaph2=-0.404, beta1=-0.826, beta2=1.000, gamma=0.640
- 训练 GPSigmoidHybridModel
- LOCV (Leave-One-Condition-Out) 交叉验证
- 结果:
  - 训练拟合: r≈1.00 (in-sample 完美)
  - LOCV: v_self r=-0.09⚠️, a r=-0.44⚠️, SPE r=-0.09⚠️
  - 警示: 8 个点不足以支持 GP 泛化
- 输出: `step4_gp_sigmoid_model_cleaned_main.pkl`, `step4_locv_results_cleaned.csv`

**Step 5: 行为层面验证**
- 使用 GP+Sigmoid 预测的 DDM 参数进行 DDM 试次级模拟
- 将模拟行为与真实行为对比:
  - Correct RT: r=0.981 ✅
  - ACC: r=0.968 ✅
  - Omission Rate: r=0.923 ✅
  - SPE RT: r=0.843 ✅
- 结论: Sigmoid 理论先验保证了行为层的稳健重建
- 输出: `step5_behavior_validation_metrics.csv`, `step5_simulated_matching_trials.csv`

**Step 6: 候选实验设计点**
- 25×25×25 = 15625 网格采样 (P,T,W) 空间
- 计算 GP 预测总不确定性
- 排除已有设计点邻近区域 (标准化空间 dist<0.18)
- Top-20 候选按不确定性排序
- 最高不确定性区域: P≈15-45, T=500ms, W=300-350ms
- 不确定性分解: v_self/v_stranger 各约 2.1 (占总不确定性 ~86%), a≈0.45, t≈0.13, z≈0.13
- 输出: `step6_candidate_design_points.csv`

#### B. 产出文件

| 目录 | 关键文件 |
|:---|:---|
| `2_Data/Generate_Data/GP_Sigmoid_Cleaned/` | 25+ 输出文件 (诊断/敏感性/LOCV/行为验证/候选点) |
| `3_Figures/GP_Sigmoid_Cleaned/` | 5 张可视化图 (训练拟合/LOCV/行为验证散点+柱状图/候选点) |
| `.` (项目根目录) | `RoadMap.md` — 完整路线图 (Phase 0-5) |

#### C. 两版 GP+Sigmoid 管线对比

| 维度 | 初版 (GP_Sigmoid/) | Cleaned版 (GP_Sigmoid_Cleaned/) |
|:---|:---|:---|
| 脚本数量 | 3 个独立 .py | 1 个综合管线 |
| 元数据纠正 | ❌ 否 | ✅ 用行为数据校正 G7 T 值 |
| 交叉验证 | ❌ | ✅ LOCV |
| 敏感性分析 | ❌ | ✅ 6 种策略 |
| 行为验证 | ❌ | ✅ DDM 试次级模拟 |
| 候选推荐 | ❌ | ✅ GP 不确定性驱动 |
| 输出文件数 | 4 | 25+ |
| 推荐使用 | 仅作历史参考 | **主版本** ✅ |

### 关键发现

1. **alaph1 校准后远低于默认值**: 从 1.5 降至 0.199，说明真实数据中 self 的相对漂移率优势远小于理论假设
2. **beta1 变号**: 从预测的 +0.2 变为 -0.826，高 M 条件下决策边界反而不增
3. **base_scale_a 触及上界 (10.0)**: a 的 Sigmoid 参数化可能需要重新考虑函数形式
4. **8 个条件不足以支持 GP 泛化**: LOCV 负相关普遍，需要至少新增 4-6 个条件

### 使用方式

```bash
# 一键运行完整 6 步清洗管线
python 1_Code/Python_HDDM/GP+Sigmoid/run_cleaned_validation_pipeline.py

# 可选参数
python 1_Code/Python_HDDM/GP+Sigmoid/run_cleaned_validation_pipeline.py \
  --seed 42 \
  --sigmoid-maxiter 300 \
  --candidate-topn 20
```

### 已知问题

- 初版 (`run_full_pipeline.py` + `sigmoid_calibration.py`) 已被 Cleaned版取代，保留作为历史参考
- G7 T 值需人工确认（元数据 30ms vs 数据 80ms），当前 Cleaned版按行为数据 T=80ms 处理
- Cleaned版 G7 实际 T 值仍取 30ms（来自行为数据汇总），需在确认后手动修正

---

## v0.5 — 路线图制定与深度方法论评估

**日期**: 2026-05-09  
**责任人**: AI Assistant (Trae Agent)

### 新增

- **`RoadMap.md`** — 项目完整路线图
  - Phase 0: 数据采集 + 基线 Sigmoid+DDM
  - Phase 1: GP 角色定位（代理模型/数据层混合/残差捕捉）
  - Phase 2: HDDM Docker 拟合 + 参数提取
  - Phase 3: GP+Sigmoid 建模 + LOCV + 行为验证
  - Phase 4: 实验设计优化 + 候选点推荐
  - Phase 5: 论文撰写与产出（规划中）
  - 含 ✅ 已完成 / ⚠️ 进行中 / 📋 待执行 状态标记

### 方法论澄清

- **GP 在 DDM 参数层 vs 数据生成层**:
  - 参数层 (当前方案): Sigmoid 预测 (v,a,t,z) → GP 学习参数残差 → DDM 模拟行为
  - 数据层 (v2.4): Sigmoid→DDM模拟→行为 → GP 学习行为残差
  - 结论: 参数层更优（低维、可解释、理论约束强）

- **DDM 参数合理性评估** (参照 Ratcliff & McKoon, 2008; Ratcliff et al., 2016):
  - v: G4-G8 合理 (+0.6~+2.8), G1-G3 异常负值 (-3.4~-1.7) — 高遗漏率污染
  - a: G2-G7 合理 (1.1~1.5), G1(2.01)/G8(2.42) 稍高
  - t: 整体合理 (0.26~0.66s), G6(0.66s) 在 T=500ms 下可解释
  - z: 整体合理 (0.45~0.75)

- **当前核心瓶颈**: 仅 8 个设计条件，需新增 4-6 个实验条件才能支持 GP 有效泛化

### 下一步建议

1. 🔴 确认 G7 真实 T 值（30ms 还是 80ms）
2. 🔴 新增实验条件（按 Step 6 候选点推荐 + RoadMap 建议）
3. 🟡 决定 G1/G2 从建模中是否排除
4. 🟢 排除后重新运行完整管线并更新所有指标

---

## v0.6 — RoadMap 参数表完善 + Sigmoid 优化方法论澄清 + Docker Notebook 整合

**日期**: 2026-05-09  
**责任人**: AI Assistant (Trae Agent)

### 新增

#### A. RoadMap.md Phase 0.4: Sigmoid 参数全表与可优化性分析

- 列出 Sigmoid 生成模型的**所有参数**：
  - **6 个显式参数**: alaph1, alaph2, beta1, beta2, gamma, t0
  - **10 个隐性/硬编码参数**: T_0, k_T, M_0, k_a, P1, k_min, k_max, P0, base_scale_v, base_scale_a
- 可优化性分类：✅ 已优化（7个）/ 🟡 可选未做（8个）/ ❌ 不应优化（1个）

#### B. RoadMap.md Phase 3.6: 关键发现与参数解读

- **3.6.1 Sigmoid 校准参数解读**：逐参数解释校准结果的心理/方法论含义
  - alaph1=0.199（远低于默认 1.5）：self 的相对优势仅 20% 而非 150%
  - beta1=-0.826（方向反转）：高 M 条件下边界反而降低
  - base_scale_a=10.0（触及上界）：Sigmoid 无法产生足够大的 a 值
- **3.6.2 需要解决的模型问题**：6 个问题按严重程度排列
- **3.6.3 行为层面验证积极信号**：RT r=0.981, ACC r=0.968, 遗漏率 r=0.923, SPE r=0.843

#### C. Docker_Run.ipynb 整合优化

- 将 `HDDM_Ready_Workflow.ipynb` 的 **Step 1 数据预处理** 整合至 `Docker_Run.ipynb`
  - 原始数据路径：`UnExtact/raw/`（Docker 挂载后直接可访问）
  - 预处理逻辑：过滤 Matching 试次 → 标记遗漏 → 输出 HDDM 就绪 CSV
  - 支持跳过（若 `HDDM_Ready/` 下已有文件）
- 更新工作流步骤表，所有步骤统一在 Docker 内完成
- 更新输出文件清单，加入 Step 1 产物

### 删除

- ❌ **`HDDM_Ready_Workflow.ipynb`** — 已被 `Docker_Run.ipynb`（整合版）完全取代
  - 删除原因：仅用于规划阶段，Step 2 在 Docker 外无法运行（PermissionError），实际运行的是 `Docker_Run.ipynb`

### 关键澄清

- **base_scale_v 和 base_scale_a 的来源**：这两个参数在原始 S2 代码中**确实存在**，只是以隐式方式出现（直接写 `* 3`）。校准代码将它们提取为显式可调参数。用户之前的疑问"原始 Sigmoid 中没有看到"是因为它们是硬编码的魔法数字，而非有名字的变量。

---

## v0.7 — v9 Sigmoid 系统性优化 (5 策略差分进化 + 全参数校准)

**日期**: 2026-05-17  
**责任人**: AI Assistant (Trae Agent)

### 新增功能

#### A. v9_Sigmoid_Optimization.ipynb — 系统性优化 Notebook

- **路径**: `1_Code/Python_for_Check/Sigmoid_Optimized/v9_Sigmoid_Optimization.ipynb`
- **设计目标**: 基于 RoadMap 0.4 参数全表，对 Sigmoid 生成的 **16 个参数（6 显式 + 10 隐式）** 进行系统性优化
- **与 v3-v8 的关键区别**:
  - v3-v8 是试次级 DDM 模拟器（随机 P,T,W → DDM 模拟 → RT 分布检查）
  - v9 是参数级校准器（真实 HDDM 参数 → Sigmoid 函数优化 → 行为验证）
  - v9 与 `sigmoid_calibration.py` 对齐方法论，但扩展了参数范围和优化策略

#### B. 5 策略系统性优化

| 策略 | 方法 | N数据 | N参数 | 关键结果 |
|:---|:---|:---:|:---:|:---|
| **S1** | Cleaned 基线复现 (8组, a_bound=10) | 8 | 7 | RMSE_v=1.95, RMSE_a=0.47, base_scale_a=10.0 (触界) |
| **S2** | 扩展边界 (a_bound→25) | 8 | 7 | RMSE_a=0.39, base_scale_a=22.06, base_scale_v=0.79 |
| **S3** | 排除高遗漏 G1,G2 (6组) | 6 | 7 | RMSE_v_self=1.36, rho_SPE_v=-0.12 (不良) |
| **S4** | 加权多目标 (w_v=0.4, w_spe=0.4, w_a=0.2) | 8 | 7 | 与 S2 结果相近，未显著改善 SPE 匹配 |
| **S5** | 全参数优化 (11参数, 含T_0/k_T/M_0/k_a) | 6 | 11 | **RMSE_v_self=1.12, RMSE_v_stranger=1.09 (最优)** |

#### C. 最优参数 (S5, 排除 G1,G2)

| 参数 | 默认值 | Cleaned(RoadMap) | v9最优(S5) | 方向 |
|:---|:---:|:---:|:---:|:---:|
| alaph1 | 1.50 | 0.199 | **1.112** | ↓ |
| alaph2 | -0.40 | -0.404 | **0.401** | ↑ (变号) |
| beta1 | 0.20 | -0.826 | **-0.434** | ↓ |
| beta2 | 0.00 | 1.000 | **0.281** | ↑ |
| gamma | 0.20 | 0.640 | **0.438** | ↑ |
| base_scale_v | 3.00 | 1.326 | **0.946** | ↓ |
| base_scale_a | 3.00 | 10.000 | **2.806** | ↓ |
| T_0 | 100 | 100(默认) | **63.7** | ↓ |
| k_T | 0.01 | 0.01(默认) | **0.100** | ↑ |
| M_0 | 600 | 600(默认) | **565.4** | ≈ |
| k_a | 0.01 | 0.01(默认) | **0.018** | ↑ |

#### D. 可视化产出 (5 张图表)

- `v9_prediction_vs_actual_scatter.png` — DDM 参数预测 vs 真实值散点图 (含 RMSE/r/ρ)
- `v9_residual_analysis.png` — 各条件参数残差柱状图
- `v9_strategy_parameter_comparison.png` — 5 策略间核心参数对比
- `v9_condition_level_comparison.png` — 条件级别折线图 (按 M_ms 排序)
- `v9_strategy_metrics_comparison.png` — 策略间 RMSE/Spearman ρ 对比

#### E. 数据产出

| 文件 | 内容 |
|:---|:---|
| `strategy_comparison_v9.csv` | 5 策略完整对比表 |
| `best_predictions_v9.csv` | 最优策略 (S5) 对各条件的预测 + 残差 |
| `best_params_v9.csv` | 最优参数表 |
| `parameter_comparison_before_after_v9.csv` | 默认/Cleaned/v9 三版参数对比 |
| `v9_final_report.txt` | 完整性能评估报告 |

### 关键发现

1. **Sigmoid 模型无法产生负 v 值（结构局限）**:
   - 所有 Sigmoid 组件 (v_T, v_P, a_0) 输出 ∈ [0,1]
   - 乘法组合 + 正缩放因子 → v 永远 ≥ 0
   - G1, G2, G4 的真实 HDDM v 为负值，Sigmoid 完全无法匹配
   - G1: HDDM=-3.39, Sigmoid=0.027 → 残差=-3.41

2. **alaph2 变号（-0.4 → +0.401）**:
   - v9 最优解中 stranger 反而获得 v 增强（！）
   - 这是因为模型需要 alaph1 >> alaph2 来产生 SPE_v，而非通过 alaph2 < 0
   - 这与原始设计意图（alaph2 为负以削弱 stranger）不一致

3. **base_scale_a 不再触界**: 通过 S5 的全参数优化（包含 M_0, k_a），base_scale_a 降至 2.81，无需极大值

4. **T_0 下移至 63.7ms**（默认 100ms）: T→v 的 Sigmoid 中点左移，说明在更短的 T 下即可达到半激活

5. **k_T 增至 0.1**（默认 0.01）: T→v 的陡峭度增加 10 倍，T 对 v 的影响更加敏感

6. **排除 G1,G2 对结果影响显著**: S3 (6组) 的 rho_SPE_v 反而变负 (-0.12)，说明剩余 6 组的 SPE_v 变化模式更复杂

### 使用方式

```bash
# 运行独立优化脚本 (生成所有结果文件)
python 1_Code/Python_for_Check/Sigmoid_Optimized/run_v9_optimization.py

# 或在 Jupyter 中打开 notebook 逐 Cell 运行
jupyter notebook 1_Code/Python_for_Check/Sigmoid_Optimized/v9_Sigmoid_Optimization.ipynb
```

### 已知局限

- 差分进化未完全收敛 (所有策略 `Converged: False`)，提高 maxiter 可进一步优化
- Sigmoid 无法产生负 v 值是结构性局限，可能需要重新参数化（如引入偏移项）
- 仅 6-8 个数据点校准 7-11 个参数，过参数化风险高
- S5 的 alaph2 变号为正值，需从理论角度评估其合理性
