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
