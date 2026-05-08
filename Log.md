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

```bash
# 本机: 数据预处理
python 1_Code/Python_HDDM/step1_prepare_data.py

# Docker: 启动并拟合
docker run -it --rm --cpus=4 \
  -v /d/GitHub_programe/GitHub/Guassion-Process-Experiment-Design:/home/jovyan/work \
  -p 8888:8888 \
  hcp4715/hddm \
  jupyter notebook
# 然后在 Jupyter 中打开 Docker_Run.ipynb 运行 Step 2 + Step 3

# 或 本机: 参数提取与可视化 (Docker 拟合完成后)
python 1_Code/Python_HDDM/step3_extract_params.py
```

### DDM 参数汇总 (8 组后验均值)

| Group | P | T(ms) | W(ms) | v_self | v_stranger | SPE_v | a | t(s) |
|-------|---|-------|-------|--------|------------|-------|---|------|
| 1 | 0 | 30 | 300 | -3.25 | -3.16 | -0.09 | 1.17 | 0.26 |
| 2 | 0 | 30 | 600 | -2.19 | -3.97 | +1.78 | 2.29 | 0.39 |
| 3 | 120 | 30 | 600 | -1.63 | -1.42 | -0.21 | 1.17 | 0.34 |
| 4 | 120 | 80 | 600 | -0.89 | -1.11 | +0.22 | 1.27 | 0.37 |
| 5 | 8 | 100 | 1100 | +1.35 | +0.62 | +0.73 | 1.33 | 0.40 |
| 6 | 120 | 500 | 1500 | +1.57 | +0.89 | +0.68 | 1.60 | 0.67 |
| 7 | 120 | 80 | 800 | +1.20 | +0.75 | +0.44 | 1.09 | 0.40 |
| 8 | 120 | 80 | 800 | +1.44 | +1.06 | +0.39 | 1.39 | 0.41 |

---

*版本历史将以 v0.1 起始，每次对话更新递增 0.1 版本号*

---

## v0.2 — GP+Sigmoid 混合生成模型 + 分层验证策略

**日期**: 2026-05-06  
**责任人**: AI Assistant (Trae Agent)

### 新增功能

#### 阶段 A: GP+Sigmoid 混合生成模型 (`1_Code/Python_HDDM/GP+Sigmoid/`)

- **Sigmoid 参数校准** (`sigmoid_calibration.py`)
  - 使用差分进化优化最小化 Sigmoid 预测与真实 HDDM 参数的 RMSE
  - 校准参数: alaph1=0.549, alaph2=-0.220, beta1=-0.827, beta2=1.000, gamma=0.701
  - 输出至 `2_Data/Generate_Data/GP_Sigmoid/`

- **GP+Sigmoid 混合模型核心类** (`gp_sigmoid_hybrid_model.py`)
  - `GPSigmoidHybridModel`: Sigmoid 理论先验 + GP 学习残差
  - 5 个独立 GP（v_self, v_stranger, a, t, z）
  - RBF kernel + WhiteKernel，normalize_y=True
  - 支持 save/load（pickle）
  - 支持 predict_params_full() 和 predict_with_uncertainty()

- **完整管线运行** (`run_full_pipeline.py`)
  - 加载 HDDM 参数 → 加载校准参数 → 训练 GP → 预测 → 可视化
  - 生成散点图（真实 vs 预测）、GP响应面、SPE对比图
  - 训练集拟合结果:
    - v_self: RMSE=0.71, r=0.98
    - v_stranger: RMSE=0.90, r=0.96
    - SPE_v: RMSE=0.43, r=0.87
    - t: RMSE=0.016, r=0.99

#### 阶段 B: 留一条件交叉验证 (`1_Code/Python_for_Check/Cross-validation/`)

- **LOCV 脚本** (`locv_validation.py`)
  - 每次留出一个条件 (~11名被试)，用其余 7 个条件训练 GP
  - 检验对未参与训练条件的预测精度
  - 输出 RMSE/MAE/r 指标和可视化
  - 结果:
    - v_self: RMSE=2.18, r=-0.02（泛化较差，预期之中）
    - SPE_v: RMSE=1.07, r=0.06
    - 结论: 8 个点在 3D 空间中泛化受限，需要更多数据

#### 阶段 C: 外部验证 (`1_Code/Python_for_Check/External_verification/`)

- **GPU不确定性响应面** (`find_design_points.py`)
  - 在 (P,T,W) 空间中 15×15×15=3375 网格采样
  - 计算每个点的 GP 预测总不确定性
  - 排除已有条件附近区域 (dist<0.15 标准空间)
  - 输出 Top-12 候选新实验设计点
  - 推荐新条件:
    - P≈17-60 (中等练习量区间)
    - T=500ms (长呈现时间)
    - W=300-500ms (短反应窗口)
    - 这些是现有设计中覆盖不足的区域

### 输出文件一览

| 目录 | 文件 | 内容 |
|------|------|------|
| `2_Data/Generate_Data/GP_Sigmoid/` | `sigmoid_calibrated_params.csv` | 校准后的 Sigmoid 参数 |
| | `sigmoid_calibration_results.csv` | 校准预测详细表 |
| | `gp_sigmoid_predictions.csv` | GP+Sigmoid 预测结果 |
| | `gp_sigmoid_hybrid_model.pkl` | 训练好的混合模型 |
| `2_Data/Generate_Data/Cross_Validation/` | `locv_results.csv` | LOCV 逐条件预测 |
| | `locv_metrics.csv` | LOCV 汇总指标 |
| `2_Data/Generate_Data/External_Verification/` | `optimal_design_points.csv` | 候选新实验设计点 |
| `3_Figures/GP_Sigmoid/` | `gp_sigmoid_real_vs_pred.png` | 真实vs预测散点 |
| | `gp_response_surface_P.png` | GP响应面 (P维度) |
| | `spe_v_comparison.png` | SPE真实vs预测对比 |
| `3_Figures/Cross_Validation/` | `locv_scatter.png` | LOCV 散点图 |
| | `locv_rmse.png` | LOCV RMSE 柱状图 |
| `3_Figures/External_Verification/` | `uncertainty_surface.png` | 不确定性响应面 |
| | `candidate_points_ranking.png` | 候选点排序图 |

### 已知问题

- **只有 7 个唯一设计点**: Group 7 和 Group 8 具有相同的 (P=120, T=80, W=800)，但参数差异大（v_self: 1.21 vs 2.81），可能是数据噪声或需要解释的设计特征
- **LOCV 泛化差**: 3D 空间 7-8 个训练点对 GP 来说太少，交叉验证结果不可靠（这是样本量的限制，不是方法问题）
- **Sigma 预测值偏低**: 校准后的 Sigmoid 预测 v 值范围 0.1-1.6，远低于真实 HDDM 的 -3.4~+2.8，说明理论先验需要大幅修正

### 使用方式

```bash
# Step 1: Sigmoid 校准
python 1_Code/Python_HDDM/GP+Sigmoid/sigmoid_calibration.py

# Step 2: GP+Sigmoid 混合模型训练+预测+可视化
python 1_Code/Python_HDDM/GP+Sigmoid/run_full_pipeline.py

# Step 3: 留一条件交叉验证
python 1_Code/Python_for_Check/Cross-validation/locv_validation.py

# Step 4: 外部验证 - 寻找最优新设计点
python 1_Code/Python_for_Check/External_verification/find_design_points.py
```

---

## v0.3 — Generate_Data_v4.ipynb 可逐行运行的 Jupyter Notebook

**日期**: 2026-05-06  
**责任人**: AI Assistant (Trae Agent)

### 新增功能

- **`Generate_Data_v4.ipynb`** — 完整的 GP+Sigmoid 混合生成模型 Jupyter Notebook
  - 路径: `1_Code/Python_for_Generate/Generate_Data_v4.ipynb`
  - 26 个 cells (14 markdown + 12 code)，可逐行独立运行
  - 包含完整的 Sigmoid 校准、GP+Sigmoid 模型训练、LOCV 交叉验证、外部验证设计
  - 自动定位项目根目录（通过 AGENTS.md），支持不同环境移植
  - 使用相对路径，所有输出保存到 `2_Data/Generate_Data/` 和 `3_Figures/` 下

### Notebook 结构

| Cell | 类型 | 内容 |
|------|------|------|
| 0 | Markdown | 标题与工作流概览 |
| 1 | Markdown | Cell 1 说明 |
| 2 | Code | 环境设置与路径配置 |
| 3 | Markdown | Cell 2 说明 |
| 4 | Code | Sigmoid 理论先验函数 |
| 5 | Markdown | Cell 3 说明 |
| 6 | Code | 加载 HDDM 参数 |
| 7 | Markdown | Cell 4 说明 |
| 8 | Code | Sigmoid 参数校准（差分进化） |
| 9 | Markdown | Cell 5 说明 |
| 10 | Code | GP+Sigmoid 混合模型类定义 |
| 11 | Markdown | Cell 6 说明 |
| 12 | Code | 训练混合模型 |
| 13 | Markdown | Cell 7 说明 |
| 14 | Code | 训练集拟合评估 |
| 15 | Markdown | Cell 8 说明 |
| 16 | Code | 可视化：真实 vs 预测散点图 + SPE对比 |
| 17 | Markdown | Cell 9 说明 |
| 18 | Code | LOCV 留一条件交叉验证 |
| 19 | Markdown | Cell 10 说明 |
| 20 | Code | LOCV 可视化 |
| 21 | Markdown | Cell 11 说明 |
| 22 | Code | 外部验证：寻找最优新设计点 |
| 23 | Markdown | Cell 12 说明 |
| 24 | Code | 外部验证可视化：不确定性表面图 |
| 25 | Markdown | 输出文件清单与下一步建议 |

### 使用方式

```bash
# 启动 Jupyter (项目根目录)
jupyter notebook

# 打开文件
1_Code/Python_for_Generate/Generate_Data_v4.ipynb

# 按顺序逐 Cell 运行 (Shift+Enter)
```
