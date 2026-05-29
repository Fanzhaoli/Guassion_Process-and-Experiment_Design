# Python_HDDM_Nonmatching — 含 NonMatching 试次的 DDM 拟合工作流

本目录基于 `Python_HDDM` 扩展，将 **NonMatching 试次纳入 DDM 模型**，解决原流程"仅 Matching 试次导致的基线偏差"问题。

---

## 与原版 (Python_HDDM) 的关键差异

| 项目 | Python_HDDM (原版) | Python_HDDM_Nonmatching (本版) |
|:---|:---|:---|
| **数据范围** | 仅 Matching 试次 | **Matching + NonMatching 全量试次** |
| **Response 编码** | `response = Correct` | `response` 统一编码为 "判断为Matching(1) vs 判断为NonMatching(0)" |
| **上边界含义** | 正确Matching判断 | **被试判断为 Matching**（不论判对判错） |
| **下边界含义** | 错误反应 | **被试判断为 NonMatching**（不论判对判错） |
| **CRF 0.5 参考线** | 对应随机正确率 | **对应 Matching/NonMatching 判断无偏好** |
| **输出目录** | `HDDM_Ready/`, `HDDM_Traces/` | `HDDM_Ready_Nonmatching/`, `HDDM_Traces_Nonmatching/` |
| **可视化目录** | `HDDM_Results/` | `HDDM_Results_Nonmatching/` |
| **HDDM 模型** | `depends_on={"v": "identity"}` | **相同**（保持可比性） |
| **数据列** | 5列 | **6列**（新增 `condition` 列） |

---

## Response 编码逻辑

```
核心原则: response=1 统一表示"被试判断为 Matching（上边界）"
          response=0 统一表示"被试判断为 NonMatching（下边界）"

Matching 试次 (condition=1):
  正确答案是 Matching
  → Correct=1: 被试判断正确 = 判断为 Matching → response=1 ✓
  → Correct=0: 被试判断错误 = 判断为 NonMatching → response=0 ✓

NonMatching 试次 (condition=0):
  正确答案是 NonMatching
  → Correct=1: 被试判断正确 = 判断为 NonMatching → response=0 ✓
  → Correct=0: 被试判断错误 = 判断为 Matching → response=1 ✓

代码实现:
  response = Correct                     (if condition==1)
  response = 1 - Correct                 (if condition==0)
```

### 为什么这样编码很重要？

原版（仅 Matching）中 `response_mean` 的含义是"正确率"，在非随机水平下天然 >0.5。
本版中 `response_mean` 的含义是"被试判断为 Matching 的比例"，在无偏好假设下应≈0.5，
因此 **y=0.5 参考线具有明确的行为解释力**。

---

## 目录结构

```
1_Code/Python_HDDM_Nonmatching/
├── step1_prepare_data.py         # Step 1: 数据预处理（含NonMatching）
├── step2_hddm_fit.py             # Step 2: Docker 内 HDDM 拟合
├── step3_extract_params.py       # Step 3: 参数提取 + 可视化
└── README.md                     # 本文件
```

---

## 快速开始（三步走）

### Step 1: 数据预处理

```powershell
python "1_Code\Python_HDDM_Nonmatching\step1_prepare_data.py"
```

**输入**: `2_Data/Real_Data/UnExtact/raw/*.csv`（88 个原始被试数据）

**输出**: `2_Data/Real_Data/HDDM_Ready_Nonmatching/hddm_data_group*_P*_T*_W*.csv`

这一步完成:
- 过滤 `stage='formal'` 的正式试次
- **保留全部试次**（Matching + NonMatching）
- 标记 `condition` 列（1=Matching, 0=NonMatching）
- 统一 response 编码为 "判断为Matching(1) vs 判断为NonMatching(0)"
- 标记遗漏试次（`omission=1`），RT 设为 `T+W`（截止时间）

---

### Step 2: Docker HDDM 拟合

#### 2a. 拉取镜像（首次）

```powershell
docker pull hcp4715/hddm
```

#### 2b. 启动 Docker 容器

```powershell
docker run -it --rm --cpus=4 -v /d/GitHub_programe/GitHub/Guassion-Process-Experiment-Design:/home/jovyan/work -p 8888:8888 hcp4715/hddm jupyter notebook
```

#### 2c. 在 Docker Jupyter 中运行拟合

```python
%run /home/jovyan/work/1_Code/Python_HDDM_Nonmatching/step2_hddm_fit.py
```

**输出**: `2_Data/Real_Data/HDDM_Traces_Nonmatching/*_traces.npz` + `*_stats.csv`

> **预计耗时**: 试次数翻倍（含 Matching+NonMatching），8组 × 11被试 × ~260试次，总计约 45-90 分钟。

---

### Step 3: 参数提取与可视化

```powershell
python "1_Code\Python_HDDM_Nonmatching\step3_extract_params.py"
```

**输出**:
| 文件 | 说明 |
|------|------|
| `2_Data/Real_Data/HDDM_Traces_Nonmatching/all_groups_ddm_params.csv` | 所有条件的 DDM 参数汇总 |
| `3_Figures/HDDM_Results_Nonmatching/ddm_params_by_group.png` | SPE / v / a / t 柱状图 |
| `3_Figures/HDDM_Results_Nonmatching/spe_vs_design_params.png` | SPE_v vs P, T, W 散点图 |

---

## HDDM 就绪数据格式（Step 1 输出）

| 列名 | 类型 | 说明 |
|:---|:---|:---|
| subj_idx | int | 被试索引（0 起始，组内重编号） |
| rt | float | 反应时（秒，遗漏试次 = T+W） |
| **response** | int | **1=判断为Matching, 0=判断为NonMatching** |
| identity | int | 1=self, 0=stranger |
| **condition** | int | **1=Matching试次, 0=NonMatching试次** |
| omission | int | 1=遗漏试次, 0=有效试次 |

---

## HDDM 模型设定

```python
model = hddm.HDDM(
    df,
    depends_on={"v": "identity"},
    include=["v", "a", "t", "z"],
    bias=False,
    p_outlier=0.05,
)
```

| 参数 | 设定 | 说明 |
|:---|:---|---|
| `depends_on={"v": "identity"}` | v 分 Self/Stranger | 与原版相同，保持直接可比性 |
| `include=["v", "a", "t", "z"]` | 4 个核心参数 | 与原版相同 |
| `bias=False` | 不单独估计偏差项 | z 同时承载起始点位置和偏向信息 |
| `p_outlier=0.05` | 5% 异常值 | 稳健估计 |

> **进阶选项（注释备用）**: 如需允许 z 依 identity 变化，修改为 `depends_on={"v": "identity", "z": "identity"}`。这将估计 z_self 和 z_stranger，可区分 Self/Stranger 条件下的起始点偏向差异。

---

## v 的正负号解读

| v 符号 | 含义 |
|:---:|:---|
| **v > 0** | 漂移指向**上边界**（被试倾向于判断为 Matching） |
| **v < 0** | 漂移指向**下边界**（被试倾向于判断为 NonMatching） |
| **\|v\| 大** | 证据累积速度快（决策快、一致性强） |

---

## 与原版的对比分析建议

拟合完成后，建议做以下对比:

1. **SPE_v 稳定性**: 比较原版（仅Matching）与本版（全量试次）的 SPE_v 是否一致
2. **z 参数变化**: 全量试次下 zr_bias 是否更接近 0.5（中性起始点）
3. **CRF 0.5 参考线**: 本版中 y=0.5 代表 Matching/NonMatching 判断无偏好，解读更直观

---

## 8 组实验条件

| 组别 | P | T (ms) | W (ms) | M=T+W | 被试数 |
|------|---|--------|--------|-------|-------|
| 1 | 0 | 30 | 300 | 330 | 11 |
| 2 | 0 | 30 | 600 | 630 | 12 |
| 3 | 120 | 30 | 600 | 630 | 10 |
| 4 | 120 | 80 | 600 | 680 | 11 |
| 5 | 8 | 100 | 1100 | 1200 | 11 |
| 6 | 120 | 500 | 1500 | 2000 | 10 |
| 7 | 120 | 80 | 800 | 880 | 12 |
| 8 | 120 | 80 | 800 | 880 | 11 |

---

## 注意事项

1. **遗漏率高的组** (G1-G2) 拟合可能不收敛，检查 `_stats.csv` 的 R-hat（>1.1 需警惕）
2. **condition 列**未纳入 `depends_on`，不影响参数估计，可用于后续分层分析
3. **路径一致性**: 使用 `parents[2]` 从脚本位置自动推算项目根目录
4. **Docker 内路径**: 容器挂载点在 `/home/jovyan/work`
5. **本套代码与 `Python_HDDM` 完全独立**，互不干扰
