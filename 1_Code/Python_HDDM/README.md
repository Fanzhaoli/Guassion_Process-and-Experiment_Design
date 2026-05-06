# Python_HDDM — HDDM 参数拟合工作流

本目录包含使用 DockerHDDM 对 Self-Matching Task (SMT) 真实数据进行漂移扩散模型拟合的完整工具链。

---

## 目录结构

```
1_Code/Python_HDDM/
├── Docker_User.html              # Docker 命令生成工具（浏览器打开）
├── HDDM_Ready_Workflow.ipynb     # ⭐ 主工作流 Notebook（推荐入口）
├── step1_prepare_data.py         # Step 1 独立脚本：数据预处理
├── step2_hddm_fit.py             # Step 2 独立脚本：Docker 内拟合（容器内运行）
├── step3_extract_params.py       # Step 3 独立脚本：参数提取+可视化
└── README.md                     # 本文件
```

---

## 快速开始（三步走）

### Step 1: 数据预处理

```powershell
# 在本机项目根目录执行
python "1_Code\Python_HDDM\step1_prepare_data.py"
```

**输入**: `2_Data/Real_Data/UnExtact/raw/*.csv`（88 个原始被试数据）

**输出**: `2_Data/Real_Data/HDDM_Ready/hddm_data_group*_P*_T*_W*.csv`

这一步完成：
- 过滤 `stage='formal'` 的正式试次
- 仅保留 Matching 试次（`(circle,self)` 和 `(square,stranger)`）
- 标记遗漏试次（`omission=1`），RT 设为 `T+W`（截止时间）
- 生成 HDDM 所需的标准格式（`subj_idx`, `rt`, `response`, `identity`, `omission`）

---

### Step 2: Docker HDDM 拟合

#### 2a. 拉取镜像（首次）

```powershell
docker pull hcp4715/hddm
```

#### 2b. 启动 Docker 容器

**方法 A — 使用 Docker_User.html（推荐）**

1. 浏览器打开 `Docker_User.html`
2. 输入项目根目录路径（如 `D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design`）
3. 点击生成命令，复制到 **CMD（管理员）** 中运行

**方法 B — 手动命令**

```powershell
docker run -it --rm --cpus=4 -v /d/GitHub_programe/GitHub/Guassion-Process-Experiment-Design:/data -p 8888:8888 hcp4715/hddm jupyter notebook
```

> **注意**: 路径转换规则：`D:\path\to\project` → `/d/path/to/project`（盘符小写，反斜杠改正斜杠）

#### 2c. 在 Docker Jupyter 中拟合

容器启动后，终端会输出 `http://127.0.0.1:8888/?token=...`，在浏览器打开。

在 Jupyter 中新建 Notebook，复制 `HDDM_Ready_Workflow.ipynb` 中 Step 2 的代码运行。

或直接运行脚本：

```python
%run /data/1_Code/Python_HDDM/step2_hddm_fit.py
```

**输出**: `2_Data/Real_Data/HDDM_Traces/*_traces.npz` + `*_stats.csv`

> **预计耗时**: 8 组数据，每组的 11 名被试 × ~130 试次，MCMC 3000 次采样，总计约 30-60 分钟（取决于 CPU）。

---

### Step 3: 参数提取与可视化

```powershell
# Docker 拟合完成后，回到本机执行
python "1_Code\Python_HDDM\step3_extract_params.py"
```

**输出**:
| 文件 | 说明 |
|------|------|
| `2_Data/Real_Data/HDDM_Traces/all_groups_ddm_params.csv` | 所有条件的 DDM 参数汇总 |
| `3_Figures/HDDM_Results/ddm_params_by_group.png` | SPE / v / a / t0 柱状图 |
| `3_Figures/HDDM_Results/spe_vs_design_params.png` | SPE_v vs P, T, W 散点图 |

---

## 数据格式说明

### 原始数据列（输入）

| 列名 | 说明 |
|------|------|
| groupID | 实验组别 (1-8) |
| subjectID | 被试编号 |
| stage | 阶段（practice / formal） |
| P | 练习次数 |
| T | 刺激呈现时间（秒） |
| W | 反应窗口（秒） |
| Shape | 刺激形状（circle / square） |
| Label | 身份标签（self / stranger） |
| CorrectKey | 正确按键（f / j） |
| Response | 实际按键（f / j / NA） |
| RT | 反应时（秒，NA 表示遗漏） |
| Correct | 正确与否（1/0，遗漏为 0） |

### HDDM 就绪格式（Step 1 输出）

| 列名 | 说明 |
|------|------|
| subj_idx | 被试索引（0 起始，组内重编号） |
| rt | 反应时（秒，遗漏试次 = T+W） |
| response | 1=正确, 0=错误 |
| identity | 1=self, 0=stranger |
| omission | 1=遗漏试次, 0=有效试次 |

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
| 7* | 120 | 30 | 800 | 830 | 12 |
| 8* | 120 | 80 | 800 | 880 | 11 |

> *注: Group 7 的 T 值在原始数据中记录为 80ms（与 Group 8 相同），请核实数据来源。

---

## 注意事项

1. **遗漏率高的组**: Group 1 遗漏率约 72%（W=300ms 极短），Group 2 约 52%。这些组 DDM 拟合可能不收敛，检查 `_stats.csv` 的 R-hat（>1.1 需警惕）。

2. **仅 Matching 试次**: SMT 范式中 SPE 的传统定义基于 Matching 条件，NonMatching 另行分析。

3. **路径一致性**: `.py` 脚本使用 `parents[2]` 从脚本位置自动推算项目根目录；`.ipynb` 使用 `while` 循环向上查找 `AGENTS.md`。无论从哪里启动 Jupyter 都能正确工作。

4. **Docker 内路径**: 容器内的挂载点在 `/data`，所有路径相对于 `/data` 编排。

---

## 后续分析

DDM 参数提取后，将用于 GP-SPE 建模的下游任务：

```
HDDM 参数 (v_self, v_stranger, a, t0)
    │
    ├──→ Sigmoid 理论先验（理论假设）
    ├──→ GP 残差建模（经验修正）
    └──→ 生成模型 = Sigmoid + GP(残差)
```

详见项目根目录的 `可行性分析.md`。
