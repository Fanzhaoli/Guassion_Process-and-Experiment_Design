# Guassion-Process-Experiment-Design

## 项目概述
本项目以 **Self-Matching Task** 为核心范式，研究自我优势效应（**Self-Preference Effect, SPE**）如何受到实验设计变量的系统调控。

当前聚焦的设计空间为：
- 练习次数 **P**
- 刺激呈现时间 **T**
- 反应窗口 **W**
- 以及潜在条件维度 **M**

项目主线是将传统“离散条件比较”推进为“连续实验设计空间建模”，并结合 **Drift Diffusion Model (DDM)** 与 **Gaussian Process (GP)**，构建“实验设计 → 心理参数 → 行为数据”的生成框架。

---

## 当前项目进度

### 1) 已完成/较稳定的部分
- **v1**：Sigmoid + DDM 的基础生成框架已建立。
- **v2.4.x**：生成与检查链路较完整，包含恢复检验、数据检查与真实数据对照流程。
- **真实数据**：`2_Data/Real_Data/EXP_data_combined.csv` 与分组原始文件已整理完成。

### 2) 正在推进的部分
- **v2.5**：更偏向新一轮 GP-DDM 生成探索，属于后续扩展版本。
- **v3**：可视为进一步探索 GP 残差/边界/扩展结构的实验分支。

### 3) 目前判断
- **v2.4.x**：相对更接近“可复用、可验证”的稳定版本。
- **v2.5**：更像是新一代实验性分支，而不是完全收敛的终版。
- 整体上，项目已从“能生成数据”进入到“能比较版本、能做验证、能为实验设计服务”的阶段。

---

## 版本演进简表

| 版本族 | 角色 | 主要特征 |
|---|---|---|
| `v1` | 基线模型 | Sigmoid + DDM，验证基础生成思路 |
| `v2.1 ~ v2.3` | 过渡/草稿 | 逐步增强参数映射与生成流程 |
| `v2.4 ~ v2.4.5` | 较稳定主线 | 生成、检查、恢复、真实数据对照较完整 |
| `v2.5` | 新版探索 | 更偏向 GP-DDM 的扩展与调参 |
| `v3` | 研究分支 | 探索 GP 对残差或更复杂结构的捕捉 |
| `S2_gen_data_optimized_cp*` | 另一条优化支线 | Sigmoid 优化版本，用于对照与比较 |

---

## 项目文件夹层级结构

```text
Guassion-Process-Experiment-Design/
├── 1_Code/
│   ├── Python_for_Generate/
│   │   ├── Generate_Data_v1.ipynb
│   │   ├── Generate_Data_v2.ipynb
│   │   ├── Generate_Data_v2.1.ipynb
│   │   ├── Generate_Data_v2.2.ipynb
│   │   ├── Generate_Data_v2.3.ipynb   # 草稿/过渡版本
│   │   ├── Generate_Data_v2.4.ipynb
│   │   ├── Generate_Data_v2.4.2.ipynb
│   │   ├── Generate_Data_v2.4.3.ipynb
│   │   ├── Generate_Data_v2.4.4.ipynb
│   │   ├── Generate_Data_v2.4.5.ipynb
│   │   ├── Generate_Data_v2.5.ipynb
│   │   ├── Generate_Data_v3.ipynb
│   │   ├── S2_gen_data_optimized_cp.ipynb
│   │   ├── S2_gen_data_optimized_cp_v2.ipynb
│   │   ├── simulation_SPE_wk.ipynb
│   │   └── Py/
│   │       ├── Generate_Data_v2.4_runner.py
│   │       ├── Generate_Data_v2.4_recovery.py
│   │       ├── Generate_Data_v2.5_runner.py
│   │       ├── Generate_Data_v2.5_simple.py
│   │       ├── Generate_Data_v2.5_tuning.py
│   │       └── Generate_Data_v2.5_v2.py
│   ├── Python_for_Check/
│   │   ├── step1_generative_checks.ipynb
│   │   ├── step1_generative_checks_v2.ipynb
│   │   ├── Parameter_Recovery.ipynb
│   │   ├── Check_Generate_Data.ipynb
│   │   ├── GP-SPE-Explore-2D.ipynb
│   │   ├── GP-SPE-Explore-3D.ipynb
│   │   ├── Compare_Real_Generated_DDM_Params_v2.4.3.ipynb
│   │   ├── Compare_Real_Generated_DDM_Params_v2.4.3_V2.ipynb
│   │   ├── Compare_Real_Generated_DDM_Params_v2.4.4.ipynb
│   │   ├── Compute_For_ACC/
│   │   │   └── Study3&4_analysis.Rmd
│   │   └── Sigmoid_Optimized/
│   │       ├── S2_gen_data_optimized_cp_v3.ipynb
│   │       ├── S2_gen_data_optimized_cp_v4.ipynb
│   │       ├── S2_gen_data_optimized_cp_v5.ipynb
│   │       ├── S2_gen_data_optimized_cp_v6.ipynb
│   │       ├── S2_gen_data_optimized_cp_v7.ipynb
│   │       ├── S2_gen_data_optimized_cp_v7_alpha_test.ipynb
│   │       └── S2_gen_data_optimized_cp_v8_additive.ipynb
│   └── R_for_Check/
│       └── Check_Generate_Data.Rmd
├── 2_Data/
│   ├── Generate_Data/
│   │   ├── Generate_Data_v2.4_checks/
│   │   ├── Generate_Data_v2.4.2_checks/
│   │   ├── Generate_Data_v2.4.3_checks/
│   │   ├── Generate_Data_v2.4.4_checks/
│   │   ├── Generate_Data_v2.4.5_checks/
│   │   ├── Generate_Data_v2.5/
│   │   ├── S2_gen_data_optimized_cp_v5/
│   │   └── *.csv
│   └── Real_Data/
│       ├── EXP_data_combined.csv
│       └── EXP_data_group*.csv
├── 3_Figures/
├── 4_Reports/
└── README.md
```

---

## 数据组织说明

### 生成数据
- `2_Data/Generate_Data/` 保存各版本模拟数据。
- `v2.4.x` 目录下的 `*_checks` 文件夹，表示该版本已进入较系统的检查/验证阶段。
- `v2.5/` 表示更新一轮的 GP-DDM 生成结果，适合继续调参和比较。

### 真实数据
- `2_Data/Real_Data/EXP_data_combined.csv`：整合后的真实实验数据。
- `2_Data/Real_Data/EXP_data_group*.csv`：按 group 拆分的原始数据。

---

## 为什么重点放在 GP-DDM

GP 的优势不只是“更复杂”，而是更适合做**实验设计建模**：

1. **非线性映射**：可学习 `P/T/W → DDM 参数` 的非线性关系。
2. **不确定性表达**：能同时输出均值与不确定性，方便识别“稳定区”和“边界区”。
3. **边界探索**：适合回答“什么设计最容易放大 SPE”。
4. **更灵活**：比固定 Sigmoid 更适合后续扩展到多参数、多条件和个体差异。

### 建议后续可视化任务
- 画出 `P/T/W → SPE` 的响应面图
- 画出 GP 预测均值与方差热图
- 对比 `Sigmoid vs GP` 的拟合曲线与误差
- 标注“高不确定性区域”作为下一轮实验优先采样区

---

## 下一步推进目标

### 近期目标
- 明确 **GP-DDM 用于实验设计建模** 的主线。
- 固化一个“可复用”的生成/检查版本。
- 把真实数据与生成数据的比较流程整理成标准流程。

### 中期目标
- 建立 GP 的二维/三维可视化结果。
- 找到 SPE 最强的设计区域。
- 做版本比较：`v1` / `v2.4.x` / `v2.5`。

### 长期目标
- 形成“实验设计空间 → 行为结果 → 反向优化”的闭环。
- 将项目收敛成可投稿的研究框架。

---

## 更新日志

### 2026-04-25
- 补充当前项目进度判断。
- 增加完整文件夹层级结构。
- 增加生成数据/真实数据组织说明。
- 增加 GP-DDM 的优势说明与后续可视化任务。
- 增加版本演进与更新路线。

---

## References
- Sui, J., He, X., & Humphreys, G. W. (2012). Perceptual advantages for self-related stimuli: A review. *Current Directions in Psychological Science*, 21(5), 318-323.
