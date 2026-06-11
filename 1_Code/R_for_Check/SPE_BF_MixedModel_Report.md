# SPE 贝叶斯因子分析报告 — 混合效应模型（升级版）

> **日期**: 2026-06-08 | **脚本**: [SPE_BF_MixedModel.Rmd](SPE_BF_MixedModel.Rmd)  
> **项目**: 基于高斯过程的实验设计优化  
> **前版**: 简单配对 t 检验 BF → [SPE_BF_Analysis.Rmd](SPE_BF_Analysis.Rmd)

---

## 一、升级内容

### 假设升级

| 版本 | 研究假设 | 方法 |
|:---|:---|:---|
| v1 (旧) | "任务中存在自我优势效应" | 配对 t 检验 BF（不分条件） |
| **v2 (新)** | **"在不同实验空间参数下，任务中存在自我优势效应"** | **混合效应模型 BF（lmBF）** |

### 为什么升级？

1. **匹配假设表述**：原假设只是笼统地检验"有 SPE"，没有利用"跨条件"这一关键限定
2. **复现 JASP 输出**：使用 `BayesFactor::lmBF` + 随机效应（随机截距 + 随机斜率），与 JASP 软件默认设置一致
3. **控制组间变异**：在模型中包含 `groupID_f` 作为协变量，确保检验的是"控制条件差异后的 Label 效应"
4. **序贯分析增强**：复用了 `Function_bf_avo.R` 中的序贯 BF 计算逻辑

---

## 二、模型架构

### 零模型 (H₀)
```
dv ~ 1 + subj_idx + groupID_f
```
- 随机截距: `subj_idx`（被试间基线差异）
- 固定效应: `groupID_f`（组间差异）
- **不含** Label（假设 self = stranger）

### 备择模型 (H₁)
```
dv ~ 1 + subj_idx + groupID_f + Label + Label:subj_idx
```
- 新增固定效应: `Label`（self vs stranger）
- 随机斜率: `Label:subj_idx`（每个被试的 SPE 可以不同）

### BF₁₀ (Label) = evidence for Label model / null model

### 模型设定参数（与 JASP 一致）

| 参数 | 值 | 说明 |
|:---|---:|:---|
| `rscaleFixed` | 0.5 | JZS 先验 scale（固定效应） |
| `rscaleRandom` | 1 | JZS 先验 scale（随机效应） |
| `iterations` | 10000 | MCMC 迭代次数 |

---

## 三、分析流程

```
原始数据 (88人, 8组, ~88000试次)
    │
    ├─ 过滤: stage == "formal"
    ├─ 识别: matching 试次 (基于 subjectID 奇偶性)
    ├─ 聚合: 被试 × Label 水平 (acc, rt_mean)
    │
    ├─ §4 核心分析: Label 主效应 BF
    │     └─ 混合模型: dv ~ subj_idx + groupID_f + Label + Label:subj_idx
    │
    ├─ §5 序贯分析: 逐步增加被试时的 BF 累积
    │     └─ 模拟收数据顺序, 寻找 BF 首次达 10 (或 0.1) 的 N
    │
    ├─ §6 分组分析: 每组内部独立计算 Label BF
    │     └─ 回答"哪个参数组合下有最稳定的 SPE 证据"
    │
    ├─ §7 交互分析: Label × groupID_f 交互 BF
    │     └─ 回答"SPE 强度是否因实验条件而异"
    │
    └─ §8-9 可视化 + 汇总
```

---

## 四、分析方法对比

### v1 (配对 t 检验 BF)

```r
ttestBF(x = self_acc, y = stranger_acc, paired = TRUE)
```

- 简单直接，但忽略了组间差异
- 将 8 组被试混在一起，无法回答"跨条件"的问题

### v2 (混合效应模型 BF)

```r
lmBF(acc ~ subj_idx + groupID_f + Label + Label:subj_idx,
     whichRandom = "subj_idx")
```

- 显式建模被试间变异（random intercept + random slope）
- 控制组间差异后检验 Label 效应
- 直接对应"在不同实验空间参数下"的假设
- 序贯分析可以在控制条件下跟踪 BF 累积

---

## 五、新增分析内容

### 5.1 Label × Group 交互 BF

检验 **SPE 的强度是否随实验条件变化**：

| BF₁₀ (交互) | 解读 |
|:---|:---|
| > 3 | SPE 强度因实验条件而异（设计参数调节 SPE） |
| 1/3 – 3 | 证据不明确 |
| < 1/3 | SPE 在不同条件下强度一致 |

这与 RoadMap 的核心问题直接对应：**实验设计空间参数是否影响 SPE**。

### 5.2 每组独立 Label BF

对每组 (G1-G8) 分别计算混合效应模型 BF：
- 识别哪些参数组合下有最强的 SPE 证据
- 为后续 GP 建模提供条件层面的证据强度参考

### 5.3 序贯停止标准

沿用用户设定的停止规则：
- BF₁₀ ≥ 10 → 停止，有 strong evidence for SPE
- BF₁₀ ≤ 0.1 → 停止，有 strong evidence against SPE
- 达到每组 70 人 → 停止，达到最大样本量

---

## 六、使用说明

1. 在 RStudio 中打开 `SPE_BF_MixedModel.Rmd`
2. 点击 **Knit** 或逐 chunk 运行
3. 核心输出在：
   - **§4** — Label 主效应 BF（核心结果）
   - **§5** — 序贯曲线（收数据决策依据）
   - **§9** — 最终汇总表

---

## 七、产出文件

| 文件 | 路径 |
|:---|:---|
| 升级版 Rmd | `1_Code/R_for_Check/SPE_BF_MixedModel.Rmd` |
| 前版 Rmd (配对 t) | `1_Code/R_for_Check/SPE_BF_Analysis.Rmd` |
| 前版报告 | `1_Code/R_for_Check/SPE_BF_Analysis_Report.md` |
| 本报告 | `1_Code/R_for_Check/SPE_BF_MixedModel_Report.md` |
| 序贯函数参考 | `1_Code/R_for_Check/Function_bf_avo.R` |

---

*报告自动生成 | 版本 v0.2 | 更新于 2026-06-08*
