# SPE 贝叶斯因子分析报告 — 原始行为数据

> **日期**: 2026-06-05 | **脚本**: [SPE_BF_Analysis.Rmd](SPE_BF_Analysis.Rmd)
> **项目**: 基于高斯过程的实验设计优化
> **数据**: `2_Data/Real_Data/UnExtact/raw/EXP_data_group*.csv` (88 名被试，8 组)

---

## 一、研究问题

### 核心研究假设

> **H₁（备择假设）**: 在 Self-Matching Task 中，**匹配条件下自我标签(self)比陌生人标签(stranger)的试次表现更好**（即存在自我优势效应 SPE）。
>
> **H₀（零假设）**: 自我和陌生人试次之间没有差异。

### 序贯停止规则

根据实验方案，数据收集的停止标准为：

1. **BF₁₀ ≥ 10** 或 **BF₁₀ ≤ 0.1**（即有 strong evidence 支持 H₁ 或 H₀）
2. 若达到每组 **70 人** 的最大样本量仍未达到第 1 条标准，则停止收集

---

## 二、实验设计回顾

### 2.1 Self-Matching Task 范式 (Sui et al., 2012)

- **联结阶段**: 被试学习形状与标签的配对关系（60 秒记忆）
- **匹配阶段**: 屏幕上方显示形状（圆形/正方形），下方显示标签（"自我"/"生人"），被试按键判断是否匹配
- **实验设计空间**: Ω = (P, T, W)，其中 P=练习次数, T=刺激呈现(ms), W=反应窗口(ms)

### 2.2 按键映射规则

MATLAB 脚本 `experiment_formal_newcon.m` 中的 `getPairingRules()` 函数使用 `mod(subjectID, 4)` 进行**完全交叉平衡**：

| mod(subjectID, 4) | square-self | square-stranger | circle-self | circle-stranger |
| :---------------: | :---------: | :-------------: | :---------: | :-------------: |
|    0 (4的倍数)    | **f** |        j        |      j      |   **f**   |
|     1 (4k+1)     |      j      |   **f**   | **f** |        j        |
|   2 (其他偶数)   |      j      |   **f**   | **f** |        j        |
|     3 (4k+3)     | **f** |        j        |      j      |   **f**   |

### 2.3 匹配/不匹配规则

`mod(subjectID, 2)` 决定形状-标签的"正确"配对：

| mod(subjectID, 2) | 匹配规则 (correctPairs) | Matching-self           | Matching-stranger           |
| :---------------: | :---------------------- | :---------------------- | :-------------------------- |
|     0 (偶数)     | correctPairs_0          | **square** + self | **circle** + stranger |
|     1 (奇数)     | correctPairs_1          | **circle** + self | **square** + stranger |

> **本分析仅使用 Matching 试次**——即在正确配对关系下的自我和生人标签对比，这正是 SPE 的标准测量方法。

### 2.4 8 组实验条件

| 组别 | P (练习) | T (刺激, s) | W (反应窗口, s) | 被试数 |
| :--: | :------: | :----------: | :-------------: | :----: |
|  G1  |    0    | 0.03 (30ms) |   0.3 (300ms)   |   11   |
|  G2  |    0    | 0.03 (30ms) |   0.6 (600ms)   |   12   |
|  G3  |   120   | 0.03 (30ms) |   0.6 (600ms)   |   10   |
|  G4  |   120   | 0.08 (80ms) |   0.6 (600ms)   |   11   |
|  G5  |    8    | 0.10 (100ms) |  1.1 (1100ms)  |   11   |
|  G6  |   120   | 0.50 (500ms) |  1.5 (1500ms)  |   10   |
|  G7  |   120   | 0.08 (80ms) |   0.8 (800ms)   |   12   |
|  G8  |   120   | 0.08 (80ms) |   0.8 (800ms)   |   11   |

> 注：G7 和 G8 设计相同但被试独立（RoadMap 已确认）。

---

## 三、分析方法

### 3.1 贝叶斯因子计算

使用 **R 语言 `BayesFactor` 包** (Rouder et al., 2017) 进行贝叶斯因子计算：

- **检验类型**: 配对样本 t 检验 (paired = TRUE)
- **先验**: JZS 先验，默认 `rscale = "medium"` (r = √2/2 ≈ 0.707)
- **稳健性检查**: 同时报告 `rscale = "wide"` (r=1.0) 和 `rscale = "ultrawide"` (r=√2)

### 3.2 BF 解读标准 (Jeffreys, 1961)

| BF₁₀           | 证据强度         | 方向                  |
| :--------------- | :--------------- | :-------------------- |
| > 100            | Decisive         |                       |
| 30–100          | Very strong      |                       |
| **10–30** | **Strong** | **← 停止阈值** |
| 3–10            | Substantial      |                       |
| 1–3             | Anecdotal        |                       |
| 1/3–1           | Anecdotal        |                       |
| 1/10–1/3        | Substantial      |                       |
| **< 1/10** | **Strong** | **← 停止阈值** |
| < 1/100          | Very strong      |                       |
| < 1/300          | Decisive         |                       |

### 3.3 分析流程

```
原始数据 (88人, ~88000试次)
    │
    ├─ 过滤: stage == "formal"
    ├─ 识别: matching 试次 (基于 subjectID 奇偶性)
    ├─ 聚合: 每个被试 → (self_acc, stranger_acc, self_rt, stranger_rt)
    │
    ├─ 主分析: 全体被试 BF (ttestBF, paired=TRUE)
    ├─ 序贯分析: 模拟逐步增加被试时 BF 的累积变化
    ├─ 分组分析: 每组内部的 BF
    └─ 稳健性检查: 不同 rscale 下的 BF 变化
```

---

## 四、使用方法

### 4.1 环境要求

- **R** (≥ 4.0)
- **RStudio** (推荐)
- R 包：`tidyverse`, `BayesFactor`, `bruceR`, `ggplot2`, `cowplot`, `scales`

### 4.2 运行步骤

1. 在 RStudio 中打开 `SPE_BF_Analysis.Rmd`
2. 点击 **Knit** 按钮，或逐 chunk 运行
3. 结果输出为 HTML 文档

### 4.3 关键代码说明

```r
# 核心 BF 计算（配对 t 检验）
bf_acc <- ttestBF(
  x      = df_wide$self_acc,         # 每个被试的 self 正确率
  y      = df_wide$stranger_acc,     # 每个被试的 stranger 正确率
  paired = TRUE,                      # 配对设计
  rscale = "medium"                   # JZS 先验 scale
)

# 序贯分析：逐步增加被试
# 从第10个被试开始，每次增加1个，重新计算 BF
# 找到 BF 首次达到 10 或 1/10 的被试数
```

---

## 五、预期输出

运行 `.Rmd` 后，将获得以下输出：

### 5.1 主 BF 结果

- `BF10_acc`: 正确率 SPE 的贝叶斯因子
- `BF10_rt`: 反应时 SPE 的贝叶斯因子
- 判断是否达到 strong evidence 阈值

### 5.2 先验稳健性表

不同 `rscale`（medium/wide/ultrawide）下的 BF 对比

### 5.3 序贯分析

- 累计被试数 vs BF 曲线
- 标注 BF=10 和 BF=1/10 阈值线
- 首次达到阈值所需的被试数

### 5.4 分组 BF

- 8 组实验条件下各自的 BF
- 对比不同参数组合对 SPE 证据强度的影响

### 5.5 可视化

- SPE 分布的箱线图（按组）
- 序贯 BF 变化趋势图
- 各组 BF 柱状图

---

## 六、结果解读指南

### 情景 A：BF₁₀ ≥ 10

> **有 STRONG 证据支持自我优势效应的存在**
>
> 说明在当前样本量下，数据强烈支持 self > stranger 的表现优势。
> 可以停止收数据，已满足序贯停止标准。

### 情景 B：BF₁₀ ≤ 0.1

> **有 STRONG 证据支持零假设（无 SPE）**
>
> 说明数据更可能来自无 SPE 的零模型。
> 这也满足停止标准——可以下结论：该任务设计中未检测到 SPE。

### 情景 C：3 ≤ BF₁₀ < 10

> **有 SUBSTANTIAL 但尚未达 STRONG 的证据**
>
> 接近但未达到停止阈值。建议：
>
> - 若接近最大样本量（如已收 50+ 人），可继续收到 70 人
> - 报告 BF 并注明为 substantial evidence

### 情景 D：1 < BF₁₀ < 3 或 1/3 < BF₁₀ < 1

> **ANECDOTAL 证据，无法定论**
>
> 数据不足以支持任何一方的结论。继续收数据直至达标或达到最大样本量。

---

## 七、与 RoadMap 的衔接

本分析使用**原始行为数据**（正确率、反应时），而 RoadMap 中的 GP+Sigmoid 模型使用 HDDM 提取的 **DDM 参数**（漂移率 v、边界 a 等）。

两个层面的分析互为补充：

| 层面 | 本分析 (R script)      | RoadMap HDDM 分析             |
| :--- | :--------------------- | :---------------------------- |
| 数据 | 原始行为数据 (RT, ACC) | DDM 参数 (v_self, v_stranger) |
| 方法 | 配对 t 检验 BF         | JZS ANOVA BF / 回归 BF        |
| 假设 | task-level SPE 存在    | 参数层面的 SPE 受 P/T/W 调节  |
| 优势 | 直接、易解释           | 分离认知过程成分              |

---

## 八、参考文献

1. Sui, J., He, X., & Humphreys, G. W. (2012). Perceptual effects of self-prioritization. *JEP: HPP*, 38(5), 1105–1114.
2. Rouder, J. N., Morey, R. D., Speckman, P. L., & Province, J. M. (2012). Default Bayes factors for ANOVA designs. *JMP*, 56(5), 356–374.
3. Rouder, J. N., Morey, R. D., Verhagen, J., Swagman, A. R., & Wagenmakers, E.-J. (2017). Bayesian analysis of factorial designs. *Psychological Methods*, 22(2), 304–321.
4. Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
5. Morey, R. D., & Rouder, J. N. (2024). *BayesFactor*: Computation of Bayes factors for common designs. R package version 0.9.12-4.7.

---

*报告自动生成 | 版本 v0.1 | 更新于 2026-06-05*
