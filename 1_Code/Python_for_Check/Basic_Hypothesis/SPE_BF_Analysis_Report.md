# 底层假设检验报告：实验设计空间参数对 SPE 的影响

> **日期**: 2026-06-02 | **分析者**: AI Assistant
> **项目**: 基于高斯过程的实验设计优化 (Guassion-Process-Experiment-Design)
> **代码**: [SPE_BF_Analysis.ipynb](SPE_BF_Analysis.ipynb)

---

## 一、研究问题

> 是否存在**稳定的证据**支持底层假设：**"实验设计空间参数 (P, T, W) 对自我优势效应 (SPE) 有影响"**？

其中：

- P = 练习次数 (practice trials)
- T = 刺激呈现时间 (stimulus presentation, ms)
- W = 反应窗口 (response window, ms)
- SPE = v_self − v_stranger（DDM 漂移率的自我 vs 陌生人差异）

---

## 二、分析方法

### 2.1 两道防线

| 防线                                        | 方法                              | 回答的问题                                                 |
| :------------------------------------------ | :-------------------------------- | :--------------------------------------------------------- |
| **防线 1：G\*Power 敏感性分析**       | 非中心 F 分布计算最低可检测效应量 | "当前设计在 80% power 下能检测到多大的效应？"              |
| **防线 2：贝叶斯因子 (Bayes Factor)** | JZS 先验 + 数值积分               | "数据支持 H₁（参数影响 SPE）还是 H₀（参数不影响 SPE）？" |

### 2.2 数据

- **来源**: 8 组 HDDM 层次模型拟合得到的**被试级** DDM 参数（从 `HDDM_Traces/*_stats.csv` 提取）
- **被试级 SPE**: 对每位被试计算 SPE = v_subj(1).mean − v_subj(0).mean
- **核心数据集 (Core)**: 排除 G1/G2（高遗漏率，ROADMAP 建议排除），保留 G3-G8，N=65
- **比对数据集**: All (G1-G8, N=88), Lenient (G2-G8, N=77), Clean (Core + 排除极端值, N=65)

### 2.3 假设设定

#### 单因素方差分析 (ANOVA)

- **H₀**: 所有实验条件的 SPE 均值相等（μ_G3 = μ_G4 = μ_G5 = μ_G6 = μ_G7 = μ_G8）
- **H₁**: 至少有两组 SPE 均值不相等

#### 多元回归

- **H₀**: SPE = β₀ + ε（P, T, W 不预测 SPE）
- **H₁**: SPE = β₀ + β₁×P + β₂×T + β₃×W + ε（P, T, W 联合预测 SPE）

---

## 三、G\*Power 敏感性分析

### 3.1 计算原理

G\*Power 使用非中心 F 分布计算在给定 α、power、组数 k、总样本量 N 下可检测的最小 Cohen's f：

- **临界 F 值**: F_crit = F_{1−α}(k−1, N−k)
- **非中心参数**: λ = N × f²
- **检验力**: Power = P(F(df₁, df₂, λ) > F_crit)

### 3.2 G\*Power GUI 操作步骤

1. 打开 `D:\Gpower\GPowerNT.exe`
2. 选择 **F tests** → **ANOVA: Fixed effects, omnibus, one-way**
3. 选择 **Sensitivity** 分析类型
4. 输入参数：

| 参数                   | 取值 |
| :--------------------- | ---: |
| α err prob            | 0.05 |
| Power (1−β err prob) | 0.80 |
| Number of groups       |    6 |
| Total sample size      |   65 |

5. 点击 **Calculate** → 获得结果：**f ≈ 0.465**

### 3.3 敏感性分析结果

| 数据集         | k |  N | f_min (80% power) | η²_min | Cohen 分类      |
| :------------- | -: | -: | ----------------: | -------: | :-------------- |
| Core: G3-G8    | 6 | 65 |   **0.465** |    0.178 | **Large** |
| Lenient: G2-G8 | 7 | 77 |             0.440 |    0.162 | Large           |
| All: G1-G8     | 8 | 88 |             0.422 |    0.151 | Large           |

> 📊 **解读**: 当前设计（6 组 × 65 被试）仅能检测到 **大效应量** (f ≥ 0.465，对应 η² ≥ 17.8%)。若希望检测中等效应量 (f=0.25)，需要约 **150−180 名被试**。

### 3.4 观察效应量 vs 最低可检测效应量

| 数据集                 |                      F |               p |            η² |           f_obs |           f_min | f_obs ≥ f_min? |
| :--------------------- | ---------------------: | --------------: | --------------: | --------------: | --------------: | :-------------- |
| All (G1-G8)            |           F(7,80)=2.60 |           0.018 |           0.186 |           0.477 |           0.422 | ✅ YES          |
| **Core (G3-G8)** | **F(5,59)=2.95** | **0.019** | **0.200** | **0.500** | **0.465** | ✅**YES** |
| Clean                  |           F(5,59)=2.95 |           0.019 |           0.200 |           0.500 |           0.465 | ✅ YES          |

> 📊 **关键发现**: 观察效应量 f=0.50 **刚好超过** 80% power 要求的最低可检测量 f=0.47。这意味着当前设计在 80% power 水平上**勉强足够**，但安全边际很小。

---

## 四、贝叶斯因子 (Bayes Factor) 分析

### 4.1 方法

使用 **JZS 先验** (Jeffreys-Zellner-Siow prior, Rouder et al., 2009, 2012) 计算 BF₁₀。

**核心公式**（将 ANOVA 视为具有 p=a−1 个虚拟变量的回归）:

$$
\text{BF}_{10} = \int_0^\infty (1+g)^{(n-p-1)/2} \cdot [1+g(1-R^2)]^{-(n-1)/2} \cdot \pi(g) \, dg
$$

其中 JZS 先验密度:

$$
\pi(g) = \frac{r}{\sqrt{2\pi}} \cdot g^{-3/2} \cdot \exp\left(-\frac{r^2}{2g}\right), \quad r = 0.5 \text{ (default medium)}
$$

数值积分使用 `scipy.integrate.quad`。

### 4.2 解读标准 (Jeffreys, 1961)

| BF₁₀         | 证据强度                    |
| :------------- | :-------------------------- |
| > 100          | Decisive for H₁            |
| 30−100        | Very strong for H₁         |
| 10−30         | Strong for H₁              |
| 3−10          | Substantial for H₁         |
| **1−3** | **Anecdotal for H₁** |
| 1/3−1         | Anecdotal for H₀           |
| 1/10−1/3      | Substantial for H₀         |
| < 1/10         | Strong or stronger for H₀  |

### 4.3 ANOVA BF 结果

| 数据集                        |       r_scale |           k |            N |             R² |         BF₁₀ | 证据                          |
| :---------------------------- | ------------: | ----------: | -----------: | --------------: | -------------: | :---------------------------- |
| All G1-G8 (medium)            |           0.5 |           8 |           88 |           0.186 | **3.02** | **Substantial for H₁** |
| All G1-G8 (wide)              |           1.0 |           8 |           88 |           0.186 |           3.39 | Substantial for H₁           |
| **Core G3-G8 (medium)** | **0.5** | **6** | **65** | **0.200** | **2.83** | **Anecdotal for H₁**   |
| Core G3-G8 (wide)             |           1.0 |           6 |           65 |           0.200 |           3.29 | Substantial for H₁           |
| Lenient G2-G8 (medium)        |           0.5 |           7 |           77 |           0.177 |           2.27 | Anecdotal for H₁             |
| Clean (medium)                |           0.5 |           6 |           65 |           0.200 |           2.83 | Anecdotal for H₁             |

### 4.4 回归 BF 结果 (SPE ~ P + T + W)

| 数据集                        |       r_scale |  n | p |             R² |         BF₁₀ | 证据                        |
| :---------------------------- | ------------: | -: | -: | --------------: | -------------: | :-------------------------- |
| **Core G3-G8 (medium)** | **0.5** | 65 | 3 | **0.055** | **0.76** | **Anecdotal for H₀** |
| Core G3-G8 (wide)             |           1.0 | 65 | 3 |           0.055 |           0.55 | Anecdotal for H₀           |
| All G1-G8 (medium)            |           0.5 | 88 | 3 |           0.054 |           0.92 | Anecdotal for H₀           |

### 4.5 配对 BF（各组间成对比较，Clean 数据集，r=0.707）

所有 15 对比较的 BF₁₀ 均在 **0.38−0.93** 范围，全部落在"Anecdotal for H₀"区间。**没有任何一组对比显示出实质性差异的证据**。

---

## 五、综合结论

### 5.1 结果汇总

| 分析              | 指标                     | 值                      | 结论                   |
| :---------------- | :----------------------- | :---------------------- | :--------------------- |
| G\*Power          | 最低可检测 f (80% power) | **0.465 (Large)** | —                     |
| G\*Power          | 观察到的 f               | **0.500**         | ✅ 刚超过阈值          |
| ANOVA F-test      | F(5,59), p               | **2.948, 0.019**  | ⚠️ 显著但边际        |
| ANOVA η²        | —                       | **0.200**         | 中等效应               |
| ANOVA BF₁₀      | r=0.5 (medium)           | **2.83**          | ⚠️ Anecdotal for H₁ |
| ANOVA BF₁₀      | r=1.0 (wide)             | **3.29**          | ✅ Substantial for H₁ |
| Regression R²    | P+T+W                    | **0.055**         | ❌ 线性预测力极弱      |
| Regression BF₁₀ | r=0.5 (medium)           | **0.76**          | ⚠️ Anecdotal for H₀ |

### 5.2 最终判断

#### 频数主义视角 (F-test + G\*Power)

- p=0.019 < 0.05，实验条件间 SPE 有统计学显著差异
- 观察效应量 f=0.50 刚好超过 80% power 的最低要求 (f=0.47)
- **底层假设得到初步支持，但统计效力处于安全边际上**

#### 贝叶斯视角 (Bayes Factor)

- ANOVA BF₁₀=2.83 (default prior r=0.5)：Anecdotal 支持 H₁，**不足以下定论**
- ANOVA BF₁₀=3.29 (wide prior r=1.0)：Substantial 支持 H₁，**勉强可接受**
- 回归 BF₁₀=0.76：轻微倾向 H₀，P/T/W 的线性组合几乎无法预测 SPE
- **贝叶斯证据不够稳定，取决于先验选择**

### 5.3 为什么会这样？

1. **模型复杂度惩罚**: JZS 先验自动惩罚模型复杂度。6 组模型比 1 组多 5 个参数，BF 自动扣分
2. **R²=0.20 不够大**: 虽然 F-test 显著，但贝叶斯视角需要更大的 R² 来克服模型的参数复杂度惩罚
3. **被试间变异大**: SPE 的被试内/组内标准差 (~1.28) 远大于组间差异 (~0.50)
4. **线性模型不适合**: P/T/W 对 SPE 的关系很可能不是线性的（GP 建模正是为解决此问题）

### 5.4 建议

| 优先级 | 建议                               | 理由                                                       |
| :----- | :--------------------------------- | :--------------------------------------------------------- |
| 🔴 高  | 增加实验条件至 10−12 个           | 当前 6−8 个条件不足以稳定估计参数-SPE 关系                |
| 🔴 高  | 使用 GP 非线性建模替代线性回归     | 回归 R² 仅 5.5%，线性假设不成立                           |
| 🟡 中  | 每条件增加至 15−20 名被试         | 降低组内标准差对 BF 的影响                                 |
| 🟡 中  | 报告 BF 结果并注明先验敏感性       | 提高分析透明度                                             |
| 🟢 低  | 补充 pre-registration 的样本量规划 | 使用 G\*Power 结果（f_min=0.47）作为下一轮实验的效应量基准 |

---

## 六、代码与产出

| 产出                       | 路径                                                                   |
| :------------------------- | :--------------------------------------------------------------------- |
| Notebook (可逐行运行)      | `1_Code/Python_for_Check/Basic_Hypothesis/SPE_BF_Analysis.ipynb`     |
| 本报告                     | `1_Code/Python_for_Check/Basic_Hypothesis/SPE_BF_Analysis_Report.md` |
| 可视化 (SPE overview)      | `3_Figures/Basic_Hypothesis/SPE_overview.png`                        |
| 可视化 (Power analysis)    | `3_Figures/Basic_Hypothesis/Gpower_power_analysis.png`               |
| 可视化 (BF sensitivity)    | `3_Figures/Basic_Hypothesis/BF_prior_sensitivity.png`                |
| 可视化 (Summary dashboard) | `3_Figures/Basic_Hypothesis/BF_summary_dashboard.png`                |

---

## 七、参考文献

1. Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
2. Rouder, J. N., Morey, R. D., Speckman, P. L., & Province, J. M. (2012). Default Bayes factors for ANOVA designs. *Journal of Mathematical Psychology*, 56(5), 356-374.
3. Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review*, 16(2), 225-237.
4. Liang, F., Paulo, R., Molina, G., Clyde, M. A., & Berger, J. O. (2008). Mixtures of g priors for Bayesian variable selection. *Journal of the American Statistical Association*, 103(481), 410-423.

---

*报告自动生成 | 版本 v0.3 | 更新于 2026-06-02*
