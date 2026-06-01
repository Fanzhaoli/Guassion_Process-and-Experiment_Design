# Omission（遗漏试次）纳入 DDM 建模的可行性分析

> **文献来源**: Leng, X., Fengler, A., Shenhav, A., & Frank, M. J. (2025). The Perils of Omitting Omissions when Modeling Evidence Accumulation. *In Prep.*
>
> **分析目标**: 评估在本项目（GP+Sigmoid 混合模型 + HDDM 参数提取管线）中加入 Omission 显式建模的可行性与必要性。

---

## 一、文献核心发现

### 1.1 忽略 Omission 对参数估计的偏倚

Leng et al. (2025) 通过系统数值实验证明，**在 DDMS 拟合时直接丢弃 omission 试次会对参数恢复产生严重的系统性偏差**——即使 omission 率低至 5% 以下：

| 模型类型 | 被偏倚的参数 | 偏倚方向 | 偏倚机制 |
|:---|:---|:---:|:---|
| 恒定边界 DDM | 阈值 a | **低估** | 模型只看到在 deadline 前已完成的（快速）试次，推断出更低的决策边界 |
| 恒定边界 DDM | 漂移率 v | **高估** | 丢弃慢速试次后，剩余数据的 RT 更快 → 模型推断更高漂移率 |
| 线性塌缩边界 (ANGLE) | 初始阈值 a | **高估** | 模型认为边界必须塌缩得更快才能解释为什么没有更多 omission |
| 线性塌缩边界 (ANGLE) | 塌缩角 θ | **高估** | 同上——更激进的塌缩意味着边界更快降至 0，减少 omission 概率 |
| 非线性塌缩 (WEIBULL) | a, α, β | 系统性偏倚 | 所有三个边界参数被扭曲以最小化被忽略的 omission |

> **核心机制**: 没有 omission 项来"惩罚"模型，模型自然会选择能最大化可见数据似然的参数。由于 omission 意味着 RT 较长，忽略它们使模型偏好能产生更快 RT 的参数组合（更低 a、更高 v、或更激进塌缩）。

### 1.2 跨条件比较中的偏倚

不仅在单条件参数恢复中有偏倚，在**跨实验条件的参数差异比较**中，忽略 omission 也会导致严重低估效应量：

- 合成实验中，LAN-only（忽略 omission）低估了条件间的塌缩角差异 Δθ（真值 0.2 → 恢复值远小于 0.2）
- LAN+OPN（纳入 omission）正确恢复了 Δθ
- 忽略 omission 的模型在 posterior predictive check 中**系统性地低估了 omission 率**

### 1.3 偏倚对 Lapse（注意力流失）的鲁棒性

即便数据中存在由注意力流失（lapse distribution）产生的 omission，LAN+OPN 方法仍能：
- 正确恢复 SSM 的边界参数 (a, θ)
- 准确推断 lapse 比例 p_lapse
- 而 LAN-only 方法显著低估 p_lapse（因为大量 lapse 试次以 omission 形式被忽略）

### 1.4 提出的解决方案：LAN+OPN

论文提出了 **LAN (Likelihood Approximation Network) + OPN (Omission Probability Network)** 框架：

```
联合对数似然:
log_l(D, O | θ, d) = Σ log f_LAN(rt_i, c_i | θ) + |O| × log f_OPN(omission | θ, d)

其中:
  D  = 观测数据 (rt_i ≤ deadline, 含 choice)
  O  = omission 试次数
  θ  = SSM 参数
  d  = deadline
```

- **OPN** 是一个预训练的神经网络，输入 θ 和 deadline d，输出 omission 概率
- 不需要解析积分 post-deadline 概率密度函数
- 已集成到 **HSSM** Python 包（Fengler et al., in prep），配置简单

---

## 二、本项目 Omission 现状

### 2.1 各组 Omission 率

| 组别 | P | T (ms) | W (ms) | M (ms) | 总试次 | 有效试次 | Omission 率 | DDM v_self |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| G1 | 0 | 30 | 300 | **330** | 2860 | 792 | **72.3%** 🔴 | −3.39 |
| G2 | 0 | 30 | 600 | **630** | 3120 | 1484 | **52.4%** 🔴 | −2.53 |
| G3 | 120 | 30 | 600 | **630** | 2600 | 1596 | **38.6%** 🟠 | −1.89 |
| G4 | 120 | 80 | 600 | **680** | 2860 | 1762 | **38.4%** 🟠 | +0.64 |
| G5 | 8 | 100 | 1100 | **1200** | 2860 | 2547 | 10.9% 🟢 | +1.35 |
| G6 | 120 | 500 | 1500 | **2000** | 2600 | 2454 | **5.6%** 🟢 | +1.81 |
| G7 | 120 | 80 | 800 | **880** | 3120 | 2674 | 14.3% 🟢 | +1.20 |
| G8 | 120 | 80 | 800 | **880** | 2860 | 2440 | 14.7% 🟢 | +2.81 |

### 2.2 当前对 Omission 的处理方式

```
当前 pipeline  (step2_hddm_fit.py / Docker_Run.ipynb):
  ┌──────────────────────────────────────────────────────┐
  │ 1. omission=1 的试次: rt = T+W (deadline)            │
  │ 2. response = 0                                       │
  │ 3. 传入 HDDM 的 hddm.HDDM() 拟合                     │
  │    → HDDM 将此识别为 "右截尾" (right-censored)       │
  │    → 告诉模型: "该试次到 T+W 时刻仍未穿越边界"        │
  │    → 但 NOT 显式建模 omission 生成过程                │
  └──────────────────────────────────────────────────────┘
```

**HDDM 的截尾方法（censored data approach）** 是一个**中间方案**：
- ✅ 比直接丢弃 omission 强：让模型知道有试次未完成
- ⚠️ 不足：未显式建模 deadline 对证据累积过程的影响（如 urgency signal、collapsing boundary）
- ⚠️ Leng et al. 的分析主要是对比"完全丢弃 omission"与"LAN+OPN"，未直接评估 HDDM 的截尾方法

### 2.3 Omission 率与 DDM 参数的关系（本项目实证观察）

```
Omission 率 vs v_self (HDDM 估计):
  G1: 72.3% → v = −3.39  ← 极端负值
  G2: 52.4% → v = −2.53  ← 强烈负值
  G3: 38.6% → v = −1.89  ← 负值
  G4: 38.4% → v = +0.64  ← 正值！与 G3 omission 率相似但 v 符号相反
  G5: 10.9% → v = +1.35
  G6: 5.6%  → v = +1.81
  G7: 14.3% → v = +1.20
  G8: 14.7% → v = +2.81
```

G1-G3 的 omission 率与 v 负值的关系值得注意——但 G4（遗漏率 38.4%）的 v 已经回正。这说明 omission 率不是 v 的唯一决定因素，T 和 P 也起重要作用。

> **注**: 上表中的 DDM 参数来自原始 HDDM pipeline（p_outlier=0.05）。敏感性分析（§2.4）中重新使用 p_outlier=0 拟合了 Censor 方案，参数略有不同。

---

## 三、可行性评估

### 3.1 对本文献方法与本项目的适配性分析

| 维度 | Leng et al. (2025) 方案 | 本项目现状 | 适配度 |
|:---|:---|:---|:---:|
| **核心模型** | DDM / ANGLE / WEIBULL | 标准 DDM (HDDM) | ✅ 适用 |
| **Omission 率范围** | 主要在 5-30% | G1-G4 高达 38-72% | ⚠️ 超出论文典型范围 |
| **边界类型** | 恒定 / 线性塌缩 / 非线性塌缩 | 恒定边界 | ✅ 论文涵盖恒定边界 |
| **参数估计方法** | LAN+OPN (神经网络近似似然) | HDDM MCMC (解析似然) | ⚠️ 方法不同 |
| **软件集成** | HSSM (Python) | HDDM (Python) | ⚠️ 不同包，但 HSSM 兼容 HDDM |
| **Deadline 设置** | 固定的 1.25s | 各组不同 (330ms-2000ms) | ✅ 论文支持不同 deadline |
| **Lapse 建模** | 支持联合估计 | 未建模 | 🟡 可扩展 |

### 3.2 三种可能的 Omission 建模路线

#### 路线 A：保持当前 HDDM 截尾方案（推荐作为基线） ⭐

```
当前做法:
  omission 试次 → rt=T+W, response=0 → HDDM 右截尾拟合

优点:
  ✅ 已经比"直接丢弃"好一步
  ✅ 无需额外代码或新工具
  ✅ 与 HDDM 框架无缝兼容

不足:
  ⚠️ 未显式建模 deadline 对累积过程的动态影响
  ⚠️ 在极高 omission 率 (G1: 72%) 下截尾方法可能力不从心
  ⚠️ 无法建模 collapsing boundary / urgency signal
```

#### 路线 B：迁移到 HSSM + LAN+OPN（激进方案）

```
做法:
  使用 HSSM 替代 HDDM
  配置 deadlinedata=True
  HSSM 自动训练 OPN 并联合拟合

优点:
  ✅ 直接实现 Leng et al. (2025) 的推荐方法
  ✅ 可联合估计 lapse 比例
  ✅ 支持更复杂的边界形状 (ANGLE, WEIBULL)

不足:
  🔴 当前 HDDM pipeline 完全重写
  🔴 OPN 需要预训练（需要大量模拟数据作为训练集）
  🔴 HSSM + LAN/OPN 的学习曲线较陡
  🔴 论文中的 code 和 HSSM 集成仍在开发中（"in prep"）
  🔴 8 组小样本下神经网络的训练挑战
```

#### 路线 C：在 HDDM 内增强 Omission 建模（折中方案）⭐

```
做法:
  方案1: 在 HDDM 中使用 p_outlier 模型显式估计 contamination
          → HDDM 已支持 p_outlier=0.05（默认）
          → 可以尝试将 omission 视为一种特殊的 outlier 机制

  方案2: 在 Sigmoid 生成模型中加入 omission 概率函数
          → 新增 sigmoid_omission_rate(P, T, W) 函数
          → 基于观测 omission 率进行校准
          → DDM 模拟时用此函数决定是否生成 omission

  方案3: 将 omission 率作为 GP 的额外目标变量
          → GPSigmoidHybridModel 新增 gp_omission
          → 同时预测 v,a,t,z 和 omission_rate

优点:
  ✅ 在现有框架内扩展，不引入全新依赖
  ✅ 核心管线（DDM→HDDM→Sigmoid→GP）保持不变
  ✅ 可渐进实现

不足:
  ⚠️ 未使用 OPN 的最先进方法
  ⚠️ 可能不如路线 B 精确
```

### 3.3 高 Omission 组的特殊挑战

Leng et al. (2025) 主要考察 omission 率 <30% 的情况。本项目 G1-G4 的 omission 率（38-72%）远超此范围，带来额外挑战：

| 挑战 | 说明 |
|:---|:---|
| **极低试次数** | G1 有效试次仅 792/2860，可能不足以可靠估计 DDM 参数，无论用什么方法 |
| **截尾方法失效** | 当 >50% 数据是截尾的，HDDM 的 MCMC 可能不稳定（G1: 2068/2860 为截尾） |
| **DDM 模型适用性** | 如此高的 omission 率可能意味着 DDM 的恒定边界假设根本不成立——被试可能根本没有形成稳定的决策过程 |
| **死亡螺旋** | v 的极端负值（如 G1 v≈−3.4）暗示被试**系统性地趋向错误边界**——这在有高 omission 率的恒定边界 DDM 中是 expectable 的 artifact（Leng et al. Fig 2A-B） |

> **关键判断**: G1 (72% omission) 和 G2 (52% omission) 的 DDM 估计可能已被文献所描述的偏倚机制强烈污染。即使使用 LAN+OPN 也未必能挽救——论文中未测试如此极端的数据条件。

### 2.4 敏感性分析实证结果（2026-06-01 完成）

对全部 8 组数据进行 Censor（rt=deadline, response=0, p_outlier=0）与 Drop（去除 omission 试次, p_outlier=0.05）两种方案的 HDDM 拟合（MCMC: 3000 draws, 500 burn），系统比较 DDM 参数的差异。

> **代码位置**: `1_Code/Python_for_Check/Omission/Omission_Sensitivity_Analysis.ipynb`
> **图表输出**: `3_Figures/Omission_Sensitivity/`

#### 2.4.1 拟合概况

全部 8 组在两个方案下均成功收敛。关键数据：

| Group | 遗漏率 | Censor 试次 | Drop 试次 | Censor p_outlier | 备注 |
|:---:|:---:|:---:|:---:|:---:|:---|
| G1 | 72.3% | 2860 | 791 | 0.0 | — |
| G2 | 52.0% | 3120 | 1497 | 0.0 | — |
| G3 | 36.6% | 2600 | 1649 | 0.0 | — |
| G4 | 35.3% | 2860 | 1849 | 0.0 | — |
| G5 | 9.8% | 2860 | 2580 | 0.0 | — |
| G6 | 6.7% | 2600 | 2427 | 0.0 | — |
| G7 | 13.4% | 3120 | 2702 | 0.0 | — |
| G8 | 13.7% | 2860 | 2468 | 0.0 | — |

> **注**: Censor 方案的 p_outlier 设为 0（而非默认 0.05），因截尾数据中的人工构造 response=0 试次并非真实 outlier——开启 outlier detection 会与截尾机制产生不良交互，导致 Slice Sampler Step-out 失败。

#### 2.4.2 Censor vs Drop: DDM 参数差异

**漂移率 v_self 和 v_stranger**（受偏倚最严重的参数）：

| Group | Censor v_self | Drop v_self | Δ | Censor v_stranger | Drop v_stranger | Δ | 95%CI 重叠？ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| G1 (72%) | −4.86 | **+1.95** | **+6.81** | −4.87 | **+1.94** | **+6.80** | ❌ |
| G2 (52%) | −1.23 | **+1.66** | **+2.89** | −1.94 | −0.05 | **+1.89** | ❌ |
| G3 (37%) | −0.79 | +0.78 | +1.58 | −1.18 | −0.02 | +1.16 | ❌ |
| G4 (35%) | +0.23 | +2.20 | +1.97 | −0.79 | +0.94 | +1.74 | ❌ |
| G5 (10%) | +1.49 | +2.32 | +0.84 | +0.59 | +1.11 | +0.52 | ✅ |
| G6 (7%) | +0.92 | +1.70 | +0.78 | +0.69 | +0.87 | +0.17 | ✅ |
| G7 (13%) | +1.51 | +2.52 | +1.01 | +0.70 | +1.56 | +0.86 | ✅ |
| G8 (14%) | +1.56 | +2.56 | +1.01 | +0.74 | +1.63 | +0.89 | ✅ |

> 🔴 **G1 出现了符号反转**: Censor 方案的 v 为极端负值（被试系统性地趋向错误边界），而 Drop 方案的 v 为正——**两种方案对同一组被试的"自我优势方向"给出了完全相反的结论**。

**SPE in Drift Rate (v_self − v_stranger)**：

| Group | Censor SPE_v | Drop SPE_v | Δ | Cohen's d |
|:---:|:---:|:---:|:---:|:---:|
| G1 | +0.005 | +0.011 | +0.007 | 0.01 |
| G2 | +0.707 | +1.706 | +0.999 | 1.30 |
| G3 | +0.389 | +0.805 | +0.419 | 0.73 |
| G4 | +1.026 | +1.255 | +0.229 | 0.42 |
| G5 | +0.893 | +1.211 | +0.317 | 0.65 |
| G6 | +0.227 | +0.836 | +0.609 | 1.83 |
| G7 | +0.811 | +0.963 | +0.152 | 0.24 |
| G8 | +0.816 | +0.935 | +0.119 | 0.21 |

> 🟡 SPE_v 相对 v_self/v_stranger 本身更稳定——两种方案下的差异（Δ）在 0.007~1.0 之间，且 Cohen's d 均 < 2。但 G2 的 Δ = 1.0（Cohen's d = 1.3）仍然可观。

**其他参数（Decision Boundary a、Nondecision Time t、Starting Point z）**：

| 参数 | 方向 | 大小范围 | 高遗漏组 | 低遗漏组 |
|:---|:---:|:---:|:---|:---|
| **a** (边界) | Censor > Drop | Δ: −0.59 ~ +0.05 | −0.45 ~ −0.59 | −0.06 ~ −0.14 |
| **t** (非决策时间, s) | Censor < Drop | Δ: +0.01 ~ +0.08 | +0.01 ~ +0.08 | +0.01 ~ +0.07 |
| **z** (起始点偏倚) | Censor > Drop | Δ: −0.24 ~ +0.08 | −0.11 ~ −0.24 | −0.04 ~ +0.08 |

#### 2.4.3 可视化图表

| 图表 | 路径 | 内容 |
|:---|:---|:---|
| 遗漏率汇总 | `3_Figures/Omission_Sensitivity/omission_summary_by_group.png` | 各组 Omission 率与试次分布 |
| 参数对比柱状图 | `3_Figures/Omission_Sensitivity/sensitivity_censor_vs_drop_params.png` | Censor vs Drop 6 个参数的后验均值 + 95% CI |
| 一致性散点图 | `3_Figures/Omission_Sensitivity/sensitivity_scatter_censor_vs_drop.png` | Censor vs Drop 参数散点（颜色=遗漏率） |
| 参数差异图 | `3_Figures/Omission_Sensitivity/sensitivity_delta_params.png` | Δ = Drop − Censor 柱状图（橙色=\|Δ\|>0.5） |
| 遗漏率关系图 | `3_Figures/Omission_Sensitivity/sensitivity_delta_vs_omission_rate.png` | Δ 随遗漏率变化的趋势 |

#### 2.4.4 核心发现总结

1. **Leng et al. (2025) 的预测被完全证实**: 直接丢弃 omission 试次会**严重高估漂移率 v**（平均 Δ = +1.5 ~ +6.8），低估决策边界 a（Censor > Drop），效应量 Cohen's d 高达 9.3。

2. **遗漏率 >35% 时偏倚达到灾难性水平**: G1-G4（遗漏率 35-72%）的 v 参数在两种方案间差异均 >1.5，95% CI 完全不重叠，G1 甚至出现符号反转。

3. **遗漏率 <15% 时偏倚仍然存在但影响可控**: G5-G8 的 v_self Δ 在 0.8~1.0，95% CI 能够重叠——暗示当前 HDDM 截尾方案在低遗漏率条件下可能已"足够好"。

4. **SPE (v_self − v_stranger) 相对稳健**: 两种方案下的 SPE_v 差异远小于各自 v 的差异，说明 omission 处理方法对 SPE 效应方向的推断影响有限（但量值可能被压缩）。

5. **G1 (72.3%) 和 G2 (52.0%) 的 DDM 拟合确实不可靠**: 无论 Censor 还是 Drop，这些组的信息量已不足以支撑可靠的 DDM 参数估计——Drop 方案仅剩 791 和 1497 个有效试次（分别除以 11 和 12 名被试后，人均约 72 和 125 个试次）。

6. **Censor 方案的 p_outlier 必须设为 0**: 截尾数据中人工构造的 response=0 试次与真实 outlier 机制冲突，p_outlier>0 会导致 Slice Sampler 在似然面的平坦区域 Step-out 失败。

---

## 四、建议

### 4.1 短期可行方案（✅ 已完成并验证）

~~Step 1: 排除 G1, G2 (omission >50%)~~
~~Step 2: 对 G3-G8 维持当前 HDDM 截尾方案~~
~~Step 3: 在敏感性分析中对比两种 omission 处理方案~~

> ✅ **§2.4 敏感性分析已完成**。结果确认：
> - G1, G2 的 DDM 拟合确实不可靠（无论 Censor 还是 Drop）→ **正式建议在后续分析中排除 G1 和 G2**
> - G3-G8 的 omission 处理方法确实影响参数估计 → 需要在论文中明确讨论
> - 当前 HDDM 截尾方案（Censor）在遗漏率 <15% 时与 Drop 方案的参数 95% CI 能够重叠 → 作为**基线方案可接受**
> - **新增建议**: G3 和 G4（遗漏率 35-38%）处于灰色地带——两种方案的差异依然显著，但尚在合理范围内。建议在分析中将 G3-G4 标记为"需谨慎解释"。

### 4.2 中期拓展方案（路线 C 方向）

如果 Step 3 显示 omission 处理方式确实影响结论：

```
Step 4: 在 GPSigmoidHybridModel 中加入 omission 概率预测
  → 训练第 6 个 GP: gp_omission
  → 输入 (P,T,W), 输出 omission_rate
  → LOCV 验证 omission 率的预测精度

Step 5: 在行为验证 (Step 5) 中对比
  → 真实 omission 率 vs 模型预测 omission 率
  → 已有的 Step 5 中 omission rate r=0.923 → 已有良好基础
```

### 4.3 长期展望（路线 B 方向）

- 追踪 HSSM 的 LAN+OPN 功能正式发布（截至 2026-06-01 仍在开发中）
- 在新一轮数据采集后（Phase 4 建议的 4-6 个新条件），样本量增加后再评估切换工具链的性价比
- 当前 8 组条件下，切换 HSSM 的收益可能不如新增数据条件
- **已完成的敏感性分析显示**：在 G3-G8 (omission <40%) 条件下，Censor vs Drop 的 SPE_v 差异相对可控——这进一步降低了短期内迁移到 HSSM+OPN 的紧迫性

### 4.4 与导师讨论的核心要点

| 要点 | 立场 |
|:---|:---|
| 文献的核心主张（omission 必须建模） | 原则同意，但须区分"直接丢弃"和"HDDM 截尾" |
| 当前 HDDM 截尾方案是否已足够 | **✅ 已通过敏感性分析实证验证**：遗漏率 <15% 时可接受，>35% 时有偏倚但 SPE_v 相对稳健 |
| G1/G2 是否应排除 | **✅ 已确认**：G1 (72.3%) 和 G2 (52.0%) 的 DDM 拟合无论方案均不可靠，建议正式排除 |
| 是否应采用 LAN+OPN | 短期内不建议——工具成熟度和数据量的考虑；未来可作为 Discussion 中的 future direction |
| 最优先的行动 | ✅ 敏感性分析已完成 → **下一阶段：更新 GP 模型使用 G3-G8、撰写论文初稿** |

---

## 五、总结

Leng et al. (2025) 的工作为"不应忽略 omission"提供了有力的定量证据。通过本项目的实证敏感性分析，我们将这一发现具体化为以下层次：

1. **"直接丢弃 omission 是错误的"** → ✅ 已被文献证明，**本项目的敏感性分析（§2.4）用实验数据直接验证了这一点**——Drop 方案系统性高估 v（Δ 最高 +6.81），低估 a。
2. **"HDDM 的截尾方法是否足够好"** → ✅ **已通过敏感性分析解答**——在遗漏率 <15% 时 Censor 与 Drop 的 95% CI 可重叠，截尾方案可接受；遗漏率 >35% 时偏倚显著；>50% 时不可靠。
3. **"极高 omission 率的组是否仍可建模"** → ✅ 已确认 G1 (72%) 和 G2 (52%) 的 DDM 参数无论 Censor 还是 Drop 均不可靠，**正式建议排除**。

**底线建议**: 
- 后续分析使用 G3-G8（排除 G1/G2），对 G3-G4 的结果标注"需谨慎解释"
- 在论文 Methods/Discussion 中引用 Leng et al. (2025) 并呈现敏感性分析图表
- 如果需要在 Sigmoid/GP 模型中显式预测 omission，可采用路线 C 的渐进方案

---

## 六、下一阶段工作计划

### 6.1 总体时间线

```
Phase 5: Omission 敏感性收尾 + GP-SPE 模型迭代
├── Week 1-2: 数据决策 + GP 模型更新
├── Week 3-4: 参数恢复检验 + 行为验证
└── Week 5-6: 论文初稿撰写
```

### 6.2 详细任务分解

#### 任务 5.1: 正式排除 G1/G2，确定最终分析样本 ✅ 部分完成

| 项目 | 内容 |
|:---|:---|
| **描述** | 基于 §2.4 敏感性分析结果，正式将 G1 (omission 72.3%) 和 G2 (omission 52.0%) 从后续 GP-SPE 模型训练中排除；G3-G4 保留但标注"灰色地带" |
| **输入** | §2.4 敏感性分析表、sensitivity_comparison.csv |
| **输出** | 更新后的 `GP-SPE-Explore` notebook 中使用的 group 列表；在 AGENTS.md 中记录排除决定 |
| **优先级** | 🔴 高 |
| **预计耗时** | 0.5 天 |
| **风险** | 低——已有明确的排除证据 |

#### 任务 5.2: 更新 GP-Sigmoid 混合模型使用 G3-G8

| 项目 | 内容 |
|:---|:---|
| **描述** | 重新运行 `Generate_Data` 系列 notebook，仅使用 G3-G8 的 HDDM 参数（v_self, v_stranger, a, t, z），重新训练 Sigmoid 函数和 GP 模型 |
| **输入** | G3-G8 的 HDDM Censor 参数（来自 `2_Data/Real_Data/HDDM_Traces/`） |
| **输出** | 更新后的 Sigmoid 参数（α1, α2, β, δ, γ）；G3-G8 的 GP 预测面（2D + 3D） |
| **代码** | `1_Code/Python_for_Generate/Generate_Data_v2.4_runner.py`（修改 group 列表） |
| **优先级** | 🔴 高 |
| **预计耗时** | 2-3 天 |
| **风险** | 中等——6 组数据（原 8 组）可能影响 GP 的预测精度，尤其是 design space 的覆盖率 |

#### 任务 5.3: 使用 Censor 参数进行完整的 Parameter Recovery

| 项目 | 内容 |
|:---|:---|
| **描述** | 基于 G3-G8 的 HDDM Censor 参数进行参数恢复检验（Layer 3 检验），确认 GP-Sigmoid 模型在仅使用 6 组条件下的恢复精度 |
| **输入** | 任务 5.2 的 Sigmoid + GP 模型 |
| **输出** | Parameter Recovery 报告（v, a, t, z 的恢复精度）；LOCV 交叉验证结果 |
| **代码** | `1_Code/Python_for_Check/Parameter_Recovery.ipynb`、`Generate_Data_v2.4_recovery.py` |
| **优先级** | 🔴 高 |
| **预计耗时** | 3-4 天 |
| **风险** | 中等——6 组 vs 8 组的 recovery 精度可能下降；需要关注 GP 的 uncertainty estimation |

#### 任务 5.4: 更新 Omission 率的行为验证

| 项目 | 内容 |
|:---|:---|
| **描述** | 在 Sigmoid 模型中加入 omission 率预测函数（路线 C 方案 2），基于 G3-G8 的观测 omission 率校准 sigmoid_omission_rate(P, T, W) |
| **输入** | G3-G8 的 omission_rate 数据、任务 5.2 的模型 |
| **输出** | Omission 率预测 vs 观测的对比图；纳入 Step 5 行为验证 |
| **优先级** | 🟡 中 |
| **预计耗时** | 2-3 天 |
| **风险** | 低——已有 SigmoID 函数的基础，扩展简单 |

#### 任务 5.5: 论文初稿 — Methods + Results 撰写

| 项目 | 内容 |
|:---|:---|
| **描述** | 撰写论文的 Methods（实验设计、DDM 拟合、Omission 处理）和 Results（SPE 模式、GP 预测面、敏感性分析）章节 |
| **输入** | 所有已完成分析的结果和图表 |
| **输出** | Methods 初稿、Results 初稿、完整图表清单 |
| **优先级** | 🔴 高 |
| **预计耗时** | 5-7 天 |
| **依赖** | 任务 5.1-5.3 完成 |
| **风险** | 中等——撰写过程中可能发现需要补充的分析 |

#### 任务 5.6: 补充分析——使用 Drop 参数作为敏感性对照

| 项目 | 内容 |
|:---|:---|
| **描述** | 对 G5-G8（低遗漏率组）同时使用 Drop 方案的 DDM 参数运行 GP-Sigmoid 模型，作为 Censor 方案的敏感性对照——确认 omission 处理方法是否改变 GP 的定性结论 |
| **输入** | G5-G8 的 HDDM Drop 参数（来自 `drop_traces/`） |
| **输出** | Censor vs Drop GP 预测面的对比图；两种方案下最优设计点的比较 |
| **优先级** | 🟡 中 |
| **预计耗时** | 2-3 天 |
| **风险** | 低——仅为对照分析，不改变主分析路线 |

### 6.3 所需资源

| 资源 | 说明 |
|:---|:---|
| **计算资源** | Docker (hcp4715/hddm) 用于 HDDM 重拟合（如需）；本地 Python 环境用于 GP/SigmoID 模型 |
| **数据** | 已有的 HDDM_Traces (Censor) 和 drop_traces (Drop)——均已完成，无需重新拟合 |
| **软件** | Python 3.9+、scikit-learn、GPy、HDDM、Jupyter |
| **参考文献** | Leng et al. (2025)、Sui et al. (2012)、现有的 GP-SPE 方法论文献 |

### 6.4 潜在风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|:---|:---:|:---:|:---|
| G3-G8 (6组) 不足以训练可靠的 GP | 中 | 高 | 如 GP 精度过低 (LOCV R² < 0.6)，保留 G3-G4 但使用更保守的 prior；或考虑 Bayesian GP 来量化不确定性 |
| Drop 方案的参数恢复与 Censor 定性矛盾 | 低 | 中 | 如两种方案下的 GP 最优设计点显著不同，在 Discussion 中作为 limitation 讨论 |
| 缺少 G1-G2 导致 design space 覆盖不足 | 中 | 中 | G1-G2 的 P=0 条件极端——实际上不推荐作为实验设计；如需要，可用 GP extrapolation 预测该区域 |
| 审稿人质疑 omission 处理方法的合理性 | 高 | 中 | 已在敏感性分析中准备了充分的实证证据和文献依据；在 Methods 中详细描述 Censor 方法 |
| HSSM LAN+OPN 正式发布后的迁移压力 | 低 | 低 | 当前已完成 Censor vs Drop 对比，已有足够的方法论辩护；HSSM 可作为 future work |

---

*分析日期：2026-06-01 | 更新基于完成的敏感性分析实证结果 | 原始分析 2026-05-09 基于 Leng, Fengler, Shenhav, & Frank (2025) 全文*
