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

---

## 四、建议

### 4.1 短期可行方案（立即可执行）

```
Step 1: 排除 G1, G2 (omission >50%)
  → 这类组的 DDM 拟合已不可靠，无论用什么方法

Step 2: 对 G3-G8 维持当前 HDDM 截尾方案
  → HDDM 的 censored 方法已经比"直接丢弃"好

Step 3: 在敏感性分析中对比：
  (a) 当前方案: censored (rt=T+W, response=0)
  (b) 替换方案: 直接丢弃 omission (p_outlier=0)
  → 观察两种方案下 DDM 参数是否有显著差异
  → 若差异大→确认 omission 处理方法对结论有影响
  → 若差异小→当前方案可能已经足够
```

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

- 追踪 HSSM 的 LAN+OPN 功能正式发布
- 在新一轮数据采集后（Phase 4 建议的 4-6 个新条件），样本量增加后再评估切换工具链的性价比
- 当前 8 组条件下，切换 HSSM 的收益可能不如新增数据条件

### 4.4 与导师讨论的核心要点

| 要点 | 立场 |
|:---|:---|
| 文献的核心主张（omission 必须建模） | 原则同意，但须区分"直接丢弃"和"HDDM 截尾" |
| 当前 HDDM 截尾方案是否已足够 | **需实证验证**（见 §4.1 Step 3） |
| 是否应采用 LAN+OPN | 短期内不建议——工具成熟度和数据量的考虑 |
| 最优先的行动 | 排除 G1/G2 + 新增实验条件 + 实证对比 omission 处理方案 |

---

## 五、总结

Leng et al. (2025) 的工作为"不应忽略 omission"提供了有力的定量证据。但在将这一发现映射到本项目时，需要区分几个层次：

1. **"直接丢弃 omission 是错误的"** → 已被文献证明，本项目当前也不这样做
2. **"HDDM 的截尾方法是否足够好"** → 文献未直接评估，需要设计对比实验
3. **"极高 omission 率的组是否仍可建模"** → 文献未覆盖此范围，可能性低

**底线建议**: 在论文的 Methods/Discussion 中明确讨论 omission 处理方法的潜在影响，并引用 Leng et al. (2025) 作为方法论依据。如果需要在 Sigmoid/GP 模型中显式预测 omission，可采用路线 C 的渐进方案。

---

*分析日期：2026-05-09 | 基于 Leng, Fengler, Shenhav, & Frank (2025) 全文*
