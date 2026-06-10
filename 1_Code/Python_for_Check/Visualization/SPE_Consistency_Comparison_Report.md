# SPE（自我优势效应）计算方法一致性检查报告

## 一、概述

本报告旨在对比分析两个位置的SPE（Self-Prioritization Effect，自我优势效应）计算方法，确保算法逻辑、计算公式和数据处理流程的一致性。

**检查位置：**
1. **可视化应用**：`visualization_app.html` - 前端JavaScript实现
2. **探索性分析**：`exploratory_workflow.py` - 后端Python实现

---

## 二、SPE定义与计算公式对比

### 2.1 visualization_app.html（JavaScript）

**文件位置：** [visualization_app.html](file:///d:/GitHub_programe/GitHub/Guassion-Process-Experiment-Design/1_Code/Python_for_Check/Visualization/visualization_app.html)

**核心计算逻辑：**

#### 方法1：分位数级别的SPE计算（第1500-1603行）
```javascript
// renderDataSPE() 函数
const selfAcc = selfBin.length > 0 ? selfBin.filter(t => t.Correct === 1).length / selfBin.length * 100 : 0;
const strangerAcc = strangerBin.length > 0 ? strangerBin.filter(t => t.Correct === 1).length / strangerBin.length * 100 : 0;
const spe = selfAcc - strangerAcc;  // SPE = Self ACC(%) - Stranger ACC(%)
```

#### 方法2：被试级别的SPE计算（第1635-1669行）
```javascript
// renderMultiSubjectComparison() 函数
const selfACC = selfResp > 0 ? selfCorrect / selfResp * 100 : null;
const strangerACC = strangerResp > 0 ? strangerCorrect / strangerResp * 100 : null;
const accSelfMinusOther = (selfACC !== null && strangerACC !== null) ? selfACC - strangerACC : null;
const spe = accSelfMinusOther;  // SPE = Self ACC(%) - Stranger ACC(%)
```

**定义总结：**
| 指标 | 公式 | 取值范围 | 解读 |
|------|------|----------|------|
| SPE | Self ACC(%) - Stranger ACC(%) | [-100, 100] | 正值=Self更准，负值=Stranger更准 |

---

### 2.2 exploratory_workflow.py（Python）

**文件位置：** [exploratory_workflow.py](file:///d:/GitHub_programe/GitHub/SPE_Database/Datasets/8_Exploratory_Analysis/exploratory_workflow.py)

**核心计算逻辑（第568-611行）：**

```python
def _compare_self_vs_others(dataset_id: str, subject_level: pd.DataFrame, measure: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities = sorted(set(subject_level["Shape_Standardized_Identity"].unique()) - {"Self"})
    
    for identity in identities:
        subset = subject_level.loc[subject_level["Shape_Standardized_Identity"].isin(["Self", identity])].copy()
        wide = subset.pivot_table(index="Subject", columns="Shape_Standardized_Identity", values="value", aggfunc="mean")
        
        paired = wide[["Self", identity]].dropna()
        if len(paired) < MIN_SUBJECTS_FOR_EFFECT:
            continue
        
        if measure == "RT_ms":
            diff = paired[identity] - paired["Self"]  # RT差异 = Other RT - Self RT
            direction = "positive_d_means_self_faster"
        else:
            diff = paired["Self"] - paired[identity]  # ACC差异 = Self ACC - Other ACC
            direction = "positive_d_means_self_more_accurate"
        
        sd_diff = float(diff.std(ddof=1))
        dz = float(diff.mean() / sd_diff) if len(diff) > 1 and sd_diff > 0 else np.nan
        # ... t检验等统计分析
```

**定义总结：**
| 指标 | 公式 | 取值范围 | 解读 |
|------|------|----------|------|
| SPE差异 | Self ACC - Other ACC | [-1, 1]（比例） | 正值=Self更准 |
| Cohen's dz | mean(diff) / sd(diff) | 连续值 | 标准化效应量 |

---

## 三、核心差异对比

### 3.1 条件范围差异

| 维度 | visualization_app.html | exploratory_workflow.py |
|------|----------------------|------------------------|
| **条件限制** | 无明确限制，使用所有条件数据 | **仅Matching条件**（第537行） |
| **代码依据** | 直接使用formal试次 | `matching_df = sub.loc[sub["Matching"] == MATCHING_LABEL].copy()` |
| **设计意图** | 可视化探索所有条件 | 遵循Sui(2012)范式，SPE定义为Matching条件下的自我优势 |

**关键差异**：探索性分析严格遵循原始SPE定义，仅在Matching条件下计算；而可视化应用未做此限制，可能包含NonMatching条件的数据。

---

### 3.2 对比对象范围

| 维度 | visualization_app.html | exploratory_workflow.py |
|------|----------------------|------------------------|
| **对比对象** | 仅 Self vs Stranger | Self vs 所有其他身份类型 |
| **支持的身份** | Self, Stranger | Self, Close, Acquaintance, Celebrity, Stranger, NonPerson |
| **代码依据** | `t.Identity === 'Self'` / `t.Identity === 'Stranger'` | `identities = sorted(set(...) - {"Self"})` |

**关键差异**：探索性分析支持多种身份类型的对比，而可视化应用仅限于Self vs Stranger的对比。

---

### 3.3 效应量指标

| 维度 | visualization_app.html | exploratory_workflow.py |
|------|----------------------|------------------------|
| **主要指标** | 原始百分比差值 (%) | Cohen's dz（标准化效应量） |
| **计算公式** | `spe = selfAcc - strangerAcc` | `dz = mean(diff) / sd(diff)` |
| **统计检验** | 无 | 配对t检验 |
| **数据级别** | 试次级别/分位数级别 | 被试级别聚合后计算 |

---

### 3.4 数据预处理流程

| 步骤 | visualization_app.html | exploratory_workflow.py |
|------|----------------------|------------------------|
| **试次筛选** | `stage === 'formal' && RT !== null` | 相同 |
| **RT修剪** | 无 | **MAD方法修剪**（第498-511行） |
| **RT范围** | 无限制 | `MIN_RT_MS=150` 至 `MAX_RT_MS=3000`（毫秒） |
| **数据清洗** | 无额外QC | 多步骤QC检查（第514-555行） |

**RT修剪实现（exploratory_workflow.py第498-511行）：**
```python
def _robust_trim_subject_rt(df: pd.DataFrame) -> pd.DataFrame:
    for _, subject_df in df.groupby("Subject", sort=False):
        median = float(subject_df["RT_ms"].median())
        mad = float(np.median(np.abs(subject_df["RT_ms"] - median)))
        if mad == 0 or math.isnan(mad):
            continue
        modified_z = 0.6745 * (subject_df["RT_ms"] - median) / mad
        # 保留 |modified_z| <= 3.5 的数据
```

---

### 3.5 数据级别差异

| 维度 | visualization_app.html | exploratory_workflow.py |
|------|----------------------|------------------------|
| **计算级别** | 试次级别 → 分箱聚合 | 被试级别聚合 → 效应量计算 |
| **聚合方式** | 按RT分位数分箱 | 被试内平均 |
| **样本量要求** | 无最低要求 | `MIN_SUBJECTS_FOR_EFFECT = 5` |

---

## 四、一致性分析

### 4.1 相同点

1. **核心定义一致**：两者均定义SPE为 Self优势 = Self表现 - Other表现
2. **ACC计算方向一致**：均为 Self ACC - Other ACC（正值表示Self优势）
3. **RT计算方向一致**：均为 Other RT - Self RT（正值表示Self更快）
4. **数据来源一致**：均基于trial-level的行为数据（RT, ACC）

### 4.2 差异点

| 差异类型 | 可视化应用 | 探索性分析 | 影响评估 |
|----------|----------|------------|----------|
| **条件范围** | 全部条件 | 仅Matching | **高** - 定义不一致 |
| **对比对象** | 仅Stranger | 多种身份 | **中** - 范围不同 |
| **效应量指标** | 原始差值(%) | Cohen's dz | **中** - 可比性不同 |
| **RT修剪** | 无 | MAD方法 | **中** - 数据质量不同 |
| **统计检验** | 无 | 配对t检验 | **低** - 仅影响推断 |

### 4.3 潜在问题

**问题1：条件范围不一致**
- 可视化应用可能在NonMatching条件下计算SPE，而原始SPE定义仅适用于Matching条件
- 这会导致计算结果的含义不同

**问题2：RT异常值处理**
- 可视化应用未进行RT修剪，可能包含异常值
- 探索性分析使用MAD方法修剪极端值

**问题3：效应量可比性**
- 原始百分比差值受量表范围限制
- Cohen's dz是标准化指标，更适合跨研究比较

---

## 五、建议与优化方案

### 5.1 针对visualization_app.html的建议

**建议1：增加条件筛选限制**
```javascript
// 在SPE计算前增加条件筛选
const matchingTrials = formal.filter(t => t.Condition === 'Matching');
// 仅使用Matching条件计算SPE
```

**建议2：增加RT范围过滤**
```javascript
const MIN_RT = 0.15;  // 150ms
const MAX_RT = 3.0;   // 3000ms
const validTrials = formal.filter(t => t.RT >= MIN_RT && t.RT <= MAX_RT);
```

**建议3：在文档中明确说明计算范围**
- 在SPE图表标题或说明中注明是否包含NonMatching条件

### 5.2 统一计算标准建议

| 标准项 | 建议值 | 说明 |
|--------|--------|------|
| **条件范围** | 仅Matching | 符合原始SPE定义 |
| **对比对象** | Self vs Stranger | Stranger是最常见的对比对象 |
| **RT范围** | 150ms - 3000ms | 排除过快和过慢的反应 |
| **RT修剪** | MAD方法(z<=3.5) | 稳健的异常值处理 |
| **效应量** | 同时报告原始差值和dz | 兼顾可解释性和可比性 |

---

## 六、总结

### 6.1 一致性评估

| 评估维度 | 一致性 | 说明 |
|----------|--------|------|
| 核心定义 | **高** | SPE = Self - Other 的方向一致 |
| 条件范围 | **低** | 可视化未限制Matching条件 |
| 对比对象 | **中** | 可视化仅支持Stranger |
| 数据处理 | **中** | 探索性分析有更多QC步骤 |
| 效应量指标 | **低** | 使用不同的指标类型 |

### 6.2 关键发现

1. **主要差异在于条件范围**：可视化应用未限制仅在Matching条件下计算SPE，这与SPE的原始定义不一致
2. **探索性分析更严谨**：包含RT修剪、统计检验、QC检查等步骤
3. **指标类型不同**：可视化使用原始百分比，探索性分析使用Cohen's dz

### 6.3 行动建议

1. **立即修正**：在visualization_app.html中增加条件筛选，确保SPE仅在Matching条件下计算
2. **增加文档说明**：在可视化界面中明确说明SPE的计算范围
3. **统一数据处理**：考虑在可视化应用中增加RT范围过滤
4. **保持指标差异**：两种指标各有优势（原始值易解释，dz适合比较），可同时展示

---

**报告生成日期**：2026年6月10日  
**报告版本**：v1.0