# 项目路线图：基于高斯过程的实验设计优化

> Self-Matching Task（Sui et al., 2012）框架下，使用漂移扩散模型（DDM）+ 高斯过程（GP）优化实验设计参数(P, T, W)，最大化自我优势效应（SPE）的捕捉精度。

---

## Phase 0：项目初始化与基线建立

### 0.1 实验数据采集 ✅ 已完成

- 采集 8 组真实被试数据（共 88 名被试）
- 实验设计空间 Ω = (P, T, W, M)，其中 M = T + W
- 8 组实验条件覆盖不同参数水平：

| 组别 | P (练习次数) | T (刺激呈现, ms) | W (反应窗口, ms) | M (最大允许时间, ms) | 被试数 |
| :--: | :----------: | :--------------: | :--------------: | :------------------: | :----: |
|  G1  |      0      |        30        |       300       |         330         |   11   |
|  G2  |      0      |        30        |       600       |         630         |   12   |
|  G3  |     120     |        30        |       600       |         630         |   10   |
|  G4  |     120     |        80        |       600       |         680         |   11   |
|  G5  |      8      |       100       |       1100       |         1200         |   11   |
|  G6  |     120     |       500       |       1500       |         2000         |   10   |
|  G7  |     120     |       30*       |       800       |         880         |   12   |
|  G8  |     120     |        80        |       800       |         880         |   11   |

> ⚠️ \*G7 的 T 值存在元数据与行为数据不一致问题（元数据标记为 T=30ms，但原始数据实际 T=80ms），详见 Phase 3.2。

### 0.2 基线 Sigmoid+DDM 生成模型 ✅ 已完成

- **文件**：`1_Code/Python_for_Generate/Generate_Data_v1.ipynb`、`S2 gen_data_jh.ipynb`
- **核心函数**：
  - `compute_v_s2(T, P, condition_key)` — 漂移率 v 的 Sigmoid 参数化
  - `compute_a_s2(M)` — 决策边界 a 的 Sigmoid 参数化
  - `v_P_Function(P)` — 练习效应对 v 的非线性调制
- **默认参数**：alaph1=1.5, alaph2=-0.4, beta1=0.2, beta2=0.0, gamma=0.2

### 0.3 生成模型迭代（v2.x 系列） ✅ 已完成

- **文件**：`Generate_Data_v2.1-2.4.ipynb`
- **稳定版本**：`Generate_Data_v2.4_runner.py`
- **v2.4 特点**：Sigmoid + GP 混合方式（GP 在数据生成层面，w=0.5 加权）
- 参数恢复检验：`Generate_Data_v2.4_recovery.py`

---

## Phase 1：GP 角色定位与方法论确立 ✅ 已完成

### 1.1 GP 的三种可能角色分析 ✅ 已完成

经过对 v1、v2.4、v3 三个版本的对比分析，明确了 GP 在框架中的三种可能定位：

| 角色          | 描述                                                      | 对应版本           | 评估              |
| :------------ | :-------------------------------------------------------- | :----------------- | :---------------- |
| A) 代理模型   | GP 替代 Sigmoid，直接学习 (P,T,W) → DDM参数 的映射       | v1 (GP on anchors) | 清晰但理论先验弱  |
| B) 数据层混合 | Sigmoid + GP 加权平均生成 RT/ACC (w=0.5)                  | v2.4               | 有拟合风险        |
| C) 残差捕捉   | Sigmoid 提供理论先验，GP 学习 Sigmoid→真实DDM参数 的残差 | 当前方案           | **推荐** ✅ |

### 1.2 最终决策：分层验证策略 ✅ 已完成

```
Sigmoid 理论先验 (校准后)
        +
GP 捕捉残差 (DDM参数层面)
        =
GPSigmoidHybridModel
        ↓
交叉验证（LOCV）: Leave-One-Condition-Out
        ↓
外部验证（新实验条件的行为/参数双层面验证）
```

---

## Phase 2：HDDM 真实数据参数提取 ✅ 已完成

### 2.1 HDDM 数据预处理 ✅ 已完成

- **文件**：`1_Code/Python_HDDM/step1_prepare_data.py`
- **输入**：`2_Data/Real_Data/UnExtact/raw/EXP_data_group*.csv` (88 个原始数据文件)
- **输出**：`2_Data/Real_Data/HDDM_Ready/hddm_data_group*.csv` (8 组 HDDM-ready 数据)
- **处理逻辑**：
  - 仅保留 `stage='formal'` 的实验试次
  - 仅保留 Matching 试次：`(circle,self) | (square,stranger)`
  - NA 试次标记为 `omission=1`，RT 设置为 T+W（右截尾）
  - 检验模型列：`subj_idx, rt, response, identity, omission`

### 2.2 Docker HDDM 分层模型拟合 ✅ 已完成

- **文件**：`1_Code/Python_HDDM/step2_hddm_fit.py`
- **拟合环境**：Docker `hcp4715/hddm`
- **模型规格**：
  - 分层贝叶斯 DDM（Hierarchical DDM）
  - `depends_on={"v": "identity"}` — v 依赖 identity (self vs stranger)
  - MCMC: 3000 samples, 500 burn-in, p_outlier=0.05
  - 每组独立拟合
- **输出**：`2_Data/Real_Data/HDDM_Traces/hddm_data_group*_traces.npz` 和 `*_stats.csv`

### 2.3 参数提取与整合 ✅ 已完成

- **文件**：`1_Code/Python_HDDM/step3_extract_params.py`
- **输出**：`2_Data/Real_Data/HDDM_Traces/all_groups_ddm_params.csv`
- **关键发现**：

| 组别 | v_self | v_stranger |      SPE_v      |  a  | t(s) |   z   |       遗漏率%       |
| :--: | :----: | :--------: | :-------------: | :--: | :---: | :---: | :------------------: |
|  G1  | -3.39 |   -3.36   | **-0.02** | 2.01 | 0.263 | 0.749 | **72.3%** ⚠️ |
|  G2  | -2.53 |   -2.62   | **+0.09** | 1.18 | 0.436 | 0.553 | **52.4%** ⚠️ |
|  G3  | -1.89 |   -1.73   | **-0.16** | 1.22 | 0.348 | 0.565 |        38.6%        |
|  G4  | +0.64 |   -0.02   | **+0.66** | 1.42 | 0.353 | 0.571 |        38.4%        |
|  G5  | +1.35 |   +0.64   | **+0.71** | 1.34 | 0.398 | 0.452 |        10.9%        |
|  G6  | +1.81 |   +1.17   | **+0.65** | 1.48 | 0.663 | 0.714 |         5.6%         |
|  G7  | +1.20 |   +0.75   | **+0.45** | 1.09 | 0.405 | 0.497 |        14.3%        |
|  G8  | +2.81 |   +2.55   | **+0.26** | 2.42 | 0.352 | 0.498 |        14.7%        |

---

## Phase 3：GP+Sigmoid 混合模型构建与验证

### 3.1 Sigmoid 参数校准 ✅ 已完成

- **两个版本**：
  - `GP_Sigmoid/sigmoid_calibration.py` — **初版**，使用 HDDM 原始参数文件
  - `GP_Sigmoid_Cleaned/` (内含) — **Cleaned 版**，纠正元数据后重新校准
- **方法**：差分进化算法（Differential Evolution）最小化 Sigmoid 预测 vs HDDM 真实参数的 RMSE
- **校准对象**：alaph1, alaph2, beta1, beta2, gamma, base_scale_v, base_scale_a (7个参数)

**两版校准结果对比：**

| 参数                  | 初版 (GP_Sigmoid) | Cleaned版 (GP_Sigmoid_Cleaned) | 默认值 |
| :-------------------- | :---------------: | :----------------------------: | :----: |
| alaph1 (self增强)     |       0.549       |             0.199             |  1.5  |
| alaph2 (stranger调制) |      -0.220      |             -0.404             |  -0.4  |
| beta1 (高M边界)       |      -0.827      |             -0.826             |  0.2  |
| beta2 (低M边界)       |       1.000       |             1.000             |  0.0  |
| gamma (练习效应)      |       0.701       |             0.640             |  0.2  |
| base_scale_v          |       1.026       |             1.326             |  3.0  |
| base_scale_a          |      10.000      |             10.000             |  3.0  |
| 最终RMSE              |       1.614       |             1.617             |   —   |

> ⚠️ 校准后 alaph1 远小于默认值（0.2 vs 1.5），说明真实数据中 self 的相对增强远小于初始假设。base_scale_a 触及上界（10.0），提示 a 的 Sigmoid 参数化可能需要重新考虑。

### 3.2 数据清洗与元数据纠正 ⚠️ 进行中

- **发现的问题**：
  1. **G7 元数据不一致**：行为数据 T=30ms，但原始文件和文件名中 T=80ms — **需确认实际采集参数**
  2. **G1-G2 高遗漏率**：遗漏率 >50%，HDDM 参数估计可靠性存疑
  3. **G7-G8 设计重复**：两者均为 (P=120, T=80, W=800)，但得到不同的 DDM 参数（G7: SPE=0.45, G8: SPE=0.26）— 可能反映被试间变异
- **敏感性分析**：已生成多种排除策略的数据表（`step2_sensitivity_*.csv`）

### 3.3 GP+Sigmoid 混合模型训练 ✅ 已完成

- **模型文件**：`1_Code/Python_HDDM/GP+Sigmoid/gp_sigmoid_hybrid_model.py`
- **核心架构**：`GPSigmoidHybridModel`
  - 对 5 个 DDM 参数（v_self, v_stranger, a, t, z）各训练一个独立 GP
  - GP 学习 `真实HDDM参数 - Sigmoid预测值` 的残差
  - 最终预测 = Sigmoid预测 + GP残差预测
- **GP 核函数**：ConstantKernel × RBF + WhiteKernel

### 3.4 模型交叉验证 ⚠️ 初步完成，需关注结果

**训练集拟合（in-sample）：**

| 目标       | RMSE |   r   |
| :--------- | :---: | :---: |
| v_self     |  ~0  | ~1.00 |
| v_stranger |  ~0  | ~1.00 |
| a          | 0.208 | 0.900 |
| t          | 0.008 | 0.998 |
| z          |  ~0  | ~1.00 |
| SPE_v      |  ~0  | ~1.00 |

**LOCV（Leave-One-Condition-Out）：**

| 目标       |      RMSE      |          r          |
| :--------- | :------------: | :------------------: |
| v_self     | **2.21** | **-0.09** ⚠️ |
| v_stranger |      1.89      |         0.19         |
| a          |      0.88      |      -0.44 ⚠️      |
| t          |      0.14      |      -0.31 ⚠️      |
| z          |      0.19      |      -0.54 ⚠️      |
| SPE_v      |      0.72      |      -0.09 ⚠️      |

> ⚠️ **关键警示**：LOCV 中 DDM 参数层面的交叉验证表现很差（负相关占多数），**但仅有 8 个训练点**，这使得 LOCV 极其不稳定。结论：8 个条件不足以支持 GP 的有效泛化，必须增加实验条件。

### 3.5 行为层面验证 ✅ 已完成

尽管 DDM 参数层面的 LOCV 不理想，行为层面的模拟验证表现出色：

| 指标            | 层面      |  RMSE  |         r         |
| :-------------- | :-------- | :-----: | :----------------: |
| Accuracy        | identity  |  0.153  | **0.968** ✅ |
| Omission Rate   | identity  |  0.145  | **0.923** ✅ |
| Correct RT Mean | identity  | 66.8 ms | **0.981** ✅ |
| SPE RT          | condition | 21.1 ms | **0.843** ✅ |

> ✅ **积极信号**：行为层面验证效果良好，说明 GP+Sigmoid 模型**在行为重建层面是有效的**，即使 DDM 参数层面的 GP 泛化受限于数据量。

---

## Phase 4：实验设计优化与候选点推荐 ⚠️ 初步完成

### 4.1 基于不确定性的候选实验点 ✅ 已完成

- **方法**：计算 GP 在新设计点的预测不确定性 → 选最高不确定性 top-20 → 排除已有设计点的邻近区域
- **输出**：`step6_candidate_design_points.csv` (20 个推荐新实验条件)
- **不确定性分解**：

| rank | P |  T  |  W  | v_self_std | v_stranger_std | a_std | t_std | z_std |
| :--: | :-: | :-: | :-: | :--------: | :------------: | :---: | :---: | :---: |
|  1  | 35 | 500 | 300 |    2.09    |      2.03      | 0.45 | 0.13 | 0.13 |

- **主要不确定性来源**：v_self 和 v_stranger（各约 2.1），远大于 a/t/z
- **最高不确定性区域**：T=500ms, W=300ms 附近（低 W + 高 T 的组合）

### 4.2 下一轮实验设计建议 📋 待执行

推荐的优先实验条件：

1. **T=500ms, W=300ms, P=15~45** — 高不确定性区域，填补设计空间
2. **解决 G7/G8 重复设计问题** — 区分或合并
3. **新增中间 P 水平**：P=30, P=60（当前仅有 P=0, 8, 120）

---

## Phase 5：模型应用与最终产出 📋 待执行

### 5.1 剩余问题处理

- [X] **G7 元数据确认**：确认 T 的真实值是 30ms 还是 80ms
- [ ] **G1/G2 数据质量评估**：是否需要从建模中排除高遗漏组
- [ ] **文献调研**：DDM 参数取值范围的心理/统计合理性评估
- [ ] **模型稳健性检验**：增加数据量后重新 LOCV

### 5.2 论文结构规划（建议）

1. **Introduction**：SMT 范式 + DDM + 实验设计优化的必要性
2. **Methods**：
   - 2.1 8 组实验设计空间
   - 2.2 Sigmoid 理论参数化（Sigmoid v/a functions）
   - 2.3 GP+Sigmoid 混合模型（GP residual capture）
   - 2.4 验证策略（LOCV + 外部新数据）
3. **Results**：
   - HDDM 提取的真实 DDM 参数模式
   - GP+Sigmoid 模型拟合与 LOCV
   - 行为层面验证
   - 候选实验设计点推荐
4. **Discussion**：
   - GP 方法的优势与局限（小样本问题）
   - 与 Sui et al. (2012) 的对比
   - 实验设计优化的实际指导意义

### 5.3 关键文献参考

| 主题             | 关键文献                                                                                                             |
| :--------------- | :------------------------------------------------------------------------------------------------------------------- |
| SMT 范式         | Sui, J., He, X., & Humphreys, G. W. (2012). Perceptual effects of self-prioritization.*JEP:HPP*, 38(5), 1105-1114. |
| DDM 参数范围     | Ratcliff, R., & McKoon, G. (2008). The diffusion decision model.*Neural Computation*, 20(4), 873-922.              |
| GP in Psychology | Schulz, E., Speekenbrink, M., & Krause, A. (2018). A tutorial on GP regression.*JMP*, 85, 1-16.                    |
| 实验设计优化     | Myung, J. I., Cavagnaro, D. R., & Pitt, M. A. (2013). Optimal experimental design.*JMP*, 57(3), 202-217.           |

---

## 📊 项目总体进度

| Phase   | 内容                    | 状态                        |
| :------ | :---------------------- | :-------------------------- |
| Phase 0 | 实验数据采集 + 基线模型 | ✅ 已完成                   |
| Phase 1 | GP 角色定位             | ✅ 已完成                   |
| Phase 2 | HDDM 参数提取           | ✅ 已完成                   |
| Phase 3 | GP+Sigmoid 建模 + 验证  | ✅ 初步完成（⚠️ 8点不足） |
| Phase 4 | 候选点推荐 + 下一轮实验 | 📋 待执行                   |
| Phase 5 | 论文撰写与产出          | 📋 待执行                   |

### 当前最大瓶颈

> **仅有 8 个设计条件**是项目当前面临的核心限制。GP 在 8 个点上无法有效泛化（LOCV 失败），需要在 Phase 4 中至少新增 4-6 个实验条件。

---

*最后更新：2026-05-09 | 版本：v0.1*
