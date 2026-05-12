# 项目详细总结与后续代码任务说明书

此文档供其他AI理解并编写可逐行运行的 `.ipynb` 文件。请严格遵循以下描述，确保新AI能准确承接项目方向，避免偏差。

---

## 一、项目最终目标与核心主张

**目标**：证明 **GP+Sigmoid+DDM 混合生成模型** 在捕捉真实自我优势效应（SPE）数据上，显著优于传统的 **纯 Sigmoid+DDM 模型**。通过模型比较、参数景观验证和预测评估，说明引入 GP 能更好地解释实验设计空间对心理机制的非线性、非单调影响，从而推动从“效应检验”走向“设计空间建模与优化”的新范式。

**核心主张**：

- 传统 Sigmoid 模型只能表达单调、固定形状的映射（P,T,W → DDM参数），难以捕捉真实的复杂交互。
- GP 作为数据驱动的残差修正层，在保留理论结构的同时弥补了 Sigmoid 的不足。
- 通过对比两者对真实数据的**行为预测能力**、**参数趋势复现**和**不确定性量化**，可证实 GP 的必要性。

---

## 二、研究背景与架构

- 实验：Self-Matching Task，被试在**不同刺激呈现时间（T）、反应窗口（W）、练习次数（P）**下判断人脸标签（self/stranger）。数据包含 Matching 和 NonMatching 试次，**生成模型仅模拟 Matching 条件**。
- 核心创新：将实验设计参数(P,T,W)形式化为连续设计空间Ω，通过生成模型“Ω → 心理参数(v,a,t0,z) → 行为(RT, response, omission)”进行非参数建模。
- 生成模型结构（最新 v2.4.5）：
  - **S2 机制函数**（理论层）：基于先验知识，Sigmoid 形式调节 `v` 和 `a`。`v` 受 T、P 和 self/stranger 标签影响，`a` 受 M=T+W 影响。
  - **GP 残差层**：输入归一化的(P,T,W)，输出对 v 和 a 的非线性修正。
  - **混合**：`v_mix = w_gp * gp_v + (1-w_gp) * v_s2`，同理 a。
  - **试次噪声**：`v_final ~ N(v_mix, v_noise)`，`a_final` 通过重采样保持正值。
  - **DDM 仿真**：带 deadline 的 Euler 累积，加入 lapse omission 机制，输出 RT、response、omission。
- **关键前提**：真实数据与生成数据的比较必须限于 **Matching 试次**；真实数据需从原始文件中根据 `CorrectKey == matchKey` 恢复 Matching 标签。

---

## 三、已有代码资产与版本总结

以下版本位于项目不同目录，核心思想一致，**最终生成引擎为 v2.4.5，验证框架为 notebook ⑩**。

### 生成模型版本演化

- **初版（纯 Sigmoid）**：`compute_v(T,P,cond)` 用乘积型 Sigmoid；`compute_a(M)` 用单 Sigmoid；z=a/2。可生成整体或指定条件的被试数据。无 GP。
- **V1 GP 替代型**：四个独立 GPR 直接预测 v,a,t0,z，anchor 数据由原 Sigmoid 生成。
- **V2.1–V2.2**：引入 `HybridDDMParameterGenerator` 类，Sigmoid + GP 线性混合，权重 w。但 GP 训练数据为人工合成（sin/cos），且 t0 固定、z 公式有误（后期被实际 z=a/2 覆盖）。
- **V2.4**：**关键版本**。建立 S2 机制函数 + GP 修正 + DDM 仿真结构。保留了 S2 的心理学设定（self/stranger 对 v 的不同乘数，M>600 对 a 的影响）。归一化 P,T,W 输入 GP。混合权重 w_gp。
- **V2.4.2**：增加条件间参数配对可视化。
- **V2.4.3**：增加 P/T/W 对参数和行为的边际效应图。
- **V2.4.4**：引入**漏答（omission）**机制（deadline omission + lapse omission），但存在硬截断 `max(0.1, a_final)` 导致 a=0.1 堆积问题。
- **V2.4.5**：修正 a 用重采样代替硬截断，保留 omission 机制。**这是当前推荐的生成模型**。
- **重要函数/类**：
  - `HybridDDMParameterGenerator`：含 `predict_params(X)` 返回 v, a, t0, z（但 z 实际未用，仿真中设置 `z_final = a_final/2`）。
  - `compute_v_s2(T,P,condition_key)`：S2 漂移率。
  - `compute_a_s2(M)`：S2 边界。
  - `normalize_PTW_to_unit(P,T,W)`：将原始尺度归一化至约[-1,1]。
  - `simulate_ddm_with_deadline`、`compute_lapse_omission_prob`、`sample_a_positive`。
  - 生成函数 `generate_dataset_s2_na_v245` 输出包含所有中间参数和行为结果的数据框。

### 比较验证 notebook

- **⑩ Compare v2.4.3_V2（Matching-only 合理版）**：**最关键的真实数据比较参考**。步骤包括：
  - 读取 `EXP_data_combined.csv` 和 `gp_ddm_v2.4.3_large.csv`（或更新）
  - 恢复 Matching 列：`match_key = ['f','j','j','f']`，`Matching = (CorrectKey == match_key)`。
  - 只保留 Matching 行，并将 `T,W` 转换为毫秒（真实数据中 T,W 为秒，乘 1000）。
  - 生成数据原为连续值，需通过欧氏距离映射到最近的真实条件（最近邻匹配）。
  - 使用 EZ-diffusion 从行为数据反推 v,a,t（z 固定 a/2）。
  - 构建三层趋势对比图：真实行为估计 vs 生成行为估计 vs 生成潜在参数。
  - 包含 Spearman 秩相关矩阵、边缘趋势图。
- **⑫ Compare v2.4.4**：增加 omit_rate 单独比较，参数估计排除 omission 试次。
- **数据路径（基于项目根目录）**：
  - 真实数据：`2_Data/Real_Data/EXP_data_combined.csv`
  - 生成数据 v2.4.3：`2_Data/Generate_Data/Generate_Data_v2.4.3_checks/gp_ddm_v2.4.3_large.csv`
  - 生成数据 v2.4.5：`2_Data/Generate_Data/Generate_Data_v2.4.5_checks/gp_ddm_v2.4.5_large.csv`

**请确保新AI知晓所有路径基于 `BASE_DIR = Path.cwd().parent.parent`，即 notebook 在 `1_Code` 的某子目录下运行时，能回溯到项目根目录。**

---

## 四、当前状态与未完成的关键证明

**已完成**：

- 灵活的数据生成引擎 v2.4.5（含 GP + Omission）。
- 真实数据 Matching-only 下的趋势可视化（但仅基于 EZ-diffusion 粗糙估计）。
- 定性发现：生成模型参数趋势大体跟随真实数据。

**未完成（正是后续任务）**：

- **形式化的模型比较**：没有将纯 Sigmoid 模型与 GP+Sigmoid 模型在相同真实数据上进行拟合和预测性能的统计检验。
- **缺乏层级贝叶斯 DDM 对真实数据的解析**：目前只有 EZ-diffusion 的粗糙点估计，未得到可靠的 v,a,t0 后验分布及不确定性。
- **GP 残差景观的认知解释**尚未做。
- **多指标比较报告**（WAIC、RMSE、参数复原相关系数、残差结构化检验）空白。

---

## 五、后续代码任务清单（按优先级顺序，可分批实现）

以下任务假定用户已有：

- Python 环境：numpy, pandas, matplotlib, scipy, sklearn, seaborn
- 可选：pymc、hddm、arviz（用于层级贝叶斯模型）
- 数据文件已存在于指定路径。

### 🟢 Task 1：真实数据 Matching HDDM 拟合（获取条件水平的后验参数）

**目标**：对真实 Matching 试次，使用层级贝叶斯 DDM 估计每个条件（P,T,W组合）×condition_key（self/stranger）下的 v, a, t0 后验分布。**输入**：`real_match` 数据框（已在之前的比较 notebook 中构造）。**预期输出**：一张表 `real_hddm_params.csv`，包含 `condition_id, label, v_mean, a_mean, t0_mean, v_hdi_lower, v_hdi_upper, ...` 以及被试层面的后验样本。**实现要点**：

1. 数据准备：保留 Matching 试次，确保 `rt_s`（秒）和 `correct`（0/1）列存在。缺失 RT 的试次应标记为 omission 并排除出标准 DDM 拟合（或使用带 omission 的模型，但建议先从 answered only 开始）。
2. 使用 `HDDM` 包或 `PyMC` 自定义层级模型。若用 `HDDM`，需构造包含 `subj_idx`, `rt`, `response`（编码为 1 上界，0 下界）的数据。可以按条件分组估计，但对每个条件分别建模会导致数据稀疏，更好的做法是将条件作为协变量加入回归模型（例如 v ~ label + P + T + W 的交互），但鉴于任务复杂度，可先**按条件分别拟合简单层级模型**，每个条件下被试作为随机效应。
3. Jupyter notebook 中展示 trace 诊断（Rhat<1.05, ESS）。提取后验均值、HDI。
4. 保存为 CSV。

**注意事项**：HHDM 安装可能需要特定依赖，请指明使用 conda 或 pip。如果本地环境有限，可先用 EZ-diffusion 并记录其局限，但正式比较应使用层级模型。

### 🟢 Task 2：构建纯 Sigmoid 对比模型

**目标**：提取现有 S2 函数，封装为一个可训练的“纯 Sigmoid 生成模型”，使其能通过调整内部参数（如 `alaph1, alaph2, beta1, beta2` 等）拟合真实数据，并产生与 GP 模型可比的后验预测分布。
**输入**：真实数据的条件水平行为摘要（RT、正确率、omit rate）或直接使用个体 trial 数据。**预期输出**：一个类 `PureSigmoidModel`，包含：

- `fit(real_data)` 方法：调整内部参数使得生成数据行为分布接近真实数据（最小化 RT、ACC、omit rate 的差异）。可以用基于模拟的优化（如 differential evolution）或近似贝叶斯计算。
- `predict(params, conditions)` 方法：给定设计点（P,T,W,label），输出 `v, a, t0, z`（带试次噪声）并能生成大量 replication。
- `predict_behavior(n_rep, conditions)`：返回模拟的行为数据集。
- 优化后的最佳参数存储。

**实现要点**：

1. 复制 `compute_v_s2`, `compute_a_s2` 等函数，参数化可调系数。
2. 生成仿真函数同 v2.4.5 但不含 GP 项（w_gp=0）。需包含 omission 机制以匹配真实数据，但为简化最初可仅针对 answered 试次拟合。
3. 拟合目标：最小化各条件 RT 中位数、ACC 均值、omit rate 均值的加权 RMSE。真实数据的这些统计量可从 Task 1 数据框计算。
4. 输出优化后参数和拟合优度。

### 🟢 Task 3：用真实数据训练 GP+Sigmoid 混合模型

**目标**：将 Task 1 得到的条件水平参数后验均值（v_mean, a_mean）作为训练目标，训练混合模型的 GP 部分，即让 GP 学习 Sigmoid 预测与真实参数之间的残差。
**输入**：

- X：设计的条件对应的归一化 (Pn, Tn, Wn)（注意条件对应的 label 如何处理？因为 S2 部分已经通过 condition_key 产生了不同起点的 v，所以 GP 输入可不含 condition，但您需决定 GP 是否学习条件效应残差。根据 v2.4.5，GP 只以 P,T,W 为输入，condition 效应由 S2 负责。因此训练数据按 condition 分别生成各自的目标残差。）
- Y_v：`v_real - compute_v_s2(T,P,cond)` （S2 预测的机制值）
- Y_a：`a_real - compute_a_s2(M)`
  **预期输出**：训练好的 `gen` 对象（`HybridDDMParameterGenerator`），其中 GP 已学习真实残差。
  **实现步骤**：

1. 对于每个真实条件，计算 S2 预测值，获取残差。
2. 将所有条件的残差作为训练点，训练 `gen.gp_v` 和 `gen.gp_a`。（注意：当前 `HybridDDMParameterGenerator` 的 `fit_gp` 方法接受 X, Y_v, Y_a，我们可以直接使用）。
3. 保存训练后的模型对象（pickle 或 joblib）。

### 🟢 Task 4：模型比较主分析（指标计算与可视化）

**目标**：用定量的方式比较纯 Sigmoid 模型和 GP+Sigmoid 模型对真实数据的解释力。
**输入**：训练后的两个模型、真实数据 trial-level 数据（用于后验预测检查）。**预期输出**：包含 WAIC（或 LOO-CV）、行为 RMSE、参数复原相关系数的表格，以及文章级对比图（至少4张关键图）。**实现要点**：

1. **后验预测分布生成**：对每一个真实条件（P,T,W,label），两个模型分别生成大量（如 1000 个）模拟数据集（与真实试次数相同），每个模拟数据集得到汇总统计量（RT中位数、正确率、遗漏率）。
2. **WAIC 计算**：由于模型不是概率模型，不能直接计算 WAIC。可改用**对数预测概率**估计：对每个真实 trial，计算模型生成该 trial 结果（RT, choice, omission状态）的概率密度。因为行为数据含 omission，可对 omission 建模为 Bernoulli（概率来自 lapse+deadline），对 answered trial 使用 DDM 似然（如 Wiener 分布）。你需要实现似然函数。如果实现复杂，可使用**预测误差变体**：报告模型在每个条件上的平均预测对数密度（平均化模拟 trial）。
3. 简单替代：报告**每个条件的 RMSE**，并做配对 t 检验比较两个模型在所有条件上的误差。
4. **参数复原**：比较真实 HDDM 参数均值与模型预测的潜在参数（混合模型可直接用 `v_mix`，纯 Sigmoid 用其机制输出），计算 Spearman 相关和平均绝对误差。
5. **图示**：
   - **参数景观对比图**：P-T 网格上 v 和 a 的热图（分 self/stranger），三列：真实估计、GP 模型预测、Sigmoid 模型预测。
   - **WAIC 差异或 RMSE 差异森林图**。
   - **GP 残差景观 Δv 热图**：GP 模型预测的 v 减去 Sigmoid 预测的 v，在 P-T 平面上展示。
   - **后验预测走廊**：真实数据 RT 分布直方图叠加两个模型的后验预测分布线。
6. 务必使用 `FIG_DIR` 保存图片。

### 🟡 Task 5：GP 超参数分析与认知解释

**目标**：分析 GP 核的 lengthscale 和方差，提供心理学解释。
**输入**：训练后 GP 模型的 `kernel_` 属性。
**输出**：文本解释和可能的信息图。
**要点**：提取 RBF 核的 lengthscale 数组，对应 P,T,W 维度。讨论哪个设计因素最重要。信号方差大小表示 Sigmoid 遗漏的结构量。

---

## 六、重要注意事项（避免新AI理解错误）

1. **Matching 条件**：所有比较必须限定在 Matching 试次，生成数据默认就是 Matching 设定，真实数据通过 `CorrectKey` 恢复。
2. **单位**：真实数据文件中的 `T` 和 `W` 是**秒**，必须乘以 1000 转毫秒后才能与生成数据统一；`RT` 保持秒。生成数据中的 `T`/`W` 已经是毫秒。
3. **生成数据条件映射**：生成数据中每个被试的 P,T,W 是连续随机值，需要用最近邻方法匹配到真实实验的离散条件。已在 notebook ⑩ 提供了 `nearest_real_condition` 函数，请复用。
4. **z 的设定**：生成仿真中实际使用 `z = a_final / 2`，因此估计时也使用对称起点。
5. **遗漏试次处理**：v2.4.5 产生 omission，对应列 `responded`, `omission`。真实数据也有 NA。比较时需要分别处理：参数估计用 answered trial，omission rate 单独比较。
6. **批次运行**：用户会逐任务请求代码，请确保每个生成的 notebook 可以独立运行，但共享必要的预处理函数（可将它们放在一个辅助 `.py` 文件中或复写）。
7. **随机种子**：为可重复性，设置 `np.random.seed(42)`。
8. **路径**：使用 `pathlib.Path` 并基于 notebook 所在位置获取项目根，例如：
   ```python
   BASE_DIR = Path.cwd().parent.parent  # 如果 notebook 在 1_Code/Python_for_Generate 下
   ```

   若另一个 AI 不清楚目录结构，请使用相对路径或让用户手动指定。

---

## 七、建议的第一份 notebook 内容（Task 1）

当用户说“先写 Task 1”时，请提供完整的 `.ipynb` 结构，包括：

- 导入包
- 路径设置
- 读取与预处理真实数据（精确复用 notebook ⑩ 的步骤，生成 `real_match`）
- 构造 HDDM 输入数据框
- 定义并采样层级模型（用 HDDM 或 PyMC）
- 诊断检查
- 提取后验汇总并保存至文件
- 初级可视化（迹图、后验直方图）

**确保每段代码都有注释，并说明为什么这么做。**

此总结已足够详细，可供其他 AI 无误地理解项目背景、现有进展和具体待办任务。开始请求即可。
