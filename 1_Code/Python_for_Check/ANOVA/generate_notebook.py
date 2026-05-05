import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13.0"
    }
}

cells = []

def add_md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def add_code(source):
    cells.append(nbf.v4.new_code_cell(source))

add_md("""# 阶段一：行为现象确认 — 贝叶斯混合效应模型分析

## Self-Matching Task 中实验设计参数对自我优势效应（SPE）的调节

**分析框架**: PyMC + Formulae（Python 版 brms 等效方案）

**核心问题**: 实验设计空间组（Group）是否调节 Matching × Identity 交互，即不同 P/T/W 参数下 SPE 的强度是否存在差异？

**六组实验设计参数**:
| 组别 | P (练习) | T (刺激呈现) | W (反应窗口) |
|------|---------|-------------|------------|
| 1 | 0 | 30 ms | 300 ms |
| 2 | 0 | 30 ms | 600 ms |
| 3 | 120 | 30 ms | 600 ms |
| 4 | 120 | 80 ms | 600 ms |
| 5 | 8 | 100 ms | 1100 ms |
| 6 | 120 | 500 ms | 1500 ms |
""")

add_md("## 1. 环境配置与库导入")

add_code("""import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['PYTENSOR_FLAGS'] = 'cxx='

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import pymc as pm
import arviz as az
import formulae
from formulae import design_matrices

from scipy.special import expit as sigmoid

print(f'PyMC: {pm.__version__}')
print(f'ArviZ: {az.__version__}')
print(f'Formulae: {formulae.__version__}')

rng = np.random.default_rng(42)
az.style.use('arviz-darkgrid')
""")

add_md("""## 2. 数据载入与预处理

### 2.1 数据载入与列映射

将原始数据映射为分析所需的列：
- **Subject**: 使用 `GroupInfo` 作为唯一被试标识（46人跨6组）
- **Group**: 实验设计空间组别（1-6）
- **Matching**: 匹配条件（Matching / NonMatching），基线=NonMatching
  - Matching: (circle, self) 或 (square, stranger) → 按键 'f'
  - NonMatching: (square, self) 或 (circle, stranger) → 按键 'j'
- **Identity**: 身份条件（Self / Stranger），基线=Stranger
- **RT_sec**: 反应时（秒）
- **ACC**: 正确率（1=正确, 0=错误）
""")

add_code("""DATA_DIR = Path(r'd:\\GitHub_programe\\GitHub\\Guassion-Process-Experiment-Design\\2_Data\\Real_Data')
FIG_DIR = Path(r'd:\\GitHub_programe\\GitHub\\Guassion-Process-Experiment-Design\\3_Figures\\ANOVA')
FIG_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(DATA_DIR / 'EXP_data_combined.csv')
print(f'原始数据: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列')
print(f'原始列名: {list(df_raw.columns)}')
""")

add_code("""df = df_raw.copy()

# 创建分析用列
df['Subject'] = df['GroupInfo'].astype(str)
df['Group'] = pd.Categorical(df['groupID'].astype(int), categories=[1,2,3,4,5,6], ordered=True)

df['Identity'] = df['Label'].map({'self': 'Self', 'stranger': 'Stranger'})
df['Identity'] = pd.Categorical(df['Identity'], categories=['Stranger', 'Self'])  # Stranger=基线

# Matching 定义: circle+self 或 square+stranger → Matching; 其他 → NonMatching
matching_mask = ((df['Shape']=='circle') & (df['Label']=='self')) | \\
                ((df['Shape']=='square') & (df['Label']=='stranger'))
df['Matching'] = np.where(matching_mask, 'Matching', 'NonMatching')
df['Matching'] = pd.Categorical(df['Matching'], categories=['NonMatching', 'Matching'])  # NonMatching=基线

df['ACC'] = df['Correct'].astype(int)
df['RT_sec'] = df['RT'].astype(float)

print(f'处理后的数据: {df.shape[0]} 行')
print(f'被试数: {df["Subject"].nunique()}')
print(f'组别数: {df["Group"].nunique()}')
print(f'\\n各组被试数:')
print(df.groupby('Group')['Subject'].nunique())
print(f'\\n各 Matching×Identity 组合试次数:')
print(df.groupby(['Matching', 'Identity']).size())
""")

add_md("""### 2.2 数据筛选

- RT 分析：仅正确试次（ACC == 1）
- ACC 分析：全部试次
- 不进行极端值剔除，由 Ex-Gaussian 分布自行应对离群值
""")

add_code("""# RT 分析数据集（仅正确试次）
df_rt = df[df['ACC'] == 1].dropna(subset=['RT_sec']).copy()
# ACC 分析数据集（全部试次）
df_acc = df.copy()

print(f'RT 分析数据: {len(df_rt)} 试次 (正确试次)')
print(f'ACC 分析数据: {len(df_acc)} 试次 (全部试次)')
print(f'\\n总体正确率: {df["ACC"].mean()*100:.1f}%')
print(f'RT 范围: {df_rt["RT_sec"].min():.3f}s - {df_rt["RT_sec"].max():.3f}s')
print(f'RT 均值: {df_rt["RT_sec"].mean():.3f}s (SD={df_rt["RT_sec"].std():.3f}s)')
""")

add_md("## 3. 探索性数据分析（EDA）")

add_code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
group_params = [
    'Group 1: P=0,T=30,W=300', 'Group 2: P=0,T=30,W=600',
    'Group 3: P=120,T=30,W=600', 'Group 4: P=120,T=80,W=600',
    'Group 5: P=8,T=100,W=1100', 'Group 6: P=120,T=500,W=1500'
]

for idx, g in enumerate([1,2,3,4,5,6]):
    ax = axes[idx//3][idx%3]
    sub = df_rt[df_rt['Group'] == g]
    for ident, color in [('Self', '#E74C3C'), ('Stranger', '#3498DB')]:
        dat = sub[(sub['Identity']==ident) & (sub['Matching']=='Matching')]['RT_sec']
        ax.hist(dat, bins=40, alpha=0.5, color=color, label=ident, density=True)
    ax.set_title(group_params[idx], fontsize=10)
    ax.set_xlabel('RT (s)')
    ax.legend(fontsize=8)

fig.suptitle('RT Distribution by Group (Matching condition, Self vs Stranger)', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / 'EDA_RT_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
""")

add_code("""# RT 描述统计
rt_summary = df_rt.groupby(['Group', 'Matching', 'Identity'])['RT_sec'].agg(
    ['mean', 'std', 'count']
).round(3)
print('RT 描述统计（秒）：')
display(rt_summary)

# ACC 描述统计
acc_summary = df.groupby(['Group', 'Matching', 'Identity'])['ACC'].agg(
    ['mean', 'count']
).round(3)
acc_summary.columns = ['ACC_mean', 'N']
print('\\nACC 描述统计：')
display(acc_summary)
""")

add_md("""## 4. RT 贝叶斯混合效应模型（Ex-Gaussian 族）

### 4.1 分布族选择理由

选择 **Ex-Gaussian 分布**（指数-高斯卷积分布）：

1. **理论合理性**: RT 建模的经典分布，由正态 + 指数卷积而成，能同时捕捉核心反应的对称变异（μ）和右尾长尾（ν）
2. **对离群值鲁棒**: 不需人为剔除极端 RT，指数尾部自然吸收
3. **可解释性**: μ 反映典型反应速度，1/ν 反映极端慢反应程度
4. **与 DDM 关系**: Ex-Gaussian 可视作 Wiener 扩散过程首通时间的一种近似

> **备注**: 原任务描述要求 R/brms 的 `exgaussian()` 族。Python/PyMC 中对应使用 `pm.ExGaussian`。Bambi 不内置支持此族，因此直接使用 PyMC 建模。
""")

add_md("### 4.2 构建设计矩阵")

add_code("""# 全模型公式: RT ~ Group * Matching * Identity
formula_full = 'RT_sec ~ Group * Matching * Identity'
dm_full = design_matrices(formula_full, data=df_rt)
X_full = np.asarray(dm_full.common)  # 固定效应设计矩阵
col_names_full = list(dm_full.common.terms.keys())

print(f'全模型设计矩阵维度: {X_full.shape}')
print(f'列名 ({len(col_names_full)}):')
for i, name in enumerate(col_names_full):
    print(f'  [{i}] {name}')
""")

add_code("""# 被试随机截距的索引
subjects_rt = df_rt['Subject'].values
subject_idx_rt, subject_labels_rt = pd.factorize(subjects_rt)
n_subjects_rt = len(subject_labels_rt)

# 简化模型公式: 去掉三因素交互
formula_reduced = 'RT_sec ~ Group*Matching + Group*Identity + Matching*Identity'
dm_reduced = design_matrices(formula_reduced, data=df_rt)
X_reduced = np.asarray(dm_reduced.common)

print(f'简化模型设计矩阵维度: {X_reduced.shape}')
print(f'被试数: {n_subjects_rt}')
""")

add_md("""### 4.3 RT 全模型（含三因素交互）

**模型规格**:
- 似然: RT ~ ExGaussian(mu, sigma, nu)
- mu = intercept + Xβ + u[subject_idx] （线性预测 + 随机截距）
- 先验: β ~ Normal(0, 10), Intercept ~ Normal(0, 10), σ ~ HalfNormal(1), ν ~ HalfNormal(1)
- 随机截距: u ~ Normal(0, σ_u), σ_u ~ HalfNormal(1)
- MCMC: 4链 × 2000迭代 (1000预热), adapt_delta=0.99, max_treedepth=15
""")

add_code("""n_pred = X_full.shape[1]

with pm.Model(coords={'subject': subject_labels_rt}) as model_rt_full:
    # 数据
    X_data = pm.MutableData('X', X_full)
    
    # 固定效应先验
    intercept = pm.Normal('Intercept', mu=0.0, sigma=10.0)
    beta = pm.Normal('beta', mu=0.0, sigma=10.0, shape=n_pred)
    
    # 随机截距
    sigma_u = pm.HalfNormal('sigma_u', sigma=1.0)
    u = pm.Normal('u', mu=0.0, sigma=sigma_u, dims='subject')
    
    # 线性预测子 (ExGaussian 的 mu 参数)
    mu = pm.Deterministic('mu', intercept + pm.math.dot(X_data, beta) + u[subject_idx_rt])
    
    # Ex-Gaussian 的 sigma 和 nu 参数
    sigma_param = pm.HalfNormal('sigma_param', sigma=1.0)
    nu_param = pm.HalfNormal('nu_param', sigma=1.0)
    
    # 似然
    y = pm.ExGaussian('y', mu=mu, sigma=sigma_param, nu=nu_param, 
                       observed=df_rt['RT_sec'].values)

print('RT 全模型构建完成')
pm.model_to_graphviz(model_rt_full)
""")

add_code("""# ⚠️ 采样耗时较长，预计 20-60 分钟（取决于 CPU）
# 若无 GPU/多核加速，可先减少 draws/tune 试用
print('开始 RT 全模型采样...')
with model_rt_full:
    idata_rt_full = pm.sample(
        draws=2000, tune=1000, chains=4,
        target_accept=0.99,
        max_treedepth=15,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}
    )
print('RT 全模型采样完成!')
""")

add_md("### 4.4 RT 简化模型（无三因素交互）")

add_code("""n_pred_red = X_reduced.shape[1]
col_names_red = list(dm_reduced.common.terms.keys())

with pm.Model(coords={'subject': subject_labels_rt}) as model_rt_reduced:
    X_data_r = pm.MutableData('X', X_reduced)
    
    intercept_r = pm.Normal('Intercept', mu=0.0, sigma=10.0)
    beta_r = pm.Normal('beta', mu=0.0, sigma=10.0, shape=n_pred_red)
    
    sigma_ur = pm.HalfNormal('sigma_u', sigma=1.0)
    ur = pm.Normal('u', mu=0.0, sigma=sigma_ur, dims='subject')
    
    mu_r = pm.Deterministic('mu', intercept_r + pm.math.dot(X_data_r, beta_r) + ur[subject_idx_rt])
    
    sigma_param_r = pm.HalfNormal('sigma_param', sigma=1.0)
    nu_param_r = pm.HalfNormal('nu_param', sigma=1.0)
    
    y_r = pm.ExGaussian('y', mu=mu_r, sigma=sigma_param_r, nu=nu_param_r,
                         observed=df_rt['RT_sec'].values)

print('RT 简化模型构建完成')

print('开始 RT 简化模型采样...')
with model_rt_reduced:
    idata_rt_reduced = pm.sample(
        draws=2000, tune=1000, chains=4,
        target_accept=0.99,
        max_treedepth=15,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}
    )
print('RT 简化模型采样完成!')
""")

add_md("""### 4.5 模型比较：三因素交互的证据强度

使用 LOO-CV 比较全模型与简化模型。ELPD 差异若 > 2×SE，说明有实质性改进。""")

add_code("""# LOO 比较
compare_rt = az.compare(
    {'Full (with 3-way)': idata_rt_full, 'Reduced (no 3-way)': idata_rt_reduced},
    ic='loo', scale='deviance'
)
print('=== RT 模型 LOO 比较 ===')
display(compare_rt)
""")

add_code("""# 手动计算 ELPD 差异及标准误
loo_full = az.loo(idata_rt_full, pointwise=True)
loo_reduced = az.loo(idata_rt_reduced, pointwise=True)

diff_elpd = float(loo_full.elpd_loo - loo_reduced.elpd_loo)
n_obs = len(loo_full.pareto_k)
se_diff = np.sqrt(n_obs * np.var(loo_full.loo_i.values - loo_reduced.loo_i.values))

print(f'ELPD 差异 (Full - Reduced): {diff_elpd:.2f}')
print(f'差异标准误 (SE): {se_diff:.2f}')
print(f'|ΔELPD| / SE = {abs(diff_elpd)/se_diff:.2f}')
print()
if diff_elpd - 2*se_diff > 0:
    print('✅ 结论: 全模型（含三因素交互）显著优于简化模型')
    print('   → 支持实验设计参数 (P, T, W) 调节 SPE')
elif diff_elpd + 2*se_diff < 0:
    print('✅ 结论: 简化模型更优 → 三因素交互证据不足')
else:
    print('⚠️ 结论: 两模型差异不显著 → 三因素交互证据不充分')
    print('   后续 DDM/GP 建模可能揭示更细致的参数化调节效应')
""")

add_md("""## 5. ACC 贝叶斯混合效应模型（Bernoulli 族）

### 5.1 ACC 全模型（含三因素交互）

**模型规格**:
- 似然: ACC ~ Bernoulli(p)
- logit(p) = intercept + Xβ + u[subject_idx]
- 先验: β ~ Normal(0, 10), Intercept ~ Normal(0, 10)
- MCMC: 同 RT 模型设置
""")

add_code("""# ACC 设计矩阵
formula_acc_full = 'ACC ~ Group * Matching * Identity'
dm_acc_full = design_matrices(formula_acc_full, data=df_acc)
X_acc_full = np.asarray(dm_acc_full.common)
col_names_acc_full = list(dm_acc_full.common.terms.keys())

subjects_acc = df_acc['Subject'].values
subject_idx_acc, subject_labels_acc = pd.factorize(subjects_acc)

print(f'ACC 全模型设计矩阵: {X_acc_full.shape}, 被试: {len(subject_labels_acc)}')
""")

add_code("""n_pred_acc = X_acc_full.shape[1]

with pm.Model(coords={'subject': subject_labels_acc}) as model_acc_full:
    X_data_acc = pm.MutableData('X', X_acc_full)
    
    intercept_a = pm.Normal('Intercept', mu=0.0, sigma=10.0)
    beta_a = pm.Normal('beta', mu=0.0, sigma=10.0, shape=n_pred_acc)
    
    sigma_ua = pm.HalfNormal('sigma_u', sigma=1.0)
    ua = pm.Normal('u', mu=0.0, sigma=sigma_ua, dims='subject')
    
    logit_p = intercept_a + pm.math.dot(X_data_acc, beta_a) + ua[subject_idx_acc]
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    
    y_a = pm.Bernoulli('y', logit_p=logit_p, observed=df_acc['ACC'].values)

print('ACC 全模型构建完成')

print('开始 ACC 全模型采样...')
with model_acc_full:
    idata_acc_full = pm.sample(
        draws=2000, tune=1000, chains=4,
        target_accept=0.99,
        max_treedepth=15,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}
    )
print('ACC 全模型采样完成!')
""")

add_md("### 5.2 ACC 简化模型（无三因素交互）")

add_code("""formula_acc_reduced = 'ACC ~ Group*Matching + Group*Identity + Matching*Identity'
dm_acc_reduced = design_matrices(formula_acc_reduced, data=df_acc)
X_acc_reduced = np.asarray(dm_acc_reduced.common)
col_names_acc_red = list(dm_acc_reduced.common.terms.keys())

with pm.Model(coords={'subject': subject_labels_acc}) as model_acc_reduced:
    X_data_acc_r = pm.MutableData('X', X_acc_reduced)
    
    intercept_ar = pm.Normal('Intercept', mu=0.0, sigma=10.0)
    beta_ar = pm.Normal('beta', mu=0.0, sigma=10.0, shape=n_pred_acc_red)
    
    sigma_uar = pm.HalfNormal('sigma_u', sigma=1.0)
    uar = pm.Normal('u', mu=0.0, sigma=sigma_uar, dims='subject')
    
    logit_p_r = intercept_ar + pm.math.dot(X_data_acc_r, beta_ar) + uar[subject_idx_acc]
    
    y_ar = pm.Bernoulli('y', logit_p=logit_p_r, observed=df_acc['ACC'].values)

print('ACC 简化模型构建完成')

print('开始 ACC 简化模型采样...')
with model_acc_reduced:
    idata_acc_reduced = pm.sample(
        draws=2000, tune=1000, chains=4,
        target_accept=0.99,
        max_treedepth=15,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}
    )
print('ACC 简化模型采样完成!')
""")

add_md("### 5.3 ACC 模型比较")

add_code("""compare_acc = az.compare(
    {'Full (with 3-way)': idata_acc_full, 'Reduced (no 3-way)': idata_acc_reduced},
    ic='loo', scale='deviance'
)
print('=== ACC 模型 LOO 比较 ===')
display(compare_acc)
""")

add_md("""## 6. 自我优势效应（SPE）点估计与可视化

SPE 定义: 在 Matching 条件下，Self 条件预期值 − Stranger 条件预期值
- **SPE_RT**: 负值 = 自我更快（典型 SPE），正值 = 陌生人更快
- **SPE_ACC**: 正值 = 自我更准（典型 SPE），负值 = 陌生人更准
""")

add_md("### 6.1 后验中计算 SPE_RT（基于 mu 参数）")

add_code("""groups = [1, 2, 3, 4, 5, 6]
identities = ['Self', 'Stranger']
matchings = ['NonMatching', 'Matching']

def make_pred_data_rt(group, identity, matching):
    '''构造预测用的单行数据框，通过 formulae 生成设计矩阵'''
    row = {'groupID': group, 'Label': identity.lower()}
    if matching == 'Matching':
        row['Shape'] = 'circle' if identity == 'Self' else 'square'
    else:
        row['Shape'] = 'square' if identity == 'Self' else 'circle'
    base = pd.DataFrame([row])
    base['Group'] = pd.Categorical(base['groupID'].astype(int), 
                                    categories=[1,2,3,4,5,6], ordered=True)
    base['Matching'] = pd.Categorical([matching], 
                                       categories=['NonMatching', 'Matching'])
    base['Identity'] = pd.Categorical([identity], 
                                       categories=['Stranger', 'Self'])
    base['Subject'] = 'group1_1'  # 占位，会在做差时消掉
    return base
""")

add_code("""# 从 RT 全模型后验中提取每个 Group×Identity×Matching 的 mu 后验
mu_estimates_rt = {}
for g in groups:
    for ident in identities:
        for match in matchings:
            pred_df = make_pred_data_rt(g, ident, match)
            dm_pred = design_matrices(formula_full, data=pred_df)
            X_pred = np.asarray(dm_pred.common)
            
            posterior = idata_rt_full.posterior
            intercept_s = posterior['Intercept'].values.flatten()
            beta_s = posterior['beta'].values.reshape(-1, posterior['beta'].shape[-1])
            
            # mu = intercept + Xβ (忽略随机截距，因为是组水平估计)
            mu_s = intercept_s + X_pred @ beta_s.T
            mu_estimates_rt[(g, ident, match)] = mu_s.flatten()

# 计算 SPE_RT = Self - Stranger (in Matching condition)
spe_rt_dict = {}
print('=== SPE_RT (Matching 条件下 Self - Stranger) ===')
print(f'{"Group":<8} {"Median(ms)":<12} {"95% HDI low(ms)":<16} {"95% HDI high(ms)":<16}')
print('-' * 52)
for g in groups:
    spe = mu_estimates_rt[(g, 'Self', 'Matching')] - mu_estimates_rt[(g, 'Stranger', 'Matching')]
    spe_rt_dict[g] = spe
    hdi = az.hdi(spe, hdi_prob=0.95)
    med = np.median(spe)
    print(f'{g:<8} {med*1000:<12.1f} {hdi[0]*1000:<16.1f} {hdi[1]*1000:<16.1f}')
""")

add_md("### 6.2 后验中计算 SPE_ACC（基于 p 参数）")

add_code("""# 从 ACC 全模型后验中计算
p_estimates_acc = {}
for g in groups:
    for ident in identities:
        for match in matchings:
            pred_df = make_pred_data_rt(g, ident, match)
            dm_pred = design_matrices(formula_acc_full, data=pred_df)
            X_pred = np.asarray(dm_pred.common)
            
            posterior_a = idata_acc_full.posterior
            intercept_a = posterior_a['Intercept'].values.flatten()
            beta_a = posterior_a['beta'].values.reshape(-1, posterior_a['beta'].shape[-1])
            
            logit_p_s = intercept_a + X_pred @ beta_a.T
            p_s = sigmoid(logit_p_s).flatten()
            p_estimates_acc[(g, ident, match)] = p_s

spe_acc_dict = {}
print('=== SPE_ACC (Matching 条件下 Self - Stranger) ===')
print(f'{"Group":<8} {"Median":<12} {"95% HDI low":<16} {"95% HDI high":<16}')
print('-' * 52)
for g in groups:
    spe = p_estimates_acc[(g, 'Self', 'Matching')] - p_estimates_acc[(g, 'Stranger', 'Matching')]
    spe_acc_dict[g] = spe
    hdi = az.hdi(spe, hdi_prob=0.95)
    med = np.median(spe)
    print(f'{g:<8} {med:<12.3f} {hdi[0]:<16.3f} {hdi[1]:<16.3f}')
""")

add_md("### 6.3 可视化：SPE_RT 和 SPE_ACC 点估计 + 95% HDI")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

group_labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
group_param_labels = [
    'G1\\nP=0,T=30\\nW=300', 'G2\\nP=0,T=30\\nW=600',
    'G3\\nP=120,T=30\\nW=600', 'G4\\nP=120,T=80\\nW=600',
    'G5\\nP=8,T=100\\nW=1100', 'G6\\nP=120,T=500\\nW=1500'
]

# --- SPE_RT 图 ---
ax = axes[0]
x_pos = np.arange(len(groups))
medians_rt, hdi_low_rt, hdi_high_rt = [], [], []
for g in groups:
    spe = spe_rt_dict[g] * 1000  # 转换为 ms
    hdi = az.hdi(spe, hdi_prob=0.95)
    medians_rt.append(np.median(spe))
    hdi_low_rt.append(hdi[0])
    hdi_high_rt.append(hdi[1])

errors_low = [m - l for m, l in zip(medians_rt, hdi_low_rt)]
errors_high = [h - m for m, h in zip(medians_rt, hdi_high_rt)]

ax.errorbar(x_pos, medians_rt, yerr=[errors_low, errors_high],
            fmt='o', capsize=8, capthick=2, color='#2C3E50', markersize=10,
            markeredgecolor='#2C3E50', markerfacecolor='#E74C3C', 
            elinewidth=2, zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(group_param_labels, fontsize=8)
ax.set_ylabel('SPE (ms)', fontsize=12)
ax.set_title('SPE_RT: Self − Stranger RT in Matching Condition', fontsize=13, fontweight='bold')
ax.set_xlabel('Experimental Design Group', fontsize=12)

# 标注
for i, (m, l, h) in enumerate(zip(medians_rt, hdi_low_rt, hdi_high_rt)):
    sig = '★' if (l > 0 or h < 0) else ''  # HDI 不跨零 = 显著
    ax.annotate(f'{m:.0f}{sig}ms', (i, m), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=9)

ax.grid(True, alpha=0.3)

# --- SPE_ACC 图 ---
ax = axes[1]
medians_acc, hdi_low_acc, hdi_high_acc = [], [], []
for g in groups:
    spe = spe_acc_dict[g]
    hdi = az.hdi(spe, hdi_prob=0.95)
    medians_acc.append(np.median(spe))
    hdi_low_acc.append(hdi[0])
    hdi_high_acc.append(hdi[1])

errors_low_a = [m - l for m, l in zip(medians_acc, hdi_low_acc)]
errors_high_a = [h - m for m, h in zip(medians_acc, hdi_high_acc)]

ax.errorbar(x_pos, medians_acc, yerr=[errors_low_a, errors_high_a],
            fmt='o', capsize=8, capthick=2, color='#2C3E50', markersize=10,
            markeredgecolor='#2C3E50', markerfacecolor='#3498DB',
            elinewidth=2, zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(group_param_labels, fontsize=8)
ax.set_ylabel('SPE (Accuracy Difference)', fontsize=12)
ax.set_title('SPE_ACC: Self − Stranger Accuracy in Matching Condition', fontsize=13, fontweight='bold')
ax.set_xlabel('Experimental Design Group', fontsize=12)

for i, (m, l, h) in enumerate(zip(medians_acc, hdi_low_acc, hdi_high_acc)):
    sig = '★' if (l > 0 or h < 0) else ''
    ax.annotate(f'{m:.3f}{sig}', (i, m), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=9)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'SPE_Group_Comparison.png', dpi=200, bbox_inches='tight')
plt.show()
print('SPE 图已保存到:', FIG_DIR / 'SPE_Group_Comparison.png')
""")

add_md("""## 7. 模型诊断

检查收敛性指标:
- **R̂** (Gelman-Rubin): 所有参数应 < 1.01
- **Bulk ESS**: 有效样本数应 > 1000
- **Tail ESS**: 尾部有效样本数应 > 1000
""")

add_code("""print('=== RT 全模型诊断 ===')
summary_rt = az.summary(idata_rt_full, var_names=['Intercept', 'beta', 'sigma_u', 'sigma_param', 'nu_param'])
display(summary_rt)

rhat_max = summary_rt['r_hat'].max()
ess_bulk_min = summary_rt['ess_bulk'].min()
ess_tail_min = summary_rt['ess_tail'].min()

print(f'\\n最大 R̂: {rhat_max:.4f} {\"✅\" if rhat_max < 1.01 else \"❌ 收敛问题!\"}')
print(f'最小 Bulk ESS: {ess_bulk_min:.0f} {\"✅\" if ess_bulk_min > 1000 else \"⚠️ 偏低\"}')
print(f'最小 Tail ESS: {ess_tail_min:.0f} {\"✅\" if ess_tail_min > 1000 else \"⚠️ 偏低\"}')
""")

add_code("""print('=== ACC 全模型诊断 ===')
summary_acc = az.summary(idata_acc_full, var_names=['Intercept', 'beta', 'sigma_u'])
display(summary_acc)

rhat_max_a = summary_acc['r_hat'].max()
ess_bulk_min_a = summary_acc['ess_bulk'].min()
ess_tail_min_a = summary_acc['ess_tail'].min()

print(f'\\n最大 R̂: {rhat_max_a:.4f} {\"✅\" if rhat_max_a < 1.01 else \"❌ 收敛问题!\"}')
print(f'最小 Bulk ESS: {ess_bulk_min_a:.0f} {\"✅\" if ess_bulk_min_a > 1000 else \"⚠️ 偏低\"}')
print(f'最小 Tail ESS: {ess_tail_min_a:.0f} {\"✅\" if ess_tail_min_a > 1000 else \"⚠️ 偏低\"}')
""")

add_code("""# 后验迹图 (Trace Plot)
az.plot_trace(idata_rt_full, var_names=['Intercept', 'sigma_param', 'nu_param', 'sigma_u'],
              figsize=(12, 10))
plt.tight_layout()
plt.savefig(FIG_DIR / 'RT_Trace_Plot.png', dpi=150, bbox_inches='tight')
plt.show()
""")

add_md("""## 8. 结果报告与初步解读

### 8.1 关键发现汇总""")

add_code("""print('=' * 60)
print('阶段一：行为现象确认 — 分析结果汇总')
print('=' * 60)

print(f'\\n【数据概况】')
print(f'  被试总数: {df[\"Subject\"].nunique()} (跨 {df[\"Group\"].nunique()} 组)')
print(f'  总试次: {len(df)} (正确: {df[\"ACC\"].sum()}, 错误: {(df[\"ACC\"]==0).sum()})')

print(f'\\n【RT 模型比较 (Ex-Gaussian)】')
print(f'  全模型 (含三因素交互) vs 简化模型 (无三因素交互)')
print(f'  ELPD 差异: {diff_elpd:.2f} (SE={se_diff:.2f})')
if diff_elpd - 2*se_diff > 0:
    print(f'  → 结论: 全模型显著更优，支持 Group×Matching×Identity 三因素交互')
    print(f'  → 解读: 实验设计参数 (P, T, W) 调节了自我优势效应 (SPE)')
elif diff_elpd + 2*se_diff < 0:
    print(f'  → 结论: 简化模型更优，不支持三因素交互')
else:
    print(f'  → 结论: 证据不足以确定三因素交互的存在')
    print(f'  → 建议: DDM/GP 模型可能更敏感地检测参数化调节效应')

print(f'\\n【ACC 模型比较 (Bernoulli)】')
compare_acc_df = az.compare(
    {'Full': idata_acc_full, 'Reduced': idata_acc_reduced},
    ic='loo', scale='deviance'
)
print(f'  ELPD 差异: {compare_acc_df.loc[\"Full\", \"elpd_loo\"] - compare_acc_df.loc[\"Reduced\", \"elpd_loo\"]:.2f}')
""")

add_md("""### 8.2 初步解读方向

1. **若三因素交互的 BF/LOO 证据 > 10（或 ELPD 差异 > 2×SE）**:
   - 实验设计参数（P, T, W）确实调节了自我优势效应
   - 从图中可见哪些组别的 SPE 最强/最弱
   - 这为后续 DDM 建模提供了理论依据：参数需要在不同设计空间下取值不同

2. **若证据不足**:
   - 可能原因：组间变异较大、被试量不足（每组仅7-10人）、SPE 效应本身较小
   - 后续 DDM 分析的价值：DDM 将 RT 和 ACC 联合建模，可能比单独分析更敏感
   - GP 建模可能发现非线性的参数-SPE 关系

3. **SPE 图解读**:
   - 负的 SPE_RT = 自我更快（典型 SPE），注意 y=0 参考线
   - 正的 SPE_ACC = 自我更准
   - 不同组间 SPE 大小的差异（即使交互不显著）仍可为后续 GP 建模提供先验信息
""")

add_md("""## 9. 补充：传统 ANOVA 对比（可选）

为与贝叶斯结果对比，提供传统 RM-ANOVA 分析。""")

add_code("""# 计算每个被试在每个条件下的平均 RT (仅 Matching 条件)
df_agg = df_rt[df_rt['Matching'] == 'Matching'].groupby(
    ['Subject', 'Group', 'Identity']
)['RT_sec'].mean().reset_index()

# pivot 为宽格式
df_wide = df_agg.pivot_table(
    index=['Subject', 'Group'], columns='Identity', values='RT_sec'
).reset_index()
df_wide['SPE'] = df_wide['Self'] - df_wide['Stranger']

print('各组 SPE 描述统计 (传统方法):')
print(df_wide.groupby('Group')['SPE'].agg(['mean', 'std', 'count']).round(4))
print(f'\\n总体 SPE: {df_wide[\"SPE\"].mean()*1000:.1f} ms')
""")

add_md("""---

## 附录：文件输出

所有图表已保存至 `3_Figures/ANOVA/`:
- `EDA_RT_distribution.png` — RT 分布探索图
- `SPE_Group_Comparison.png` — SPE 点估计 + 95% HDI
- `RT_Trace_Plot.png` — MCMC 迹图

模型对象（idata）可在 Notebook 中直接使用 `az.plot_*` 系列函数进一步探索。
""")

nb.cells = cells

notebook_path = Path(r'd:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\ANOVA\SPE_Bayesian_Analysis.ipynb')
nbf.write(nb, notebook_path)
print(f'Notebook created: {notebook_path}')
print(f'Cells: {len(cells)}')
