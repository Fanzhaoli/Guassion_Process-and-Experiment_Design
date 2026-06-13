#!/usr/bin/env python3
"""
================================================================================
CRF (Conditional Response Function) 图 — 基于 Stim Coding z-bias 仿真
================================================================================

实验范式: Self Matching Task (Sui, 2012)
仿真原理: Stim Coding — 通过操纵 DDM 起始点参数 z 模拟匹配键偏好
CRF 逻辑: 按反应时(RT)分位数分箱，每箱计算 P(Matching)，刻画决策倾向的动态变化

参考代码:
  - ddm_stim_coding/Response_Bias/simulation/data_generation.py  (z-bias 数据生成)
  - ddm_stim_coding/plot_simulation_CRF.ipynb                     (CRF 绘图逻辑)
  - Visualization/V4/spe_reaction_window_analysis.py               (CRF-SPE 分析)

输出:
  - 仿真数据 -> 2_Data/Generate_Data/HDDM_Stim-Coding_Simulation/
  - 图表     -> 3_Figures/HDDM_Stim-Coding_Simulation/
================================================================================
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 0. 路径配置 (兼容 Docker 和本地)
# ============================================================================
try:
    BASE_DIR = Path("/home/jovyan/work")
    if not BASE_DIR.exists():
        raise FileNotFoundError
except (FileNotFoundError, OSError):
    BASE_DIR = Path(r"D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")

CODE_DIR  = BASE_DIR / "1_Code" / "Python_for_Check" / "HDDM_Stim-Coding_Simulation"
DATA_DIR  = BASE_DIR / "2_Data" / "Generate_Data" / "HDDM_Stim-Coding_Simulation"
FIG_DIR   = BASE_DIR / "3_Figures" / "HDDM_Stim-Coding_Simulation"

for d in [CODE_DIR, DATA_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"FIG_DIR:  {FIG_DIR}")

# ============================================================================
# 1. 数据生成 — Stim Coding z-bias 仿真
# ============================================================================
# 核心原理 (Stim Coding):
#   刺激A (stimulus=1): params = {a, v: v+dc, t, z,      ...}
#   刺激B (stimulus=0): params = {a, v: v-dc, t, z: 1-z, ...}
#   - 两个刺激的 v 符号相反、z 互为镜像，实现物理-心理坐标转换
#   - z > 0.5 => 起始点靠近"匹配"边界 => 更多匹配响应
#
#   choice = response         (stimulus == 1)
#   choice = 1 - response     (stimulus == 0)
#   choice == 1 => 被试按了"匹配"键

try:
    import hddm
    HAS_HDDM = True
    print(f"HDDM version: {hddm.__version__}")
except ImportError:
    HAS_HDDM = False
    print("WARNING: HDDM not installed. Using direct DDM simulation (Wiener process).")


# ---------------------------------------------------------------------------
# 直接 Wiener 扩散过程仿真 (当 HDDM 不可用时作为回退)
# ---------------------------------------------------------------------------
def simulate_ddm_trial(a, v, t, z, dc=0.0, dt=0.001, max_steps=10000):
    """
    用 Euler-Maruyama 方法模拟单个 DDM 试次。

    参数:
        a:  边界分离
        v:  漂移率
        t:  非决策时间 (秒)
        z:  起始点 (0-1 比例)
        dc: 漂移准则 (刺激相关偏差)
        dt: 时间步长
    """
    x = z * a  # 起始位置
    step = 0
    while 0 < x < a and step < max_steps:
        x += v * dt + np.sqrt(dt) * np.random.randn()
        step += 1
    rt_decision = step * dt
    response = 1 if x >= a else 0
    rt_total = rt_decision + t
    return rt_total, response


def generate_simulated_trials_with_z_bias(
    n_subjects=30,
    trials_per_condition=150,
    z_levels=None,
    a_mean=1.2,
    a_std=0.2,
    v_mean=1.0,
    v_std=0.3,
    t_mean=0.30,
    t_std=0.05,
    dc_mean=0.0,
    dc_std=0.05,
    seed_base=420,
):
    """
    生成包含 z-bias 操控的仿真数据 (Stim Coding 原理)。

    设计:
      - z_levels: 不同起始点水平, e.g. [0.50, 0.55, 0.60, 0.65]
      - z=0.50: 无偏 (neutral baseline)
      - z>0.50: 偏向匹配键 (模拟 Self 优势效应)
      - 每个刺激 (stimulus 0/1) 各 half_trials 试次
    """
    if z_levels is None:
        z_levels = [0.50, 0.55, 0.60, 0.65]

    all_trials = []
    subj_params_list = []

    for subj_id in range(n_subjects):
        subj_seed = seed_base + subj_id * 1000
        np.random.seed(subj_seed)
        random.seed(subj_seed)

        # 从组分布采样个体参数
        subj_a = max(0.4, np.random.normal(a_mean, a_std))
        subj_t = max(0.1,  np.random.normal(t_mean, t_std))
        subj_v = np.random.normal(v_mean, v_std)
        subj_dc = np.random.normal(dc_mean, dc_std)

        subj_params_list.append({
            'subj_idx': subj_id,
            'a': subj_a,
            'v': subj_v,
            't': subj_t,
            'dc': subj_dc,
        })

        half = trials_per_condition // 2

        for cond_label, z_val in [('neutral', 0.50),
                                   ('z_bias_small', 0.55),
                                   ('z_bias_medium', 0.60),
                                   ('z_bias_large', 0.65)]:
            # z 添加个体噪声
            z_subj = np.clip(z_val + np.random.normal(0, 0.02), 0.3, 0.7)

            for stimulus in [1, 0]:
                if HAS_HDDM:
                    # 使用 HDDM 内置生成器
                    if stimulus == 1:
                        params = {
                            'a': subj_a, 'v': subj_v + subj_dc,
                            't': subj_t, 'z': z_subj,
                            'sv': 0, 'sz': 0, 'st': 0
                        }
                    else:
                        params = {
                            'a': subj_a, 'v': subj_v - subj_dc,
                            't': subj_t, 'z': 1 - z_subj,
                            'sv': 0, 'sz': 0, 'st': 0
                        }
                    df_sim, _ = hddm.generate.gen_rand_data(
                        params=params, size=half, subjs=1,
                        subj_noise=0, seed=subj_seed + stimulus
                    )
                    rts = df_sim['rt'].values
                    responses = df_sim['response'].values.astype(int)
                else:
                    # 直接 Wiener 仿真
                    rts = []
                    responses = []
                    for _ in range(half):
                        if stimulus == 1:
                            rt, resp = simulate_ddm_trial(
                                subj_a, subj_v + subj_dc, subj_t, z_subj
                            )
                        else:
                            rt, resp = simulate_ddm_trial(
                                subj_a, subj_v - subj_dc, subj_t, 1 - z_subj
                            )
                        rts.append(rt)
                        responses.append(resp)
                    rts = np.array(rts)
                    responses = np.array(responses, dtype=int)

                # Stim Coding 坐标转换: choice = 匹配键选择
                # stimulus=1: response=1 => 匹配; stimulus=0: response=0 => 匹配
                if stimulus == 1:
                    choices = responses  # 1=匹配, 0=不匹配
                else:
                    choices = 1 - responses  # 翻转: 0->1(匹配), 1->0(不匹配)

                for i in range(half):
                    all_trials.append({
                        'subj_idx': subj_id,
                        'condition': cond_label,
                        'z_level': z_val,
                        'z_subj': z_subj,
                        'stimulus': stimulus,
                        'rt': rts[i],
                        'response_raw': int(responses[i]),
                        'choice': int(choices[i]),  # 1=匹配, 0=不匹配
                    })

    data = pd.DataFrame(all_trials)
    subj_params = pd.DataFrame(subj_params_list)
    return data, subj_params


# ============================================================================
# 2. CRF 计算 (Conditional Response Function)
# ============================================================================
def compute_crf(data, n_quantiles=5, condition_col='condition'):
    """
    计算条件响应函数 (CRF)。

    原理:
      1. 按 RT 升序排列所有试次
      2. 等分为 n_quantiles 个分箱 (bin)
      3. 每箱内计算:
         - 平均 RT
         - P(Matching) = choice==1 的试次占比
         - 标准误 (用于 95% CI)

    复刻自:
      - V4/spe_reaction_window_analysis.py: compute_crf()
      - ddm_stim_coding/plot_simulation_CRF.ipynb
    """
    results = []
    for cond in sorted(data[condition_col].unique()):
        cond_data = data[data[condition_col] == cond].copy()
        if len(cond_data) < n_quantiles * 2:
            continue

        cond_data = cond_data.sort_values('rt')
        n = len(cond_data)
        q_size = n // n_quantiles

        for q in range(n_quantiles):
            start = q * q_size
            end = n if q == n_quantiles - 1 else start + q_size
            bin_data = cond_data.iloc[start:end]

            rt_mean = bin_data['rt'].mean()
            p_match = bin_data['choice'].mean()
            n_bin = len(bin_data)

            # 二项分布标准误
            se = np.sqrt(p_match * (1 - p_match) / n_bin) if n_bin > 1 else 0

            results.append({
                'condition': cond,
                'bin': q + 1,
                'n': n_bin,
                'rt_mean_ms': rt_mean * 1000,
                'p_matching': p_match,
                'se': se,
                'ci_lo': max(0, p_match - 1.96 * se),
                'ci_hi': min(1, p_match + 1.96 * se),
            })

    return pd.DataFrame(results)


# ============================================================================
# 3. 可视化
# ============================================================================
def plot_crf_zbias(crf_df, fig_dir, data=None):
    """
    绘制 CRF 图: P(Matching) vs RT，按 z-bias 水平分线。

    图表规范:
      - 横坐标: RT (ms)
      - 纵坐标: P(Matching) — 选择匹配按键的概率
      - 不同颜色/线型对应不同 z-bias 水平
      - 误差棒: 95% CI
      - 水平虚线: P = 0.5 (无偏基线)
    """
    # --- 配色与样式 ---
    condition_config = {
        'neutral':        {'label': 'Neutral (z=0.50)',        'color': '#757575', 'marker': 'o', 'ls': '-'},
        'z_bias_small':   {'label': 'Small Bias (z=0.55)',    'color': '#ff9800', 'marker': 's', 'ls': '--'},
        'z_bias_medium':  {'label': 'Medium Bias (z=0.60)',   'color': '#e91e63', 'marker': '^', 'ls': '-.'},
        'z_bias_large':   {'label': 'Large Bias (z=0.65)',    'color': '#9c27b0', 'marker': 'D', 'ls': ':'},
    }

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    # -------- 图 1: 主 CRF 图 --------
    fig, ax = plt.subplots(figsize=(10, 7))

    for cond, cfg in condition_config.items():
        cdf = crf_df[crf_df['condition'] == cond]
        if len(cdf) == 0:
            continue
        ax.errorbar(
            cdf['rt_mean_ms'], cdf['p_matching'],
            yerr=[cdf['p_matching'] - cdf['ci_lo'], cdf['ci_hi'] - cdf['p_matching']],
            marker=cfg['marker'], linestyle=cfg['ls'], color=cfg['color'],
            linewidth=2, markersize=9, capsize=4, capthick=1.5,
            label=cfg['label'], alpha=0.9
        )

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
               label='P=0.5 (unbiased)')
    ax.set_xlabel('Reaction Time (ms)')
    ax.set_ylabel('P(Matching)')
    ax.set_title('Conditional Response Function (CRF)\n'
                 'Self Matching Task — Stim Coding z-bias Simulation',
                 fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    plt.tight_layout()
    path_main = fig_dir / 'figure_01_CRF_zbias_main.png'
    fig.savefig(path_main, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path_main}")

    # -------- 图 2: RT 分布 + CRF 叠加 --------
    if data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左: RT 分布
        ax = axes[0]
        colors_dist = {
            'neutral':       '#bdbdbd',
            'z_bias_small':  '#ffcc80',
            'z_bias_medium': '#f48fb1',
            'z_bias_large':  '#ce93d8',
        }
        for cond, color in colors_dist.items():
            cdf_rt = data[data['condition'] == cond]['rt'] * 1000
            ax.hist(cdf_rt, bins=40, alpha=0.4, color=color,
                    label=condition_config[cond]['label'], density=True)
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('Density')
        ax.set_title('RT Distribution by z-bias Level', fontweight='bold')
        ax.legend(fontsize=8)

        # 右: P(Matching) 饼图 (中位数 RT 处)
        ax = axes[1]
        median_rt_data = []
        for cond, cfg in condition_config.items():
            cdf = crf_df[crf_df['condition'] == cond]
            if len(cdf) > 0:
                median_bin = cdf.iloc[len(cdf) // 2]  # 取中位 bin
                median_rt_data.append({
                    'condition': cfg['label'],
                    'rt_ms': median_bin['rt_mean_ms'],
                    'p_matching': median_bin['p_matching'],
                    'color': cfg['color'],
                })
        med_df = pd.DataFrame(median_rt_data)
        bars = ax.barh(
            med_df['condition'], med_df['p_matching'],
            color=med_df['color'], alpha=0.8, edgecolor='black', linewidth=0.8
        )
        for bar, p in zip(bars, med_df['p_matching']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{p:.3f}', va='center', fontsize=10, fontweight='bold')
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('P(Matching) at Median RT')
        ax.set_title('Matching Probability by z-bias Level\n(at median RT bin)',
                     fontweight='bold')
        ax.set_xlim(0, 1.05)

        plt.tight_layout()
        path_dist = fig_dir / 'figure_02_CRF_RT_distribution.png'
        fig.savefig(path_dist, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path_dist}")

    # -------- 图 3: SPE (Self-Preference Effect) 作为 z 的函数 --------
    fig, ax = plt.subplots(figsize=(8, 6))

    summary = crf_df.groupby('condition').agg(
        mean_p=('p_matching', 'mean'),
        se_p=('p_matching', lambda x: np.std(x) / np.sqrt(len(x))),
        mean_rt=('rt_mean_ms', 'mean'),
    ).reset_index()

    # 计算 SPE = P_matching(condition) - P_matching(neutral)
    neutral_p = summary.loc[summary['condition'] == 'neutral', 'mean_p'].values
    if len(neutral_p) > 0:
        neutral_p = neutral_p[0]
        summary['SPE'] = summary['mean_p'] - neutral_p
        bias_conds = summary[summary['condition'] != 'neutral']

        z_values = [0.55, 0.60, 0.65]
        colors_spe = ['#ff9800', '#e91e63', '#9c27b0']
        labels_spe = ['z=0.55', 'z=0.60', 'z=0.65']

        for z, color, label, (_, row) in zip(
            z_values, colors_spe, labels_spe,
            bias_conds.iterrows()
        ):
            ax.bar(label, row['SPE'], color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.8)
            ax.text(label, row['SPE'] + 0.005, f"{row['SPE']:.3f}",
                    ha='center', fontsize=11, fontweight='bold')

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6)
        ax.set_ylabel('SPE = P(Matching) - P(Matching|Neutral)')
        ax.set_title('Self-Preference Effect (SPE) by z-bias Level',
                     fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        path_spe = fig_dir / 'figure_03_SPE_by_zbias.png'
        fig.savefig(path_spe, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path_spe}")

    return path_main


# ============================================================================
# 4. 主流程
# ============================================================================
def main():
    print("=" * 70)
    print("  CRF Analysis — Stim Coding z-bias Simulation")
    print("  Self Matching Task (Sui, 2012)")
    print("=" * 70)
    print()

    # ---- 4a. 生成仿真数据 ----
    print("[1/4] Generating simulated data with z-bias manipulation...")
    print(f"      Using {'HDDM' if HAS_HDDM else 'Direct Wiener'} simulator")

    data, subj_params = generate_simulated_trials_with_z_bias(
        n_subjects=30,
        trials_per_condition=150,
        z_levels=[0.50, 0.55, 0.60, 0.65],
        a_mean=1.2,
        v_mean=1.0,
        t_mean=0.30,
        seed_base=420,
    )

    print(f"      Generated {len(data)} trials")
    print(f"      Subjects: {data['subj_idx'].nunique()}")
    print(f"      Conditions: {sorted(data['condition'].unique())}")
    print(f"      Mean RT: {data['rt'].mean()*1000:.1f} ms")

    # ---- 4b. 保存仿真数据 ----
    print("\n[2/4] Saving simulation data...")
    data_path = DATA_DIR / "simulation_zbias_crf_data.csv"
    params_path = DATA_DIR / "simulation_zbias_subject_params.csv"
    data.to_csv(data_path, index=False)
    subj_params.to_csv(params_path, index=False)
    print(f"      Data: {data_path}")
    print(f"      Params: {params_path}")

    # ---- 4c. 计算 CRF ----
    print("\n[3/4] Computing CRF (Conditional Response Function)...")
    crf_df = compute_crf(data, n_quantiles=5, condition_col='condition')
    print(f"      CRF table shape: {crf_df.shape}")
    print(f"      Bins: {crf_df['bin'].nunique()}, Conditions: {crf_df['condition'].nunique()}")

    # 打印 CRF 摘要
    print("\n      --- CRF Summary ---")
    for cond in sorted(crf_df['condition'].unique()):
        cdf = crf_df[crf_df['condition'] == cond]
        print(f"      {cond:20s}: "
              f"P(Match) mean={cdf['p_matching'].mean():.3f}, "
              f"RT range=[{cdf['rt_mean_ms'].min():.0f}, {cdf['rt_mean_ms'].max():.0f}] ms")

    # 保存 CRF 数据
    crf_path = DATA_DIR / "crf_results_zbias.csv"
    crf_df.to_csv(crf_path, index=False)
    print(f"\n      CRF data saved: {crf_path}")

    # ---- 4d. 绘图 ----
    print("\n[4/4] Generating visualizations...")
    plot_crf_zbias(crf_df, FIG_DIR, data=data)

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print(f"  Data:  {DATA_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
