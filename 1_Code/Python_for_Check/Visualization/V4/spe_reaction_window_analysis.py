#!/usr/bin/env python3
"""
SPE (Self-Preference Effect) 反应时窗口分析脚本
================================================
目的: 系统分析不同反应时窗口下 SPE 的表现特征，精确确定产生显著 SPE 的反应时阈值范围。

理论基础:
- SPE = P_Match(Self) - P_Match(Stranger)
- 根据自我优势效应(SPE)双重机制假设:
  1) z-bias: 起始点偏向 Self 匹配键 (全局效应)
  2) v-bias: 特定反应窗口内 Self 条件 drift rate 增强 (初步推测 0.3-0.7 秒)

分析方法:
- 滑动窗口法: 在不同 RT 滑动窗口内计算 SPE 及 95% CI
- 条件响应函数 (CRF): Q=5 分位数分箱法
- 跨组别比较: 8 个实验组别的 SPE 模式对比
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 0. 路径配置
# ============================================================================
PROJECT_ROOT = Path(r"D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")
RAW_DIR = PROJECT_ROOT / "2_Data" / "Real_Data" / "UnExtact" / "raw"
V4_DIR = PROJECT_ROOT / "1_Code" / "Python_for_Check" / "Visualization" / "V4"
OUT_DIR = V4_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验条件常量
CONDITIONS = {
    1: {"P": 0,   "T": 0.03, "W": 0.3,  "label": "G1 | P0_T30_W300"},
    2: {"P": 0,   "T": 0.03, "W": 0.6,  "label": "G2 | P0_T30_W600"},
    3: {"P": 120, "T": 0.03, "W": 0.6,  "label": "G3 | P120_T30_W600"},
    4: {"P": 120, "T": 0.08, "W": 0.6,  "label": "G4 | P120_T80_W600"},
    5: {"P": 8,   "T": 0.10, "W": 1.1,  "label": "G5 | P8_T100_W1100"},
    6: {"P": 120, "T": 0.50, "W": 1.5,  "label": "G6 | P120_T500_W1500"},
    7: {"P": 0,   "T": 0.10, "W": 1.1,  "label": "G7 | P0_T100_W1100"},
    8: {"P": 120, "T": 0.03, "W": 0.8,  "label": "G8 | P120_T30_W800"},
}

# 组别质量分类
QUALITY_MAP = {1: "exclude", 2: "exclude", 3: "caution",
               4: "good", 5: "good", 6: "good",
               7: "good", 8: "good"}

# 排除组别 (质量低)
EXCLUDE_GROUPS = {1, 2}

# ============================================================================
# 1. 实验规则 (100% 复现 app_server.py 及 R 脚本)
# ============================================================================
def get_match_key(subject_id):
    """获取被试的匹配键: 基于 subjectID 循环分配 f/j"""
    return ['f', 'j', 'j', 'f'][((subject_id - 1) % 4)]

def get_correct_order(subject_id):
    """获取被试的形状-标签正确对应关系"""
    if subject_id % 2 == 0:
        return {"square": "self", "circle": "stranger"}
    else:
        return {"square": "stranger", "circle": "self"}

def compute_condition(shape, label, subject_id):
    """计算试次条件: Matching / NonMatching"""
    expected_label = get_correct_order(subject_id)[shape]
    return "Matching" if label == expected_label else "NonMatching"

# ============================================================================
# 2. 数据加载
# ============================================================================
def load_all_data(raw_dir=RAW_DIR):
    """加载所有原始数据文件"""
    files = sorted(raw_dir.glob("EXP_data_group*.csv"))
    print(f"Loading {len(files)} data files...")
    
    all_dfs = []
    for fp in files:
        df = pd.read_csv(fp, na_values=["NA", "nan", "NaN", ""])
        fname = fp.stem  # e.g., "EXP_data_group1_1"
        parts = fname.replace("EXP_data_group", "").split("_")
        gid_from_file = int(parts[0])
        sid_from_file = int(parts[1])
        
        df['Shape'] = df['Shape'].str.strip().str.lower()
        df['Label'] = df['Label'].str.strip().str.lower()
        df['Response'] = df['Response'].str.strip().str.lower()
        df['CorrectKey'] = df['CorrectKey'].str.strip().str.lower()
        
        # 处理 stage
        df['stage'] = df['stage'].fillna("formal")
        df['stage'] = df['stage'].replace("", "formal")
        
        # RT 转换
        df['RT'] = pd.to_numeric(df['RT'], errors='coerce')
        df['RT_ms'] = df['RT'] * 1000
        df['Correct'] = pd.to_numeric(df['Correct'], errors='coerce')
        
        # 判断是否响应
        df['responded'] = df['RT'].notna() & df['Response'].notna() & \
                          ~df['Response'].isin(['na', 'nan', ''])
        
        # 计算 Condition
        sid_ref = sid_from_file
        df['Condition'] = df.apply(
            lambda row: compute_condition(row['Shape'], row['Label'], sid_ref), axis=1
        )
        
        # Identity
        df['Identity'] = df['Label'].apply(lambda x: 'Self' if x == 'self' else 'Stranger')
        
        # Match Key
        match_key = get_match_key(sid_ref)
        df['MatchKey'] = match_key
        df['ResponseIsMatch'] = np.where(
            df['responded'],
            (df['Response'] == match_key).astype(int),
            np.nan
        )
        
        # 附加信息
        df['T_ms'] = df['T'] * 1000
        df['W_ms'] = df['W'] * 1000
        df['M_ms'] = df['T_ms'] + df['W_ms']
        df['source_file'] = fname
        all_dfs.append(df)
    
    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(all_df)} trials from {len(files)} files")
    return all_df

# ============================================================================
# 3. CRF 计算 (Conditional Response Function)
# ============================================================================
def compute_crf(trials, n_quantiles=5):
    """
    计算单个条件下的 CRF (Conditional Response Function)
    逻辑: 按 RT 升序排列 -> 等分 n_quantiles 份 -> 每份计算 P(MatchKey)
    100% 复现 R 脚本和 JS 前端逻辑
    """
    d = trials[
        (trials['stage'] == 'formal') &
        (trials['responded']) &
        (trials['RT'].notna())
    ].copy()
    
    if len(d) < n_quantiles * 2:
        return pd.DataFrame()
    
    d = d.sort_values('RT')
    n_total = len(d)
    q_size = n_total // n_quantiles
    
    bins = []
    for i in range(n_quantiles):
        start = i * q_size
        end = n_total if i == n_quantiles - 1 else start + q_size
        b = d.iloc[start:end]
        p = b['ResponseIsMatch'].mean()
        n_b = len(b)
        sd_val = b['ResponseIsMatch'].std(ddof=1) if n_b > 1 else 0
        bins.append({
            'bin': i + 1,
            'n': n_b,
            'rt_mean': b['RT'].mean(),
            'rt_mean_ms': b['RT_ms'].mean(),
            'upper_prop': p,
            'sem': sd_val / np.sqrt(n_b) if n_b > 1 else 0
        })
    
    return pd.DataFrame(bins)

def compute_spe_crf(trials, n_quantiles=5):
    """计算 CRF-SPE: Self vs Stranger 的 CRF 差异"""
    crf_s = compute_crf(trials[trials['Identity'] == 'Self'], n_quantiles)
    crf_st = compute_crf(trials[trials['Identity'] == 'Stranger'], n_quantiles)
    
    m = min(len(crf_s), len(crf_st))
    if m == 0:
        return crf_s, crf_st, pd.DataFrame()
    
    spe_curve = pd.DataFrame({
        'bin': range(1, m + 1),
        'rt_mean_ms': (crf_s['rt_mean_ms'].iloc[:m].values + crf_st['rt_mean_ms'].iloc[:m].values) / 2,
        'spe_upper_prop': crf_s['upper_prop'].iloc[:m].values - crf_st['upper_prop'].iloc[:m].values,
        'spe_sem': np.sqrt(crf_s['sem'].iloc[:m].values**2 + crf_st['sem'].iloc[:m].values**2)
    })
    return crf_s, crf_st, spe_curve

# ============================================================================
# 4. 反应时滑动窗口 SPE 分析
# ============================================================================
def sliding_window_spe(trials, window_width_ms=200, step_ms=50, min_trials_per_window=20):
    """
    滑动窗口法分析 SPE 随 RT 的变化
    在每个 RT 窗口内独立计算 SPE 及其显著性
    
    参数:
        window_width_ms: 窗口宽度 (ms)
        step_ms: 滑动步长 (ms)
        min_trials_per_window: 每个窗口最少试次数
    """
    d = trials[
        (trials['stage'] == 'formal') &
        (trials['responded']) &
        (trials['RT'].notna())
    ].copy()
    
    if len(d) < min_trials_per_window * 2:
        return pd.DataFrame()
    
    rt_self = d[d['Identity'] == 'Self']['RT_ms'].values
    rt_stranger = d[d['Identity'] == 'Stranger']['RT_ms'].values
    
    # 确定滑动窗口范围
    all_rt = d['RT_ms'].values
    rt_min = np.percentile(all_rt, 2)
    rt_max = np.percentile(all_rt, 98)
    
    windows = []
    window_centers = np.arange(rt_min + window_width_ms/2, 
                                rt_max - window_width_ms/2, step_ms)
    
    for center in window_centers:
        lo = center - window_width_ms / 2
        hi = center + window_width_ms / 2
        
        # Self 试次
        self_in_window = d[(d['Identity'] == 'Self') & 
                           (d['RT_ms'] >= lo) & (d['RT_ms'] < hi)]
        n_self = len(self_in_window)
        p_self = self_in_window['ResponseIsMatch'].mean() if n_self > 0 else np.nan
        
        # Stranger 试次
        stranger_in_window = d[(d['Identity'] == 'Stranger') & 
                               (d['RT_ms'] >= lo) & (d['RT_ms'] < hi)]
        n_stranger = len(stranger_in_window)
        p_stranger = stranger_in_window['ResponseIsMatch'].mean() if n_stranger > 0 else np.nan
        
        # SPE
        if n_self >= min_trials_per_window / 2 and n_stranger >= min_trials_per_window / 2:
            spe = p_self - p_stranger
            # 两独立比率的 pooled SE
            se_self = np.sqrt(p_self * (1 - p_self) / n_self) if n_self > 0 else 0
            se_stranger = np.sqrt(p_stranger * (1 - p_stranger) / n_stranger) if n_stranger > 0 else 0
            se_spe = np.sqrt(se_self**2 + se_stranger**2)
            
            # z-test
            if se_spe > 0:
                z_stat = spe / se_spe
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = np.nan
                p_value = np.nan
            
            windows.append({
                'rt_center_ms': center,
                'rt_lo_ms': lo,
                'rt_hi_ms': hi,
                'n_self': n_self,
                'n_stranger': n_stranger,
                'n_total': n_self + n_stranger,
                'p_self': p_self,
                'p_stranger': p_stranger,
                'spe': spe,
                'se_spe': se_spe,
                'spe_ci_lo': spe - 1.96 * se_spe,
                'spe_ci_hi': spe + 1.96 * se_spe,
                'z_stat': z_stat,
                'p_value': p_value,
                'significant_05': p_value < 0.05 if not np.isnan(p_value) else False
            })
    
    return pd.DataFrame(windows)

def analyze_spe_windows_global(all_df, window_widths=[100, 200, 300, 400]):
    """
    对所有被试的聚合数据进行多窗口宽度 SPE 分析
    返回不同窗口宽度下的分析结果
    """
    formal_trials = all_df[
        (all_df['stage'] == 'formal') &
        (all_df['responded']) &
        (all_df['RT'].notna())
    ].copy()
    
    # 排除质量差的组
    formal_trials = formal_trials[~formal_trials['groupID'].isin(EXCLUDE_GROUPS)]
    
    results = {}
    for ww in window_widths:
        sw_result = sliding_window_spe(
            formal_trials,
            window_width_ms=ww,
            step_ms=ww // 4,
            min_trials_per_window=max(20, ww // 10)
        )
        if len(sw_result) > 0:
            results[ww] = sw_result
    
    return results, formal_trials

# ============================================================================
# 5. 统计分析函数
# ============================================================================
def analyze_spe_significance(sw_results, formal_trials):
    """
    分析 SPE 显著反应的 RT 范围
    """
    analysis = {}
    
    for ww, df in sw_results.items():
        sig_windows = df[df['significant_05']]
        
        if len(sig_windows) > 0:
            sig_rt_range = (sig_windows['rt_lo_ms'].min(), 
                           sig_windows['rt_hi_ms'].max())
            peak_spe = sig_windows.loc[sig_windows['spe'].idxmax()]
        else:
            sig_rt_range = (np.nan, np.nan)
            peak_spe = None
        
        analysis[ww] = {
            'n_windows': len(df),
            'n_sig_windows': len(sig_windows),
            'sig_proportion': len(sig_windows) / len(df) if len(df) > 0 else 0,
            'sig_rt_range_ms': sig_rt_range,
            'mean_spe': df['spe'].mean(),
            'max_spe': df['spe'].max(),
            'peak_rt_ms': peak_spe['rt_center_ms'] if peak_spe is not None else np.nan,
            'peak_spe': peak_spe['spe'] if peak_spe is not None else np.nan,
            'overall_rt_range_ms': (df['rt_lo_ms'].min(), df['rt_hi_ms'].max())
        }
    
    return analysis

def per_group_crf_analysis(all_df):
    """
    按组别分析 CRF-SPE
    """
    group_results = {}
    
    for gid in sorted(all_df['groupID'].unique()):
        if gid in EXCLUDE_GROUPS:
            continue
        
        group_data = all_df[all_df['groupID'] == gid]
        
        # 分别对 Matching 和 NonMatching 计算 SPE
        for cond in ['Matching', 'NonMatching']:
            cond_data = group_data[group_data['Condition'] == cond]
            crf_s, crf_st, spe_curve = compute_spe_crf(cond_data, n_quantiles=5)
            
            key = f"G{gid}_{cond}"
            group_results[key] = {
                'groupID': gid,
                'condition': cond,
                'P': CONDITIONS[gid]['P'],
                'T_ms': CONDITIONS[gid]['T'] * 1000,
                'W_ms': CONDITIONS[gid]['W'] * 1000,
                'label': CONDITIONS[gid]['label'],
                'spe_mean': spe_curve['spe_upper_prop'].mean() if len(spe_curve) > 0 else np.nan,
                'spe_max': spe_curve['spe_upper_prop'].max() if len(spe_curve) > 0 else np.nan,
                'spe_curve': spe_curve,
                'n_trials': len(cond_data),
                'n_formal_responded': len(cond_data[(cond_data['stage']=='formal') & 
                                                     (cond_data['responded'])]),
            }
    
    return group_results

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_sliding_window_spe(sw_results, out_dir):
    """绘制多窗口宽度的滑动窗口 SPE 图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#ff9800', '#2196f3', '#4caf50', '#9c27b0']
    
    for idx, (ww, df) in enumerate(sw_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        
        # SPE 曲线 + 95% CI
        ax.fill_between(df['rt_center_ms'], 
                         df['spe_ci_lo'], df['spe_ci_hi'],
                         alpha=0.15, color=colors[idx])
        ax.plot(df['rt_center_ms'], df['spe'], '-', color=colors[idx], 
                linewidth=1.5, label='SPE')
        
        # 显著区间高亮
        sig_df = df[df['significant_05']]
        if len(sig_df) > 0:
            ax.scatter(sig_df['rt_center_ms'], sig_df['spe'],
                      color='red', s=30, zorder=5, marker='o',
                      edgecolors='darkred', linewidth=0.5,
                      label=f'Sig (p<.05, n={len(sig_df)})')
        
        # 零线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # 标注 0.3-0.7 秒范围
        ax.axvspan(300, 700, alpha=0.05, color='orange')
        ax.axvline(x=300, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=700, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_title(f'Window Width = {ww} ms', fontsize=12, fontweight='bold')
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('SPE = P(Self) - P(Stranger)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        sig_rt_range = (sig_df['rt_lo_ms'].min(), sig_df['rt_hi_ms'].max()) if len(sig_df) > 0 else (np.nan, np.nan)
        ax.text(0.02, 0.98, 
                f"Significant RT: [{sig_rt_range[0]:.0f}, {sig_rt_range[1]:.0f}] ms\n"
                f"Mean SPE = {df['spe'].mean():.4f}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('SPE (Self-Preference Effect) Sliding Window Analysis\n'
                 'Orange shaded area = hypothesized 0.3-0.7s range',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = out_dir / 'SPE_sliding_window_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path

def plot_crf_by_group(all_df, group_results, out_dir):
    """绘制按组别的 CRF-SPE 对比图"""
    good_groups = [g for g in sorted(all_df['groupID'].unique()) 
                   if g not in EXCLUDE_GROUPS]
    n_groups = len(good_groups)
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()
    
    for idx, gid in enumerate(good_groups):
        ax = axes[idx]
        group_data = all_df[all_df['groupID'] == gid]
        
        # Matching condition
        for ident, color, marker, ls in [('Self', '#ff9800', 'o', '-'),
                                          ('Stranger', '#2196f3', 's', '--')]:
            trials = group_data[(group_data['Condition'] == 'Matching') & 
                               (group_data['Identity'] == ident)]
            crf = compute_crf(trials, n_quantiles=5)
            if len(crf) > 0:
                ax.errorbar(crf['rt_mean_ms'], crf['upper_prop'],
                           yerr=crf['sem'] * 1.96,
                           marker=marker, linestyle=ls, color=color,
                           linewidth=1.5, markersize=6, capsize=3,
                           label=f'{ident}')
        
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(f'G{gid}: {CONDITIONS[gid]["label"]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('P(Match Key)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Add SPE annotation
        key_m = f"G{gid}_Matching"
        if key_m in group_results:
            spe_val = group_results[key_m]['spe_mean']
            ax.text(0.98, 0.05, f'SPE={spe_val:.3f}',
                    transform=ax.transAxes, fontsize=8, ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Hide unused axes
    for idx in range(n_groups, 8):
        axes[idx].set_visible(False)
    
    plt.suptitle('CRF by Group (Matching Condition): Self vs Stranger\n'
                 f'Q=5 quantiles, Error bars = 95% CI',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = out_dir / 'SPE_crf_by_group.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path

def plot_spe_summary_dashboard(sw_results, group_results, analysis, all_df, out_dir):
    """综合 SPE 仪表盘"""
    fig = plt.figure(figsize=(20, 14))
    
    # 1. 主滑动窗口 SPE (窗口=200ms) - 左上
    ax1 = fig.add_subplot(2, 3, 1)
    ww = 200
    if ww in sw_results:
        df = sw_results[ww]
        ax1.fill_between(df['rt_center_ms'], df['spe_ci_lo'], df['spe_ci_hi'],
                         alpha=0.15, color='#ff9800')
        ax1.plot(df['rt_center_ms'], df['spe'], '-', color='#ff9800', linewidth=1.5)
        sig_df = df[df['significant_05']]
        if len(sig_df) > 0:
            ax1.scatter(sig_df['rt_center_ms'], sig_df['spe'],
                       color='red', s=25, zorder=5, marker='o')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax1.axvspan(300, 700, alpha=0.05, color='orange')
    ax1.set_title(f'Sliding Window SPE (width={ww}ms)', fontweight='bold')
    ax1.set_xlabel('RT (ms)')
    ax1.set_ylabel('SPE')
    ax1.grid(True, alpha=0.3)
    
    # 2. 不同窗口宽度对比 - 右上
    ax2 = fig.add_subplot(2, 3, 2)
    colors_ww = ['#ff9800', '#2196f3', '#4caf50', '#9c27b0']
    for idx, (ww, df) in enumerate(sw_results.items()):
        color = colors_ww[idx % len(colors_ww)]
        ax2.plot(df['rt_center_ms'], df['spe'], '-', color=color,
                linewidth=1.2, alpha=0.8, label=f'W={ww}ms')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.axvspan(300, 700, alpha=0.05, color='orange')
    ax2.set_title('SPE Across Window Widths', fontweight='bold')
    ax2.set_xlabel('RT (ms)')
    ax2.set_ylabel('SPE')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 显著 RT 范围条图 - 左中
    ax3 = fig.add_subplot(2, 3, 3)
    bar_data = [(ww, info['sig_rt_range_ms']) for ww, info in analysis.items()
               if not any(np.isnan(x) for x in info['sig_rt_range_ms'])]
    if bar_data:
        ww_list = [b[0] for b in bar_data]
        lo_list = [b[1][0] for b in bar_data]
        hi_list = [b[1][1] for b in bar_data]
        ax3.barh([str(w) + 'ms' for w in ww_list],
                [hi - lo for lo, hi in zip(lo_list, hi_list)],
                left=lo_list, color='#ff9800', alpha=0.7, edgecolor='darkorange')
        ax3.axvline(x=300, color='orange', linestyle=':', linewidth=0.8)
        ax3.axvline(x=700, color='orange', linestyle=':', linewidth=0.8)
    ax3.set_title('Significant SPE RT Range by Window Width\n(Orange lines = hypothesized 300-700ms)',
                  fontweight='bold')
    ax3.set_xlabel('RT (ms)')
    
    # 4. Per-group SPE 条形图 - 左下
    ax4 = fig.add_subplot(2, 3, 4)
    good_groups = [g for g in sorted(all_df['groupID'].unique())
                   if g not in EXCLUDE_GROUPS]
    spe_match = []
    spe_nonmatch = []
    for gid in good_groups:
        km = f"G{gid}_Matching"
        kn = f"G{gid}_NonMatching"
        spe_match.append(group_results[km]['spe_mean'] if km in group_results else np.nan)
        spe_nonmatch.append(group_results[kn]['spe_mean'] if kn in group_results else np.nan)
    
    x = np.arange(len(good_groups))
    w = 0.35
    ax4.bar(x - w/2, spe_match, w, label='Matching', color='#ff9800', alpha=0.8)
    ax4.bar(x + w/2, spe_nonmatch, w, label='NonMatching', color='#2196f3', alpha=0.8)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'G{g}' for g in good_groups])
    ax4.set_title('Mean SPE by Group & Condition', fontweight='bold')
    ax4.set_ylabel('Mean SPE')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. RT 分布 + SPE density - 右中
    ax5 = fig.add_subplot(2, 3, 5)
    formal_trials = all_df[
        (all_df['stage'] == 'formal') & (all_df['responded']) & (all_df['RT'].notna())
    ]
    formal_trials = formal_trials[~formal_trials['groupID'].isin(EXCLUDE_GROUPS)]
    
    rt_self = formal_trials[formal_trials['Identity'] == 'Self']['RT_ms']
    rt_stranger = formal_trials[formal_trials['Identity'] == 'Stranger']['RT_ms']
    
    ax5.hist(rt_self, bins=80, alpha=0.5, color='#ff9800', label='Self', density=True)
    ax5.hist(rt_stranger, bins=80, alpha=0.5, color='#2196f3', label='Stranger', density=True)
    
    # 叠加 SPE 曲线 (标准化)
    if 200 in sw_results:
        df = sw_results[200]
        spe_norm = (df['spe'] - df['spe'].min()) / (df['spe'].max() - df['spe'].min() + 0.001)
        spe_norm = spe_norm * 0.8 * ax5.get_ylim()[1] if ax5.get_ylim()[1] > 0 else spe_norm
        ax5_twin = ax5.twinx()
        ax5_twin.plot(df['rt_center_ms'], df['spe'], '-', color='red', linewidth=2, label='SPE')
        ax5_twin.fill_between(df['rt_center_ms'], df['spe_ci_lo'], df['spe_ci_hi'],
                              alpha=0.1, color='red')
        ax5_twin.set_ylabel('SPE', color='red')
        ax5_twin.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax5.axvspan(300, 700, alpha=0.05, color='orange')
    ax5.set_title('RT Distribution + SPE Overlay\n(Orange = hypothesized 0.3-0.7s)',
                  fontweight='bold')
    ax5.set_xlabel('RT (ms)')
    ax5.set_ylabel('Density')
    ax5.legend(fontsize=8)
    
    # 6. 关键发现摘要 - 右下
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # 编译关键发现
    findings = []
    findings.append("Key Findings - SPE Reaction Window Analysis")
    findings.append("=" * 50)
    
    # 显著 RT 范围
    for ww, info in analysis.items():
        lo, hi = info['sig_rt_range_ms']
        if not np.isnan(lo):
            findings.append(f"Window {ww}ms: Significant SPE in [{lo:.0f}, {hi:.0f}] ms")
    
    findings.append("")
    findings.append("Hypothesis Verification (0.3-0.7s):")
    # Check overlap with 300-700ms
    for ww, info in analysis.items():
        lo, hi = info['sig_rt_range_ms']
        if not np.isnan(lo):
            overlap = max(0, min(hi, 700) - max(lo, 300))
            hypothesis_range = 400  # 700-300
            overlap_pct = overlap / hypothesis_range * 100
            findings.append(f"  W={ww}ms: Overlap with 300-700ms = {overlap_pct:.0f}%")
    
    findings.append("")
    findings.append("Group-Level SPE (Matching):")
    for gid in good_groups:
        km = f"G{gid}_Matching"
        if km in group_results:
            findings.append(f"  G{gid} ({CONDITIONS[gid]['label']}): SPE={group_results[km]['spe_mean']:.4f}")
    
    ax6.text(0.05, 0.95, '\n'.join(findings), transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('SPE Reaction Window Analysis - Comprehensive Dashboard\n'
                 f'Data: {len(all_df)} trials, Groups G3-G8 (G1,G2 excluded for quality)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = out_dir / 'SPE_analysis_dashboard.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path

# ============================================================================
# 7. 主分析流程
# ============================================================================
def main():
    print("=" * 70)
    print("  SPE (Self-Preference Effect) Reaction Window Analysis")
    print("=" * 70)
    print()
    
    # 加载数据
    print("[1/5] Loading data...")
    all_df = load_all_data()
    
    # 基本信息
    n_subjects = len(all_df[['groupID', 'subjectID']].drop_duplicates())
    print(f"       Subjects: {n_subjects}")
    print(f"       Groups: {sorted(all_df['groupID'].unique())}")
    
    formal_responded = all_df[(all_df['stage']=='formal') & (all_df['responded'])]
    print(f"       Formal responded trials: {len(formal_responded)}")
    print(f"       Omission rate: {1 - len(formal_responded)/len(all_df[all_df['stage']=='formal']):.3f}")
    
    # 滑动窗口 SPE 分析
    print("\n[2/5] Computing sliding-window SPE...")
    window_widths = [100, 200, 300, 400]
    sw_results, formal_trials = analyze_spe_windows_global(all_df, window_widths)
    
    for ww, df in sw_results.items():
        sig_count = df['significant_05'].sum()
        print(f"       Window {ww}ms: {len(df)} windows, {sig_count} significant")
        if sig_count > 0:
            sig_df = df[df['significant_05']]
            print(f"         Sig RT range: [{sig_df['rt_lo_ms'].min():.0f}, {sig_df['rt_hi_ms'].max():.0f}] ms")
            print(f"         Peak SPE: {sig_df['spe'].max():.4f} at {sig_df.loc[sig_df['spe'].idxmax(), 'rt_center_ms']:.0f} ms")
    
    # 统计分析
    print("\n[3/5] Statistical analysis...")
    analysis = analyze_spe_significance(sw_results, formal_trials)
    
    # Per-group CRF 分析
    group_results = per_group_crf_analysis(all_df)
    print(f"       Analyzed {len(group_results)} group-condition combinations")
    
    # 生成可视化
    print("\n[4/5] Generating visualizations...")
    plot_sliding_window_spe(sw_results, OUT_DIR)
    plot_crf_by_group(all_df, group_results, OUT_DIR)
    plot_spe_summary_dashboard(sw_results, group_results, analysis, all_df, OUT_DIR)
    
    # 保存分析结果
    print("\n[5/5] Saving analysis results...")
    
    # 滑动窗口结果
    for ww, df in sw_results.items():
        df.to_csv(OUT_DIR / f'spe_sliding_window_{ww}ms.csv', index=False)
    
    # 汇总统计
    summary_rows = []
    for ww, info in analysis.items():
        lo, hi = info['sig_rt_range_ms']
        row = {
            'window_width_ms': ww,
            'n_windows': info['n_windows'],
            'n_sig_windows': info['n_sig_windows'],
            'sig_proportion': info['sig_proportion'],
            'sig_rt_lo_ms': lo,
            'sig_rt_hi_ms': hi,
            'mean_spe': info['mean_spe'],
            'max_spe': info['max_spe'],
            'peak_rt_ms': info['peak_rt_ms'],
            'peak_spe': info['peak_spe']
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / 'spe_rt_window_summary.csv', index=False)
    print(f"       Summary saved to: {OUT_DIR / 'spe_rt_window_summary.csv'}")
    
    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)
    
    return all_df, sw_results, analysis, group_results

if __name__ == "__main__":
    all_df, sw_results, analysis, group_results = main()
