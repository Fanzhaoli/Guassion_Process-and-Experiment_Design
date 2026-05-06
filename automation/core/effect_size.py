"""
效应量计算模块
包含 G-power、贝叶斯因子 BF01 和效应量指标的计算

参考: 
  - Faul, F., et al. (2007). G*Power 3.
  - Rouder, J. N., et al. (2009). Bayesian t-tests.
  - Cohen, J. (1988). Statistical power analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """计算配对样本 Cohen's d

    Args:
        x, y: 配对的两组数据

    Returns:
        dict: d, p_value, t_stat, mean_diff, sd_diff
    """
    diff = x - y
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    if sd_diff == 0:
        return {'d': np.nan, 'p_value': np.nan, 't_stat': np.nan,
                'mean_diff': mean_diff, 'sd_diff': 0}

    d = mean_diff / sd_diff

    n = len(diff)
    t_stat = mean_diff / (sd_diff / np.sqrt(n))
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

    return {
        'd': float(d),
        'p_value': float(p_value),
        't_stat': float(t_stat),
        'mean_diff': float(mean_diff),
        'sd_diff': float(sd_diff),
        'n': n,
    }


def cohens_d_independent(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """计算独立样本 Cohen's d"""
    nx, ny = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))

    if pooled_sd == 0:
        return {'d': np.nan, 'p_value': np.nan, 't_stat': np.nan,
                'mean_diff': mean_x - mean_y, 'pooled_sd': 0}

    d = (mean_x - mean_y) / pooled_sd

    se = pooled_sd * np.sqrt(1 / nx + 1 / ny)
    t_stat = (mean_x - mean_y) / se if se > 0 else 0
    df = nx + ny - 2
    p_value = 2 * stats.t.sf(abs(t_stat), df=df)

    return {
        'd': float(d),
        'p_value': float(p_value),
        't_stat': float(t_stat),
        'mean_diff': float(mean_x - mean_y),
        'pooled_sd': float(pooled_sd),
    }


def g_power_analysis(effect_size: float,
                     n: int,
                     alpha: float = 0.05,
                     test_type: str = 'paired') -> Dict[str, float]:
    """事后统计功效分析 (G-power)

    Args:
        effect_size: 效应量 (Cohen's d)
        n: 样本量
        alpha: 显著性水平
        test_type: 'paired' 或 'independent'

    Returns:
        dict: power, ncp, critical_t
    """
    if test_type == 'paired':
        df = n - 1
    else:
        df = 2 * n - 2

    critical_t = stats.t.ppf(1 - alpha / 2, df)

    ncp = abs(effect_size) * np.sqrt(n) if test_type == 'paired' else \
          abs(effect_size) * np.sqrt(n / 2)

    power = 1 - stats.nct.cdf(critical_t, df, ncp) + \
            stats.nct.cdf(-critical_t, df, ncp)

    return {
        'power': float(power),
        'ncp': float(ncp),
        'critical_t': float(critical_t),
        'df': df,
        'alpha': alpha,
        'sample_size': n,
        'achieves_80pct': power >= 0.80,
    }


def bayes_factor_paired(x: np.ndarray, y: np.ndarray,
                        r_scale: float = np.sqrt(2) / 2) -> Dict[str, float]:
    """计算配对样本的贝叶斯因子 BF01

    使用 Jeffrey-Zellner-Siow (JZS) prior 的近似计算。
    BF01 > 1 支持 H0; BF01 < 1 支持 H1。

    Args:
        x, y: 配对数据（允许不相等长度，自动取共同索引的交集）
        r_scale: 先验尺度参数 (默认为 medium: sqrt(2)/2)

    Returns:
        dict: BF01, BF10, logBF10, interpretation
    """
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
    diff = x - y
    n = len(diff)

    if n < 2:
        return {'BF01': np.nan, 'BF10': np.nan, 'interpretation': 'insufficient_data'}

    t_stat, p_value = stats.ttest_rel(x, y)
    t_stat = t_stat if not np.isnan(t_stat) else 0

    # JZS Bayes Factor 近似 (Rouder et al., 2009)
    numerator = (1 + n * r_scale**2)**(-0.5)
    denominator = (1 + t_stat**2 / (n - 1))**(-n / 2)
    factor = (1 + n * r_scale**2 * (1 + t_stat**2 / (n - 1))**(-1))**(-(n + 1) / 2)

    bf10 = numerator * denominator * factor

    if np.isnan(bf10) or np.isinf(bf10) or bf10 <= 0:
        return {'BF01': np.nan, 'BF10': np.nan, 'interpretation': 'computation_error'}

    bf01 = 1.0 / bf10

    if bf01 > 100:
        interpretation = 'decisive_H0'
    elif bf01 > 30:
        interpretation = 'very_strong_H0'
    elif bf01 > 10:
        interpretation = 'strong_H0'
    elif bf01 > 3:
        interpretation = 'moderate_H0'
    elif bf01 > 1:
        interpretation = 'anecdotal_H0'
    elif bf01 > 1/3:
        interpretation = 'anecdotal_H1'
    elif bf01 > 1/10:
        interpretation = 'moderate_H1'
    elif bf01 > 1/30:
        interpretation = 'strong_H1'
    elif bf01 > 1/100:
        interpretation = 'very_strong_H1'
    else:
        interpretation = 'decisive_H1'

    return {
        'BF01': float(bf01),
        'BF10': float(bf10),
        'logBF10': float(np.log(bf10)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'interpretation': interpretation,
    }


def compute_spe_metrics(self_rt: np.ndarray,
                         stranger_rt: np.ndarray,
                         self_acc: np.ndarray = None,
                         stranger_acc: np.ndarray = None) -> Dict:
    """计算完整的 SPE 效应量指标集

    Args:
        self_rt: self 条件的 RT 数组
        stranger_rt: stranger 条件的 RT 数组
        self_acc: self 条件的准确率 (可选)
        stranger_acc: stranger 条件的准确率 (可选)

    Returns:
        dict: 包含 RT-SPE、d、power、BF01 等完整指标
    """
    common_mask = ~np.isnan(self_rt) & ~np.isnan(stranger_rt)
    self_rt_clean = self_rt[common_mask]
    stranger_rt_clean = stranger_rt[common_mask]
    n = len(self_rt_clean)

    if n < 3:
        return {'SPE_ms': np.nan, 'n_valid_pairs': n}

    # SPE (ms)
    spe_ms = (self_rt_clean - stranger_rt_clean).mean() * 1000

    # Cohen's d
    d_result = cohens_d_paired(self_rt_clean, stranger_rt_clean)

    # G-power
    power_result = g_power_analysis(d_result['d'], n)

    # Bayes Factor
    bf_result = bayes_factor_paired(self_rt_clean, stranger_rt_clean)

    result = {
        'SPE_ms': float(spe_ms),
        'SPE_self_mean_rt_ms': float(np.mean(self_rt_clean) * 1000),
        'SPE_stranger_mean_rt_ms': float(np.mean(stranger_rt_clean) * 1000),
        'cohens_d': d_result['d'],
        'd_p_value': d_result['p_value'],
        'power': power_result['power'],
        'achieves_80pct': power_result['achieves_80pct'],
        'BF01': bf_result['BF01'],
        'BF10': bf_result['BF10'],
        'interpretation': bf_result['interpretation'],
        'n_valid_pairs': n,
    }

    # ACC 分析 (如果提供)
    if self_acc is not None and stranger_acc is not None:
        valid_acc = ~np.isnan(self_acc) & ~np.isnan(stranger_acc)
        acc_diff = (self_acc[valid_acc] - stranger_acc[valid_acc]).mean()
        result['SPE_acc'] = float(acc_diff)
        result['self_mean_acc'] = float(np.mean(self_acc[valid_acc]))
        result['stranger_mean_acc'] = float(np.mean(stranger_acc[valid_acc]))

    return result


def compute_condition_level_spe(df: pd.DataFrame,
                                subject_col: str = 'subject',
                                label_col: str = 'label',
                                value_col: str = 'RT') -> pd.DataFrame:
    """计算条件层级的 SPE 汇总统计

    Args:
        df: 试次级数据
        subject_col: 被试ID列
        label_col: 标签列
        value_col: 因变量列

    Returns:
        每个设计条件×被试的 SPE DataFrame
    """
    pivot = df.pivot_table(
        index=[subject_col, 'P', 'T', 'W', 'M', 'condition_id'],
        columns=label_col,
        values=value_col,
        aggfunc='mean'
    ).reset_index()

    if 'self' in pivot.columns and 'stranger' in pivot.columns:
        pivot['SPE_raw'] = pivot['self'] - pivot['stranger']
        if 'RT' in value_col:
            pivot['SPE_ms'] = pivot['SPE_raw'] * 1000
            pivot['self_RT_ms'] = pivot['self'] * 1000
            pivot['stranger_RT_ms'] = pivot['stranger'] * 1000

    return pivot
