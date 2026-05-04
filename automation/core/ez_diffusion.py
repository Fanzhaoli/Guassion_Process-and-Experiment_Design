"""
EZ-Diffusion 参数估计模块
从行为数据 (RT, ACC) 反推 DDM 参数 (v, a, t0)

参考: Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007).
      An EZ-diffusion model for response time and accuracy.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


def ez_diffusion(rt_mean: float,
                 rt_var: float,
                 p_correct: float,
                 s: float = 1.0) -> Dict[str, float]:
    """从汇总统计量估计 DDM 参数

    使用 EZ-diffusion 的闭合解:
      给定正确率 Pc、正确反应时均值 M、正确反应时方差 V，
      反推 v, a, Ter。

    Args:
        rt_mean: 正确试次的平均反应时 (seconds)
        rt_var: 正确试次的反应时方差
        p_correct: 正确率 (0-1)
        s: 扩散系数 (默认为1)

    Returns:
        dict: {'v': drift_rate, 'a': boundary, 'ter': non_decision_time}
    """
    p_correct = np.clip(p_correct, 0.51, 0.99)

    if p_correct == 0.5:
        return {'v': 0.0, 'a': np.nan, 'ter': np.nan}

    # 对数几率比
    L = np.log(p_correct / (1 - p_correct))

    # 边界估计
    numerator = s**2 * L * (p_correct**2 * L - p_correct * L + p_correct - 0.5)
    if numerator <= 0:
        numerator = 1e-6

    denominator = p_correct * (1 - p_correct) * rt_var
    if denominator <= 0:
        denominator = 1e-6

    v_sign = 1 if p_correct >= 0.5 else -1
    v = v_sign * np.sqrt(numerator / denominator)

    if v == 0:
        v = 1e-6

    a = (s**2 * L) / v

    ter = rt_mean - (a / (2 * v)) * ((1 - np.exp(-v * a / s**2)) /
                                     (1 + np.exp(-v * a / s**2)))

    ter = np.clip(ter, 0.05, rt_mean * 0.5)

    return {
        'v': float(v),
        'a': float(a),
        'ter': float(ter)
    }


def ez_diffusion_from_data(df: pd.DataFrame,
                            rt_col: str = 'RT',
                            correct_col: str = 'response',
                            correct_value: int = 1) -> Dict[str, float]:
    """从试次级数据直接估计 DDM 参数

    Args:
        df: 包含 RT 和 response 列的 DataFrame
        rt_col: RT 列名
        correct_col: 反应列名
        correct_value: 正确反应的值

    Returns:
        dict: EZ-diffusion 估计的 DDM 参数
    """
    valid = df[df[rt_col].notna() & (df[rt_col] > 0)].copy()

    if len(valid) == 0:
        return {'v': np.nan, 'a': np.nan, 'ter': np.nan}

    p_correct = (valid[correct_col] == correct_value).mean()
    p_correct = np.clip(p_correct, 0.51, 0.99)

    correct_trials = valid[valid[correct_col] == correct_value]

    if len(correct_trials) < 2:
        return {'v': np.nan, 'a': np.nan, 'ter': np.nan}

    rt_mean = correct_trials[rt_col].mean()
    rt_var = correct_trials[rt_col].var(ddof=1)

    if rt_var <= 0:
        rt_var = 0.001

    return ez_diffusion(rt_mean, rt_var, p_correct)


def estimate_condition_level_params(df: pd.DataFrame,
                                    group_cols: list = None) -> pd.DataFrame:
    """对每个实验条件分别进行 EZ-diffusion 参数估计

    Args:
        df: 试次级行为数据
        group_cols: 用于分组的列 (如 ['P', 'T', 'W', 'label'])

    Returns:
        每个条件的 EZ-diffusion 参数 DataFrame
    """
    if group_cols is None:
        group_cols = ['P', 'T', 'W', 'label']

    # 确保分组列存在
    available_cols = [c for c in group_cols if c in df.columns]
    if not available_cols:
        available_cols = [c for c in ['condition_id', 'label'] if c in df.columns]

    results = []

    for keys, group in df.groupby(available_cols):
        params = ez_diffusion_from_data(group)

        if isinstance(keys, tuple):
            row = dict(zip(available_cols, keys))
        else:
            row = {available_cols[0]: keys}

        row.update({
            'v_ez': params['v'],
            'a_ez': params['a'],
            't0_ez': params['ter'],
            'n_trials': len(group),
            'n_valid': len(group[group['RT'].notna() & (group['RT'] > 0)]),
            'p_correct': (group['response'] == 1).mean() if 'response' in group.columns else np.nan,
        })
        results.append(row)

    return pd.DataFrame(results)


def subject_level_ez_diffusion(df: pd.DataFrame,
                               subject_col: str = 'subject',
                               label_col: str = 'label') -> pd.DataFrame:
    """被试层级的 EZ-diffusion 参数估计

    Args:
        df: 试次级行为数据
        subject_col: 被试ID列名
        label_col: 标签列名 (self/stranger)

    Returns:
        每个被试×标签的 EZ-diffusion 参数 DataFrame
    """
    group_cols = [subject_col, label_col]
    available = [c for c in group_cols if c in df.columns]

    results = []
    for keys, group in df.groupby(available):
        params = ez_diffusion_from_data(group)

        if isinstance(keys, tuple):
            row = dict(zip(available, keys))
        else:
            row = {available[0]: keys}

        row.update({
            'v_ez': params['v'],
            'a_ez': params['a'],
            't0_ez': params['ter'],
            'n_trials': len(group),
            'p_correct': (group['response'] == 1).mean() if 'response' in group.columns else np.nan,
            'rt_mean': group[group['RT'].notna()]['RT'].mean() if 'RT' in group.columns else np.nan,
        })
        results.append(row)

    result_df = pd.DataFrame(results)

    # 计算被试层级 SPE
    if label_col in result_df.columns and subject_col in result_df.columns:
        pivot = result_df.pivot_table(
            index=subject_col, columns=label_col,
            values=['v_ez', 'a_ez', 't0_ez', 'rt_mean']
        ).reset_index()

        pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
        pivot = pivot.rename(columns={f'{subject_col}_': subject_col})

        if 'rt_mean_self' in pivot.columns and 'rt_mean_stranger' in pivot.columns:
            pivot['SPE_ms'] = (pivot['rt_mean_self'] - pivot['rt_mean_stranger']) * 1000

        return pivot

    return result_df
