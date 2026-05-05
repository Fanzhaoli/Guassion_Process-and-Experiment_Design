"""
DDM 仿真引擎模块
实现漂移扩散模型的完整仿真，包括:
- Euler-Maruyama 离散化
- 带 deadline 的累积过程
- 漏答 (omission) 机制
- 试次级噪声注入

参考: Ratcliff, R., & McKoon, G. (2008). The diffusion decision model.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def simulate_ddm_euler(v: float,
                       a: float,
                       z: float,
                       t0: float,
                       dt: float = 0.001,
                       max_time_s: float = 3.0) -> Tuple[float, int]:
    """使用 Euler-Maruyama 方法仿真 DDM 过程

    Args:
        v: 漂移率 (drift rate)
        a: 决策边界 (boundary separation)
        z: 起点 (starting point)
        t0: 非决策时间 (non-decision time, seconds)
        dt: 时间步长 (seconds)
        max_time_s: 最大仿真时间 (seconds)

    Returns:
        (RT, response): 反应时(秒)和反应(1=上界,0=下界/超时)
    """
    x = float(z)
    time = 0.0
    max_steps = int(max_time_s / dt)

    for _ in range(max_steps):
        dx = v * dt + np.sqrt(dt) * np.random.randn()
        x += dx
        time += dt

        if x >= a:
            return t0 + time, 1
        if x <= 0:
            return t0 + time, 0

    return np.nan, np.nan


def simulate_ddm_with_deadline(v: float,
                                a: float,
                                z: float,
                                t0: float,
                                deadline_s: float,
                                dt: float = 0.001,
                                lapse_prob: float = 0.02) -> dict:
    """带 deadline 和 lapse omission 的 DDM 仿真

    Args:
        v: 漂移率
        a: 决策边界
        z: 起点
        t0: 非决策时间 (seconds)
        deadline_s: 反应截止时间 (seconds)
        dt: 时间步长
        lapse_prob: 注意力流失导致的漏答概率

    Returns:
        dict: 包含 RT, response, omission, responded, deadline_reached 等信息
    """
    # 检查 lapse omission
    if np.random.random() < lapse_prob:
        return {
            'RT': np.nan, 'response': np.nan,
            'omission': True, 'responded': False,
            'deadline_reached': False, 'lapse': True
        }

    x = float(z)
    time = 0.0
    max_steps = int(deadline_s / dt)

    for _ in range(max_steps):
        dx = v * dt + np.sqrt(dt) * np.random.randn()
        x += dx
        time += dt

        if x >= a:
            return {
                'RT': t0 + time, 'response': 1,
                'omission': False, 'responded': True,
                'deadline_reached': False, 'lapse': False
            }
        if x <= 0:
            return {
                'RT': t0 + time, 'response': 0,
                'omission': False, 'responded': True,
                'deadline_reached': False, 'lapse': False
            }

    # 达到 deadline 仍未决策
    return {
        'RT': np.nan, 'response': np.nan,
        'omission': True, 'responded': False,
        'deadline_reached': True, 'lapse': False
    }


def compute_lapse_omission_prob(t: float,
                                w: float,
                                base_rate: float = 0.02,
                                max_rate: float = 0.15) -> float:
    """根据刺激呈现时间T和反应窗口W计算漏答概率

    当 T 很短且 W 也很短时，漏答率升高。
    当 T 充分长时，漏答率降低到基础水平。

    Args:
        t: 刺激呈现时间 (ms)
        w: 反应窗口 (ms)
        base_rate: 基础漏答率
        max_rate: 最大漏答率

    Returns:
        漏答概率 [0, 1]
    """
    t_s = t / 1000.0
    w_s = w / 1000.0

    # T 和 W 的联合效应
    if t_s < 0.1:
        t_factor = 1.0
    elif t_s < 0.5:
        t_factor = 1.0 - (t_s - 0.1) / 0.4
    else:
        t_factor = 0.0

    if w_s < 0.5:
        w_factor = 1.0
    elif w_s < 1.0:
        w_factor = 1.0 - (w_s - 0.5) / 0.5
    else:
        w_factor = 0.0

    omission = base_rate + (max_rate - base_rate) * (t_factor * 0.5 + w_factor * 0.5)
    return np.clip(omission, 0.0, 1.0)


def sample_a_positive(a_mean: float, a_noise: float) -> float:
    """对边界 a 采样并确保正值

    使用截断正态分布重采样，避免硬截断导致的堆积问题。

    Args:
        a_mean: a 的均值
        a_noise: a 的噪声标准差

    Returns:
        正值 a 的采样
    """
    a_val = np.random.normal(a_mean, a_noise)

    attempts = 0
    max_attempts = 100

    while a_val <= 0 and attempts < max_attempts:
        a_val = np.random.normal(a_mean, a_noise)
        attempts += 1

    if a_val <= 0:
        a_val = max(0.1, a_mean)

    return float(a_val)


class DDMSimulator:
    """DDM 仿真器封装类

    提供面向批量的 DDM 仿真接口，支持向量化参数输入。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def simulate_trial(self,
                       v: float, a: float, z: float, t0: float,
                       T_ms: int, W_ms: int,
                       dt: float = 0.001,
                       use_deadline: bool = True,
                       use_lapse: bool = True) -> dict:
        """仿真单个试次

        Args:
            v, a, z, t0: DDM 参数
            T_ms: 刺激呈现时间 (ms)
            W_ms: 反应窗口 (ms)
            dt: 时间步长
            use_deadline: 是否使用 deadline 机制
            use_lapse: 是否使用 lapse omission 机制

        Returns:
            dict: 包含 RT, response, omission, responded 等
        """
        if use_lapse:
            lapse_rate = compute_lapse_omission_prob(T_ms, W_ms)
        else:
            lapse_rate = 0.0

        if use_deadline:
            deadline = (T_ms + W_ms) / 1000.0
            return simulate_ddm_with_deadline(
                v, a, z, t0, deadline, dt=dt, lapse_prob=lapse_rate
            )
        else:
            RT, response = simulate_ddm_euler(v, a, z, t0, dt=dt, max_time_s=5.0)
            omission = np.isnan(RT)
            return {
                'RT': RT, 'response': response,
                'omission': omission, 'responded': not omission,
                'deadline_reached': False, 'lapse': False
            }

    def simulate_batch(self,
                       params_df: pd.DataFrame,
                       T_ms_col: str = 'T',
                       W_ms_col: str = 'W',
                       dt: float = 0.001,
                       use_deadline: bool = True,
                       use_lapse: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
        """批量仿真多个试次

        Args:
            params_df: 包含 DDM 参数和实验条件的 DataFrame
            T_ms_col, W_ms_col: T和W的列名
            dt: 时间步长
            use_deadline: 是否使用 deadline 机制
            use_lapse: 是否使用 lapse omission 机制
            verbose: 是否打印进度

        Returns:
            添加了 RT, response, omission, responded 列的 DataFrame
        """
        import pandas as pd

        results = params_df.copy()
        rt_list = []
        resp_list = []
        omis_list = []
        responded_list = []

        total = len(results)
        for idx, row in results.iterrows():
            result = self.simulate_trial(
                v=row['v'], a=row['a'],
                z=row.get('z', row['a'] / 2.0),
                t0=row.get('t0', 0.2),
                T_ms=int(row[T_ms_col]),
                W_ms=int(row[W_ms_col]),
                dt=dt,
                use_deadline=use_deadline,
                use_lapse=use_lapse,
            )
            rt_list.append(result['RT'])
            resp_list.append(result['response'])
            omis_list.append(result['omission'])
            responded_list.append(result['responded'])

            if verbose and (idx + 1) % 500 == 0:
                print(f"  仿真进度: {idx + 1}/{total}")

        results['RT'] = rt_list
        results['response'] = resp_list
        results['omission'] = omis_list
        results['responded'] = responded_list

        return results
