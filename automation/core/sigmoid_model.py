"""
Sigmoid 调制函数模块
基于理论先验的 S2 机制函数，将实验设计参数映射到 DDM 参数

核心函数:
  - compute_v_s2: 实验设计 → 漂移率 v
  - compute_a_s2: 实验设计 → 决策边界 a
  - v_P_Function: 练习次数对漂移率的调节
  - 支持可调参数的模型拟合
"""

import numpy as np
from scipy.special import expit as sigmoid
from typing import Dict, Tuple, List, Optional


def v_P_Function(P, P1=4, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    """练习次数 P 对漂移率 v 的调节函数

    P 通过一个二阶 Sigmoid 过程影响 v:
    1. k(P) 随 P 增加 (从 k_min 到 k_max)
    2. v_P 随 k*(P-P1) 变化

    Args:
        P: 练习次数
        P1: 半效点
        k_min, k_max: k 的取值范围
        gamma: k 的增长速率
        P0: k 的半效练习次数

    Returns:
        v_P ∈ (0, 1): v 的练习调节系数
    """
    k = k_min + (k_max - k_min) / (1 + np.exp(-gamma * (np.asarray(P, dtype=float) - P0)))
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(P, dtype=float) - P1)))


def compute_v_s2(T, P, condition_key,
                 alaph1=1.5, alaph2=-0.4,
                 gamma=0.2, T_0=100.0, k_T=0.01,
                 base_scale=3.0) -> np.ndarray:
    """S2 机制: 实验设计 → 漂移率 v

    漂移率由三个因素构成:
    1. v_T: 刺激呈现时间 T 的效应 (充分处理 → 更高 v)
    2. v_P: 练习次数 P 的效应 (更多练习 → 更高 v)
    3. condition_key: Self/Stranger 标签的调制

    v = v_T × v_P × base_scale × (1 + alpha_condition)

    Args:
        T: 刺激呈现时间 (ms)
        P: 练习次数
        condition_key: 1=self, 0=stranger
        alaph1: self 条件的增强系数
        alaph2: stranger 条件的调制系数
        gamma: 练习效应速率
        T_0: T 的半效时间 (ms)
        k_T: T 对 v 的影响速率
        base_scale: 基础缩放

    Returns:
        v: 漂移率
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)

    v_T = 1.0 / (1.0 + np.exp(-k_T * (T - T_0)))
    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=gamma, P0=32)
    v_0 = v_T * v_P * base_scale

    v_1 = np.where(
        np.asarray(condition_key) == 1,
        v_0 * (1 + alaph1),
        v_0 * (1 + alaph2)
    )

    return v_1


def compute_a_s2(M, beta1=0.2, beta2=0.0, k=0.01, M_0=600.0, base_scale=3.0):
    """S2 机制: 实验设计 → 决策边界 a

    决策边界受 M = T + W (总可用时间) 的影响。
    当 M > M_0 时边界增大 (更保守), M ≤ M_0 时边界较小 (更激进)。

    Args:
        M: T + W (ms)
        beta1: M > M_0 时的边界调制
        beta2: M ≤ M_0 时的边界调制
        k: Sigmoid 斜率
        M_0: M 的临界点 (ms)
        base_scale: 基础缩放

    Returns:
        a: 决策边界
    """
    M = np.asarray(M, dtype=float)

    a_0 = 1.0 / (1.0 + np.exp(-k * (M - M_0))) * base_scale

    a_1 = np.where(
        M > M_0,
        a_0 * (1 + beta1),
        a_0 * (1 + beta2)
    )

    return a_1


class PureSigmoidModel:
    """纯 Sigmoid 生成模型

    不含 GP 修正，仅用 S2 机制函数生成 DDM 参数。
    作为 GP+Sigma 混合模型的基准对照。

    可调参数:
      alaph1: self 条件的 v 增强
      alaph2: stranger 条件的 v 调制
      beta1: 高 M 时的 a 调制
      beta2: 低 M 时的 a 调制
      gamma: 练习效应速率
      v_noise, a_noise: 试次级噪声
    """

    def __init__(self,
                 alaph1=1.5, alaph2=-0.4,
                 beta1=0.2, beta2=0.0,
                 gamma=0.2,
                 v_noise=1.0, a_noise=0.5,
                 seed: int = 42):
        self.alaph1 = alaph1
        self.alaph2 = alaph2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.v_noise = v_noise
        self.a_noise = a_noise
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def get_params(self) -> Dict:
        """获取当前参数"""
        return {
            'alaph1': self.alaph1, 'alaph2': self.alaph2,
            'beta1': self.beta1, 'beta2': self.beta2,
            'gamma': self.gamma,
            'v_noise': self.v_noise, 'a_noise': self.a_noise,
        }

    def set_params(self, params: Dict):
        """设置参数"""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def predict_params(self, T, P, W, condition_key) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """预测 DDM 参数

        Args:
            T, P, W: 实验设计参数
            condition_key: 1=self, 0=stranger

        Returns:
            (v, a, t0, z): DDM 参数
        """
        M = np.asarray(T) + np.asarray(W)

        v_s2 = compute_v_s2(T, P, condition_key,
                            alaph1=self.alaph1, alaph2=self.alaph2,
                            gamma=self.gamma)
        a_s2 = compute_a_s2(M, beta1=self.beta1, beta2=self.beta2)

        # 加入试次噪声
        v_final = self.rng.normal(v_s2, self.v_noise)
        a_final = self.rng.normal(a_s2, self.a_noise)
        a_final = np.maximum(a_final, 0.1)

        # 对称起点
        z_final = a_final / 2.0
        t0_final = np.full_like(v_final, 0.2)

        return v_final, a_final, t0_final, z_final

    def fit(self, real_data_summary: pd.DataFrame,
            n_iter: int = 200) -> Dict:
        """使用差分进化优化拟合真实数据

        Args:
            real_data_summary: 真实数据的条件水平汇总
            n_iter: 最大迭代次数

        Returns:
            dict: 优化结果
        """
        from scipy.optimize import differential_evolution

        bounds = [
            (0.1, 4.0),    # alaph1
            (-2.0, 1.0),   # alaph2
            (-1.0, 2.0),   # beta1
            (-1.0, 1.0),   # beta2
            (0.01, 1.0),   # gamma
            (0.1, 3.0),    # v_noise
            (0.1, 2.0),    # a_noise
        ]

        def objective(params):
            self.alaph1, self.alaph2, self.beta1, self.beta2, self.gamma, self.v_noise, self.a_noise = params

            total_rmse = 0.0
            n_cond = 0

            for _, row in real_data_summary.iterrows():
                v_s2, a_s2, t0_arr, z_arr = self.predict_params(
                    row['T'], row['P'], row['W'], 1 if row.get('label', 'self') == 'self' else 0
                )

                observed_rt = row.get('rt_mean', 0)
                observed_acc = row.get('p_correct', 0.5)

                # 简单估计 RT
                if v_s2[0] != 0:
                    pred_rt = a_s2[0] / (2 * abs(v_s2[0]))
                else:
                    pred_rt = 0

                # RMSE
                total_rmse += np.sqrt((observed_rt - pred_rt)**2)
                n_cond += 1

            return total_rmse / max(n_cond, 1)

        result = differential_evolution(objective, bounds, maxiter=n_iter, seed=self.seed)

        self.alaph1, self.alaph2, self.beta1, self.beta2, self.gamma, self.v_noise, self.a_noise = result.x

        return {
            'success': result.success,
            'final_rmse': result.fun,
            'optimized_params': self.get_params(),
        }
