"""
实验设计空间模块
定义并管理实验设计空间 Ω = (P, T, W, M)
P: 练习次数 (Practice trials)
T: 刺激呈现时间 (Stimulus presentation time, ms)
W: 反应窗口 (Response window, ms)
M: 元变量 (T+W, 或其他派生维度)

参考: Sui, J., He, X., & Humphreys, G. W. (2012)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import List, Dict, Tuple, Optional
import json


class DesignSpace:
    """实验设计空间管理器
    
    负责生成和管理实验设计参数空间 Ω = (P, T, W, M) 的所有合法组合。
    支持离散网格搜索、拉丁超立方采样和自适应边界探索。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # 基于已有实验条件的参数边界
        self.P_range = (0, 150)       # 练习次数
        self.T_range = (10, 600)      # 刺激呈现时间 (ms)
        self.W_range = (200, 1500)    # 反应窗口 (ms)

        # 实验条件编码: 6组真实实验条件
        self.real_conditions = [
            {'P': 0,   'T': 30,  'W': 300,  'group': 1},
            {'P': 0,   'T': 30,  'W': 600,  'group': 2},
            {'P': 120, 'T': 30,  'W': 600,  'group': 3},
            {'P': 120, 'T': 80,  'W': 600,  'group': 4},
            {'P': 8,   'T': 100, 'W': 1100, 'group': 5},
            {'P': 120, 'T': 500, 'W': 1500, 'group': 6},
        ]

    def generate_grid(self,
                      P_values: Optional[List[int]] = None,
                      T_values: Optional[List[int]] = None,
                      W_values: Optional[List[int]] = None) -> pd.DataFrame:
        """生成离散网格设计空间

        Args:
            P_values: 练习次数取值列表
            T_values: 刺激呈现时间取值列表 (ms)
            W_values: 反应窗口取值列表 (ms)

        Returns:
            包含所有参数组合的 DataFrame
        """
        if P_values is None:
            # 默认: 覆盖练习效应的关键转折点
            P_values = [0, 4, 8, 16, 32, 64, 120, 150]

        if T_values is None:
            # 默认: 从阈下到充分处理的梯度
            T_values = [10, 30, 50, 80, 100, 200, 350, 500, 600]

        if W_values is None:
            # 默认: 从迫选到自由反应的窗口梯度
            W_values = [200, 300, 400, 600, 800, 1000, 1200, 1500]

        combinations = list(product(P_values, T_values, W_values))
        design_points = []

        for P, T, W in combinations:
            M = T + W
            design_points.append({
                'P': P, 'T': T, 'W': W, 'M': M,
                'condition_id': f"P{P}_T{T}_W{W}"
            })

        self.design_df = pd.DataFrame(design_points)
        return self.design_df

    def generate_lhs(self, n_points: int = 50) -> pd.DataFrame:
        """使用拉丁超立方采样生成设计点

        Args:
            n_points: 采样点数

        Returns:
            包含采样设计点的 DataFrame
        """
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=3, seed=self.seed)
        samples = sampler.random(n=n_points)

        scaled = qmc.scale(
            samples,
            l_bounds=np.array([self.P_range[0], self.T_range[0], self.W_range[0]]),
            u_bounds=np.array([self.P_range[1], self.T_range[1], self.W_range[1]]),
        )

        P_vals = np.round(scaled[:, 0]).astype(int)
        T_vals = np.round(scaled[:, 1]).astype(int)
        W_vals = np.round(scaled[:, 2]).astype(int)

        design_points = []
        for i in range(n_points):
            P, T, W = P_vals[i], T_vals[i], W_vals[i]
            M = int(T + W)
            design_points.append({
                'P': P, 'T': T, 'W': W, 'M': M,
                'condition_id': f"LHS_{i:03d}"
            })

        self.lhs_df = pd.DataFrame(design_points)
        return self.lhs_df

    def generate_adaptive_bounds(self,
                                 gp_model,
                                 n_candidates: int = 1000,
                                 n_select: int = 10,
                                 exploration_weight: float = 0.3) -> pd.DataFrame:
        """基于GP模型的采集函数自适应选择下一个探索点

        使用Expected Improvement (EI) 结合不确定性采样，
        在SPE效应最敏感的区域增加采样密度。

        Args:
            gp_model: 已训练的GP模型
            n_candidates: 候选点数量
            n_select: 选择点数
            exploration_weight: 探索权重 (0=纯利用, 1=纯探索)

        Returns:
            下一个迭代的设计点 DataFrame
        """
        candidates = self.generate_lhs(n_candidates)

        Pn, Tn, Wn = self._normalize(candidates['P'].values,
                                     candidates['T'].values,
                                     candidates['W'].values)
        X_cand = np.column_stack([Pn, Tn, Wn])

        mu, std = gp_model.predict(X_cand, return_std=True)

        # SPE 效应量 = self_RT - stranger_RT (负数越大 = SPE越强)
        spe_magnitude = np.abs(mu)  # 效应强度
        uncertainty = std           # 模型不确定性

        # 采集函数: 加权组合
        acquisition = (1 - exploration_weight) * spe_magnitude + \
                      exploration_weight * uncertainty

        top_indices = np.argsort(acquisition)[-n_select:]
        selected = candidates.iloc[top_indices].copy()
        selected['acquisition_value'] = acquisition[top_indices]
        selected['gp_pred_mean'] = mu[top_indices]
        selected['gp_pred_std'] = std[top_indices]

        return selected.reset_index(drop=True)

    def get_real_conditions_df(self) -> pd.DataFrame:
        """获取真实实验条件的数据框"""
        df = pd.DataFrame(self.real_conditions)
        df['M'] = df['T'] + df['W']
        df['condition_id'] = df.apply(
            lambda r: f"G{r['group']}_P{r['P']}_T{r['T']}_W{r['W']}", axis=1
        )
        return df

    def _normalize(self, P, T, W):
        """将原始参数空间归一化到约[-1, 1]"""
        P_norm = (np.asarray(P, dtype=float) - 75.0) / 75.0
        T_norm = (np.asarray(T, dtype=float) - 305.0) / 295.0
        W_norm = (np.asarray(W, dtype=float) - 850.0) / 650.0
        return P_norm, T_norm, W_norm

    def normalize_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """对DataFrame中的设计点进行归一化"""
        return self._normalize(df['P'].values, df['T'].values, df['W'].values)


class DesignSpaceSummary:
    """设计空间统计摘要"""

    @staticmethod
    def summarize(design_df: pd.DataFrame) -> Dict:
        """生成设计空间的描述性统计"""
        return {
            'n_conditions': len(design_df),
            'P': {'min': int(design_df['P'].min()), 'max': int(design_df['P'].max()),
                  'unique': int(design_df['P'].nunique())},
            'T': {'min': int(design_df['T'].min()), 'max': int(design_df['T'].max()),
                  'unique': int(design_df['T'].nunique())},
            'W': {'min': int(design_df['W'].min()), 'max': int(design_df['W'].max()),
                  'unique': int(design_df['W'].nunique())},
            'M': {'min': int(design_df['M'].min()), 'max': int(design_df['M'].max()),
                  'unique': int(design_df['M'].nunique())},
        }
