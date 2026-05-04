"""
高斯过程建模模块
基于 GP 的非线性实验设计空间建模

核心功能:
  1. GP 训练与预测 (设计空间 → DDM 参数)
  2. GP + Sigmoid 混合建模架构
  3. 不确定性量化和边界探索
  4. GP 超参数分析与认知解释
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern


class GPModel:
    """高斯过程模型封装

    支持:
      - 多种核函数 (RBF, Matern)
      - 混合不确定性建模
      - 归一化输入输出
    """

    def __init__(self,
                 kernel_type: str = 'rbf',
                 length_scale: float = 1.0,
                 noise_level: float = 1e-5,
                 normalize_y: bool = True,
                 seed: int = 42):
        self.seed = seed
        self.normalize_y = normalize_y

        if kernel_type == 'rbf':
            kernel = 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        elif kernel_type == 'matern':
            kernel = 1.0 * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=noise_level)
        elif kernel_type == 'constant_rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        else:
            kernel = 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=normalize_y,
            random_state=seed,
            n_restarts_optimizer=5,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPModel':
        """训练 GP 模型

        Args:
            X: 输入特征 (n_samples, n_features) - 归一化的 (P, T, W)
            y: 目标值 (n_samples,) - DDM 参数 (v 或 a)

        Returns:
            self
        """
        self.gp.fit(X, y)
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测并返回不确定性

        Args:
            X: 输入特征
            return_std: 是否返回标准差

        Returns:
            (mu, std): 预测均值和标准差
        """
        if return_std:
            mu, std = self.gp.predict(X, return_std=True)
            return mu, std
        else:
            mu = self.gp.predict(X, return_std=False)
            return mu, None

    def get_kernel_params(self) -> Dict:
        """获取核函数超参数"""
        kernel = self.gp.kernel_
        params = {}

        if hasattr(kernel, 'get_params'):
            params = kernel.get_params()

        # 提取 lengthscale
        for name, value in params.items():
            if 'length_scale' in name:
                key = f'{name}'
                params[key] = value

        return params

    def analyze_hyperparameters(self, feature_names: List[str] = None) -> Dict:
        """分析 GP 超参数并提供认知解释

        Args:
            feature_names: 特征维度名称 (默认 ['P', 'T', 'W'])

        Returns:
            包含各维度重要性和可解释文本的字典
        """
        if feature_names is None:
            feature_names = ['P', 'T', 'W']

        kernel = self.gp.kernel_

        # 尝试提取 lengthscale
        lengthscales = None
        if hasattr(kernel, 'k1'):
            if hasattr(kernel.k1, 'length_scale'):
                lengthscales = np.atleast_1d(kernel.k1.length_scale)
        elif hasattr(kernel, 'length_scale'):
            lengthscales = np.atleast_1d(kernel.length_scale)

        if lengthscales is None:
            return {'error': '无法提取 lengthscale', 'feature_names': feature_names}

        # lengthscale 越大 → 该维度变化越平缓 → 影响越小
        # lengthscale 越小 → 该维度变化越剧烈 → 影响越大
        if len(lengthscales) == 1 and len(feature_names) > 1:
            lengthscales = np.full(len(feature_names), lengthscales[0])

        importance = 1.0 / np.maximum(lengthscales, 1e-6)
        importance_norm = importance / importance.sum()

        feature_importance = {}
        for i, name in enumerate(feature_names):
            if i < len(lengthscales):
                feature_importance[name] = {
                    'lengthscale': float(lengthscales[i]),
                    'importance': float(importance_norm[i]),
                    'interpretation': self._interpret_lengthscale(
                        float(lengthscales[i]), float(importance_norm[i])
                    )
                }

        return {
            'feature_importance': feature_importance,
            'signal_variance': float(self.gp.kernel_.k1.k1.constant_value)
            if hasattr(self.gp.kernel_, 'k1') and hasattr(self.gp.kernel_.k1, 'k1') else None,
            'noise_variance': float(self.gp.kernel_.k2.noise_level)
            if hasattr(self.gp.kernel_, 'k2') else None,
        }

    def _interpret_lengthscale(self, ls: float, importance: float) -> str:
        """基于 lengthscale 提供认知解释"""
        if importance > 0.5:
            return "该维度是目标参数的主导调节因素，设计空间在此维度上存在强非线性"
        elif importance > 0.25:
            return "该维度对目标参数有中等影响，是调节效应的次级来源"
        else:
            return "该维度的影响相对平缓，可能与其他维度存在交互效应"

    def save(self, path: Path):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> 'GPModel':
        """加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class GPHybridModel:
    """GP + Sigmoid 混合生成模型

    结构:
      v_mix = w_gp × gp_v + (1-w_gp) × v_sigmoid
      a_mix = w_gp × gp_a + (1-w_gp) × a_sigmoid

    GP 负责学习 Sigmoid 无法捕捉的非线性残差，
    混合权重 w_gp 控制 GP 贡献的比例。
    """

    def __init__(self, w_gp: float = 0.5, seed: int = 42):
        self.w_gp = w_gp
        self.seed = seed

        self.gp_v = GPModel(kernel_type='rbf', normalize_y=True, seed=seed)
        self.gp_a = GPModel(kernel_type='rbf', normalize_y=True, seed=seed + 1)

        self.X_train = None
        self.residual_v = None
        self.residual_a = None

    def fit_residuals(self,
                      X: np.ndarray,
                      target_v: np.ndarray,
                      target_a: np.ndarray,
                      sigmoid_v: np.ndarray,
                      sigmoid_a: np.ndarray):
        """用真实数据的残差训练 GP

        GP 学习 Sigmoid 预测与真实参数之间的差异。

        Args:
            X: 归一化的设计点
            target_v: 真实 (或层级贝叶斯估计) 的 v
            target_a: 真实 (或层级贝叶斯估计) 的 a
            sigmoid_v: Sigmoid 模型预测的 v
            sigmoid_a: Sigmoid 模型预测的 a
        """
        self.residual_v = target_v - sigmoid_v
        self.residual_a = target_a - sigmoid_a

        self.gp_v.fit(X, self.residual_v)
        self.gp_a.fit(X, self.residual_a)

        self.X_train = X

    def predict_params(self, X: np.ndarray,
                       sigmoid_v: np.ndarray,
                       sigmoid_a: np.ndarray,
                       return_std: bool = True) -> Dict:
        """预测混合 DDM 参数

        Args:
            X: 归一化设计点
            sigmoid_v: Sigmoid 预测的 v
            sigmoid_a: Sigmoid 预测的 a
            return_std: 是否返回不确定性

        Returns:
            dict: v_mix, a_mix, v_std, a_std 等
        """
        if return_std:
            gp_v_mu, gp_v_std = self.gp_v.predict(X, return_std=True)
            gp_a_mu, gp_a_std = self.gp_a.predict(X, return_std=True)
        else:
            gp_v_mu = self.gp_v.predict(X, return_std=False)[0]
            gp_a_mu = self.gp_a.predict(X, return_std=False)[0]
            gp_v_std = gp_a_std = None

        v_mix = self.w_gp * gp_v_mu + (1 - self.w_gp) * sigmoid_v
        a_mix = self.w_gp * gp_a_mu + (1 - self.w_gp) * sigmoid_a

        return {
            'v_mix': v_mix,
            'a_mix': a_mix,
            'gp_v_residual': gp_v_mu,
            'gp_a_residual': gp_a_mu,
            'v_std': gp_v_std,
            'a_std': gp_a_std,
        }

    def predict_spe_surface(self,
                            P_grid: np.ndarray,
                            T_grid: np.ndarray,
                            W_grid: np.ndarray,
                            sigmoid_model,
                            n_samples: int = 30) -> Dict:
        """预测整个设计空间上的 SPE 响应面

        Args:
            P_grid, T_grid, W_grid: 3D 网格坐标
            sigmoid_model: Sigmoid 模型实例
            n_samples: 每个点的采样次数 (用于蒙特卡洛)

        Returns:
            dict: spe_mean, spe_std, v_self_surface, v_stranger_surface 等
        """
        n_points = len(P_grid)
        gp_samples = np.zeros((n_samples, n_points))
        spe_samples = np.zeros((n_samples, n_points))

        for s in range(n_samples):
            # 随机采样
            v_self = []
            v_stranger = []

            for i in range(n_points):
                Pn, Tn, Wn = (float(P_grid[i]) - 75) / 75, (float(T_grid[i]) - 305) / 295, (float(W_grid[i]) - 850) / 650
                X = np.array([[Pn, Tn, Wn]])

                sv_self = compute_v_s2(T_grid[i], P_grid[i], 1)
                sv_str = compute_v_s2(T_grid[i], P_grid[i], 0)
                sa = compute_a_s2(T_grid[i] + W_grid[i])

                pred_self = self.predict_params(X, np.array([sv_self]), np.array([sa]))
                pred_str = self.predict_params(X, np.array([sv_str]), np.array([sa]))

                v_self.append(pred_self['v_mix'][0])
                v_stranger.append(pred_str['v_mix'][0])

            spe_samples[s] = np.array(v_self) - np.array(v_stranger)

        return {
            'spe_mean': spe_samples.mean(axis=0),
            'spe_std': spe_samples.std(axis=0),
            'spe_ci_lower': np.percentile(spe_samples, 2.5, axis=0),
            'spe_ci_upper': np.percentile(spe_samples, 97.5, axis=0),
        }

    def save(self, base_path: Path):
        """保存混合模型"""
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        self.gp_v.save(base_path.with_name(f'{base_path.stem}_gp_v.pkl'))
        self.gp_a.save(base_path.with_name(f'{base_path.stem}_gp_a.pkl'))

        meta = {
            'w_gp': self.w_gp,
            'seed': self.seed,
        }
        with open(base_path.with_name(f'{base_path.stem}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, base_path: Path) -> 'GPHybridModel':
        """加载混合模型"""
        base_path = Path(base_path)
        with open(base_path.with_name(f'{base_path.stem}_meta.json'), 'r') as f:
            meta = json.load(f)

        model = cls(w_gp=meta['w_gp'], seed=meta['seed'])
        model.gp_v = GPModel.load(base_path.with_name(f'{base_path.stem}_gp_v.pkl'))
        model.gp_a = GPModel.load(base_path.with_name(f'{base_path.stem}_gp_a.pkl'))
        return model


# 从 sigmoid_model 导入需要的函数
compute_v_s2 = None   # placeholder, will be imported at module level


def _lazy_import():
    global compute_v_s2
    if compute_v_s2 is None:
        from .sigmoid_model import compute_v_s2 as _cv2, compute_a_s2 as _ca2
        globals()['compute_v_s2'] = _cv2
        globals()['compute_a_s2'] = _ca2
