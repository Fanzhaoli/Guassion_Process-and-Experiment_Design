"""
生成式模型模块: "实验设计空间 → DDM参数 → 行为数据"

整合 Sigmoid (理论层) + GP (数据驱动残差层) + DDM (决策仿真)
构建完整的生成式模型管道。

核心架构:
  实验设计 Ω(P,T,W,Identity) 
    → S2机制函数 (先验理论)
    → GP残差修正 (非参数学习)
    → 混合DDM参数 (v_mix, a_mix)
    → 试次噪声注入
    → DDM Euler仿真 (带deadline)
    → 行为数据 (RT, response, omission)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import pickle
from datetime import datetime

from .design_space import DesignSpace
from .sigmoid_model import PureSigmoidModel, compute_v_s2, compute_a_s2, v_P_Function
from .gp_model import GPHybridModel, GPModel
from .ddm_engine import simulate_ddm_euler, simulate_ddm_with_deadline, compute_lapse_omission_prob, sample_a_positive
from .ez_diffusion import ez_diffusion, ez_diffusion_from_data, estimate_condition_level_params, subject_level_ez_diffusion


class GenerativeModel:
    """完整生成式模型

    封装 Sigmoid + GP + DDM 的完整生成管道。
    支持:
      - 从设计空间生成模拟数据
      - 用真实数据训练 GP 残差层
      - 模型参数保存/加载
      - 版本化实验管理
    """

    def __init__(self,
                 w_gp: float = 0.5,
                 v_noise: float = 0.3,
                 a_noise: float = 0.25,
                 seed: int = 42,
                 version: str = 'auto'):
        self.w_gp = w_gp
        self.v_noise = v_noise
        self.a_noise = a_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if version == 'auto':
            version = datetime.now().strftime('v%Y%m%d_%H%M%S')
        self.version = version

        self.sigmoid_model = PureSigmoidModel(seed=seed)
        self.gp_hybrid = GPHybridModel(w_gp=w_gp, seed=seed)
        self.design_space = DesignSpace(seed=seed)

        self.is_trained = False

    def generate_dataset(self,
                         design_df: pd.DataFrame,
                         n_subjects: int = 30,
                         trials_per_condition: int = 20) -> pd.DataFrame:
        """从设计空间生成完整的模拟行为数据集

        Args:
            design_df: 设计空间数据框
            n_subjects: 被试数量
            trials_per_condition: 每个条件的试次数

        Returns:
            包含所有中间参数和行为结果的 DataFrame
        """
        identity_levels = ['self', 'stranger']

        rows = []
        subject_id = 1

        for subj_idx in range(1, n_subjects + 1):
            for _, cond in design_df.iterrows():
                P, T, W = int(cond['P']), int(cond['T']), int(cond['W'])
                M = T + W
                condition_id = cond.get('condition_id', f"P{P}_T{T}_W{W}")

                for identity in identity_levels:
                    condition_key = 1 if identity == 'self' else 0

                    for trial in range(trials_per_condition):
                        # S2 机制预测
                        v_s2 = compute_v_s2(T, P, condition_key)
                        a_s2 = compute_a_s2(M)

                        # GP 混合预测
                        Pn, Tn, Wn = self._normalize(P, T, W)
                        X = np.array([[Pn, Tn, Wn]])

                        if self.is_trained:
                            pred = self.gp_hybrid.predict_params(
                                X,
                                np.array([v_s2]),
                                np.array([a_s2]),
                                return_std=False,
                            )
                            v_mean = pred['v_mix'][0]
                            a_mean = pred['a_mix'][0]
                        else:
                            v_mean = v_s2
                            a_mean = a_s2

                        # 试次噪声
                        v_final = self.rng.normal(v_mean, self.v_noise)
                        a_final = sample_a_positive(a_mean, self.a_noise)
                        z_final = a_final / 2.0
                        t0 = 0.2

                        # DDM 仿真
                        deadline = (T + W) / 1000.0
                        lapse_rate = compute_lapse_omission_prob(T, W)

                        result = simulate_ddm_with_deadline(
                            v_final, a_final, z_final, t0,
                            deadline, dt=0.001, lapse_prob=lapse_rate,
                        )

                        rows.append({
                            'subject': subject_id,
                            'P': P, 'T': T, 'W': W, 'M': M,
                            'condition_id': condition_id,
                            'label': identity,
                            'condition_key': condition_key,
                            'v_s2': float(v_s2),
                            'a_s2': float(a_s2),
                            'v_mix': float(v_mean),
                            'a_mix': float(a_mean),
                            'v_final': float(v_final),
                            'a_final': float(a_final),
                            't0': t0,
                            'z': float(z_final),
                            'trial': trial + 1,
                            'RT': result['RT'],
                            'response': result['response'],
                            'omission': result['omission'],
                            'responded': result['responded'],
                            'deadline_reached': result.get('deadline_reached', False),
                            'lapse': result.get('lapse', False),
                        })

                subject_id += 1

        df = pd.DataFrame(rows)
        return df

    def fit_to_real_data(self,
                         real_data: pd.DataFrame,
                         design_df: Optional[pd.DataFrame] = None):
        """用真实数据的条件水平统计训练 GP 残差层

        Args:
            real_data: 真实实验数据 (含 P,T,W,label,RT,response)
            design_df: 设计空间定义 (如果为 None，从 real_data 推导)
        """
        # 估计真实数据的条件水平 DDM 参数
        if design_df is None:
            unique_conds = real_data[['P', 'T', 'W', 'label']].drop_duplicates()
        else:
            unique_conds = design_df.copy()
            if 'label' not in unique_conds.columns:
                labels = []
                for _ in range(len(unique_conds)):
                    labels.extend(['self', 'stranger'])
                unique_conds = pd.concat([
                    unique_conds.assign(label='self'),
                    unique_conds.assign(label='stranger'),
                ], ignore_index=True)

        X_train = []
        target_v = []
        target_a = []
        sigmoid_v = []
        sigmoid_a = []

        for _, row in unique_conds.iterrows():
            P, T, W = int(row['P']), int(row['T']), int(row['W'])
            label = row['label']
            condition_key = 1 if label == 'self' else 0
            M = T + W

            # 筛选该条件的真实试次
            mask = ((real_data['P'] == P) & (real_data['T'] == T) &
                    (real_data['W'] == W) & (real_data['label'] == label))

            if 'M' in real_data.columns:
                mask = mask & (real_data['M'] == M)

            subset = real_data[mask]

            if len(subset) < 5:
                continue

            # EZ-diffusion 估计
            ez_params = ez_diffusion_from_data(subset)
            if np.isnan(ez_params['v']):
                continue

            # 归一化
            Pn, Tn, Wn = self._normalize(P, T, W)
            X_train.append([Pn, Tn, Wn])

            # Sigmoid 预测
            sv = compute_v_s2(T, P, condition_key)
            sa = compute_a_s2(M)

            target_v.append(ez_params['v'])
            target_a.append(ez_params['a'])
            sigmoid_v.append(sv)
            sigmoid_a.append(sa)

        if len(X_train) < 5:
            raise ValueError(f"有效条件数不足 ({len(X_train)}), 无法训练GP模型")

        X_train = np.array(X_train)
        target_v = np.array(target_v)
        target_a = np.array(target_a)
        sigmoid_v = np.array(sigmoid_v)
        sigmoid_a = np.array(sigmoid_a)

        self.gp_hybrid.fit_residuals(X_train, target_v, target_a, sigmoid_v, sigmoid_a)
        self.is_trained = True

    def predict_condition(self, P: int, T: int, W: int, label: str) -> Dict:
        """预测特定实验条件下的 DDM 参数和行为

        Args:
            P, T, W: 设计参数
            label: 'self' 或 'stranger'

        Returns:
            包含预测参数的字典
        """
        condition_key = 1 if label == 'self' else 0
        M = T + W

        v_s2 = compute_v_s2(T, P, condition_key)
        a_s2 = compute_a_s2(M)

        Pn, Tn, Wn = self._normalize(P, T, W)
        X = np.array([[Pn, Tn, Wn]])

        if self.is_trained:
            pred = self.gp_hybrid.predict_params(
                X, np.array([v_s2]), np.array([a_s2]), return_std=True,
            )
            v_mix = float(pred['v_mix'][0])
            a_mix = float(pred['a_mix'][0])
            v_std = float(pred['v_std'][0]) if pred['v_std'] is not None else None
            a_std = float(pred['a_std'][0]) if pred['a_std'] is not None else None
        else:
            v_mix = float(v_s2)
            a_mix = float(a_s2)
            v_std = a_std = None

        return {
            'P': P, 'T': T, 'W': W, 'M': M, 'label': label,
            'v_s2': float(v_s2), 'a_s2': float(a_s2),
            'v_mix': v_mix, 'a_mix': a_mix,
            'v_std': v_std, 'a_std': a_std,
            't0': 0.2, 'z': a_mix / 2.0,
        }

    def explore_spe_boundaries(self,
                               gp_explore_resolution: int = 20) -> Dict:
        """探索 SPE 效应的设计空间边界

        识别 SPE 产生、增强或消失的实验设计区域。

        Returns:
            dict: 包含边界分析结果
        """
        P_vals = np.linspace(0, 150, gp_explore_resolution)
        T_vals = np.linspace(10, 600, gp_explore_resolution)
        W_vals = np.linspace(200, 1500, gp_explore_resolution)

        # 固定 W 为中等值, 探索 P×T 平面
        W_fixed = 750
        results = []

        for P in P_vals:
            for T in T_vals:
                pred_self = self.predict_condition(int(P), int(T), int(W_fixed), 'self')
                pred_stranger = self.predict_condition(int(P), int(T), int(W_fixed), 'stranger')

                v_diff = pred_self['v_mix'] - pred_stranger['v_mix']

                results.append({
                    'P': int(P), 'T': int(T), 'W': int(W_fixed),
                    'v_self': pred_self['v_mix'],
                    'v_stranger': pred_stranger['v_mix'],
                    'v_diff': v_diff,
                    'v_std_self': pred_self['v_std'],
                    'v_std_stranger': pred_stranger['v_std'],
                    'spe_region': 'strong' if v_diff > 0.3 else
                                  'moderate' if v_diff > 0.1 else
                                  'weak' if v_diff > 0 else 'none',
                })

        boundary_df = pd.DataFrame(results)

        return {
            'boundary_df': boundary_df,
            'strong_spe_region': boundary_df[boundary_df['spe_region'] == 'strong'],
            'spe_disappearance': boundary_df[boundary_df['spe_region'] == 'none'],
        }

    def save(self, base_dir: Path, version: str = None):
        """保存完整模型"""
        if version is None:
            version = self.version

        base_dir = Path(base_dir) / version
        base_dir.mkdir(parents=True, exist_ok=True)

        model_path = base_dir / 'generative_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"模型已保存到: {model_path}")

    @classmethod
    def load(cls, path: Path) -> 'GenerativeModel':
        """加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _normalize(self, P, T, W):
        """归一化到约[-1, 1]"""
        P_norm = (float(P) - 75.0) / 75.0
        T_norm = (float(T) - 305.0) / 295.0
        W_norm = (float(W) - 850.0) / 650.0
        return P_norm, T_norm, W_norm


class ModelComparison:
    """模型比较框架

    比较纯 Sigmoid 模型 vs GP+Sigmoid 混合模型的预测性能。
    """

    def __init__(self, pure_model: PureSigmoidModel, hybrid_model: GenerativeModel):
        self.pure_model = pure_model
        self.hybrid_model = hybrid_model

    def compare_condition_predictions(self,
                                       design_df: pd.DataFrame,
                                       real_data: pd.DataFrame) -> pd.DataFrame:
        """比较两个模型对各条件的预测误差

        Returns:
            包含 RMSE 比较的 DataFrame
        """
        results = []

        for _, cond in design_df.iterrows():
            P, T, W = int(cond['P']), int(cond['T']), int(cond['W'])

            for label in ['self', 'stranger']:
                condition_key = 1 if label == 'self' else 0

                # 真实数据统计
                mask = ((real_data['P'] == P) & (real_data['T'] == T) &
                        (real_data['W'] == W) & (real_data['label'] == label))
                subset = real_data[mask]

                if len(subset) < 5:
                    continue

                valid = subset[subset['RT'].notna() & (subset['RT'] > 0)]
                if len(valid) < 3:
                    continue

                real_rt_mean = valid['RT'].mean()
                real_acc = (valid['response'] == 1).mean()

                # 纯 Sigmoid 预测
                pure_v, pure_a, _, _ = self.pure_model.predict_params(
                    np.array([T]), np.array([P]), np.array([W]),
                    np.array([condition_key])
                )
                pure_rt_pred = pure_a[0] / (2 * abs(pure_v[0])) if pure_v[0] != 0 else np.nan

                # 混合模型预测
                hybrid_pred = self.hybrid_model.predict_condition(P, T, W, label)

                results.append({
                    'P': P, 'T': T, 'W': W, 'label': label,
                    'real_rt_mean': real_rt_mean,
                    'real_acc': real_acc,
                    'n_trials': len(valid),
                    'pure_rt_pred': pure_rt_pred,
                    'hybrid_rt_pred': hybrid_pred['v_mix'],
                    'pure_rmse': np.sqrt((real_rt_mean - pure_rt_pred)**2) if not np.isnan(pure_rt_pred) else np.nan,
                    'hybrid_rmse': np.sqrt((real_rt_mean - hybrid_pred['v_mix'])**2),
                })

        comp_df = pd.DataFrame(results)

        # 汇总
        comp_df['improvement'] = comp_df['pure_rmse'] - comp_df['hybrid_rmse']

        return comp_df

    def summary(self, comp_df: pd.DataFrame) -> Dict:
        """生成模型比较摘要"""
        valid = comp_df.dropna(subset=['pure_rmse', 'hybrid_rmse'])

        return {
            'n_conditions': len(valid),
            'pure_mean_rmse': float(valid['pure_rmse'].mean()),
            'hybrid_mean_rmse': float(valid['hybrid_rmse'].mean()),
            'mean_improvement': float(valid['improvement'].mean()),
            'hybrid_better_pct': float((valid['improvement'] > 0).mean() * 100),
            'improvement_t_stat': float(stats.ttest_rel(
                valid['pure_rmse'], valid['hybrid_rmse']
            ).statistic) if len(valid) > 3 else np.nan,
            'improvement_p_value': float(stats.ttest_rel(
                valid['pure_rmse'], valid['hybrid_rmse']
            ).pvalue) if len(valid) > 3 else np.nan,
        }

from scipy import stats
