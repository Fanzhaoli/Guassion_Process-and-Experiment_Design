import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import pickle
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]


class GPSigmoidHybridModel:
    """GP+Sigmoid 混合生成模型

    Sigmoid 函数提供理论先验，GP 学习 Sigmoid 预测与真实 HDDM 参数之间的残差。
    最终预测: DDM参数 = Sigmoid预测 + GP残差预测

    对每个 DDM 参数 (v_self, v_stranger, a, t, z) 独立训练 GP 残差模型。
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

        self.gp_v_self = GaussianProcessRegressor(
            kernel=base_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=seed
        )
        self.gp_v_stranger = GaussianProcessRegressor(
            kernel=base_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=seed + 1
        )
        self.gp_a = GaussianProcessRegressor(
            kernel=base_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=seed + 2
        )
        self.gp_t = GaussianProcessRegressor(
            kernel=base_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=seed + 3
        )
        self.gp_z = GaussianProcessRegressor(
            kernel=base_kernel, normalize_y=True, n_restarts_optimizer=10, random_state=seed + 4
        )

        self._fitted = False
        self._X_train = None
        self._calib_params = None

    @staticmethod
    def normalize_PTW(P, T, W):
        P_min, P_max = 0.0, 120.0
        T_min, T_max = 30.0, 500.0
        W_min, W_max = 300.0, 1500.0
        P_n = (np.asarray(P, dtype=float) - (P_min + P_max) / 2) / ((P_max - P_min) / 2)
        T_n = (np.asarray(T, dtype=float) - (T_min + T_max) / 2) / ((T_max - T_min) / 2)
        W_n = (np.asarray(W, dtype=float) - (W_min + W_max) / 2) / ((W_max - W_min) / 2)
        return np.column_stack([P_n, T_n, W_n])

    @staticmethod
    def sigmoid_v_prediction(T, P, condition_key, calib_params):
        cp = calib_params
        T_arr = np.asarray(T, dtype=float)
        P_arr = np.asarray(P, dtype=float)

        def v_P_func(P_vals):
            k = 0.1 + (0.05 - 0.1) / (1 + np.exp(-cp.get("gamma", 0.2) * (P_vals - 32)))
            return 1.0 / (1.0 + np.exp(-k * (P_vals - 4)))

        v_T = 1.0 / (1.0 + np.exp(-0.01 * (T_arr - 100.0)))
        v_P = v_P_func(P_arr)
        v_0 = v_T * v_P * cp.get("base_scale_v", 3.0)

        cond = np.asarray(condition_key)
        return np.where(cond == 1, v_0 * (1 + cp.get("alaph1", 1.5)),
                        v_0 * (1 + cp.get("alaph2", -0.4)))

    @staticmethod
    def sigmoid_a_prediction(M, calib_params):
        cp = calib_params
        M_arr = np.asarray(M, dtype=float)
        a_0 = 1.0 / (1.0 + np.exp(-0.01 * (M_arr - 600.0))) * cp.get("base_scale_a", 3.0)
        return np.where(M_arr > 600, a_0 * (1 + cp.get("beta1", 0.2)),
                        a_0 * (1 + cp.get("beta2", 0.0)))

    def fit(self, df_real, calib_params=None):
        """训练 GP+Sigmoid 混合模型

        Args:
            df_real: DataFrame with columns [P, T_ms, W_ms, v_self_mean, v_stranger_mean,
                     a_mean, t_mean, z_mean] (8 rows for 8 conditions)
            calib_params: dict of calibrated Sigmoid parameters, or None for defaults
        """
        if calib_params is None:
            calib_params = {"alaph1": 1.5, "alaph2": -0.4, "beta1": 0.2, "beta2": 0.0,
                            "gamma": 0.2, "base_scale_v": 3.0, "base_scale_a": 3.0}
        self._calib_params = calib_params

        X = self.normalize_PTW(df_real["P"].values, df_real["T_ms"].values, df_real["W_ms"].values)
        self._X_train = X
        n = len(df_real)

        M_vals = df_real["T_ms"].values + df_real["W_ms"].values

        v_self_real = df_real["v_self_mean"].values
        v_stranger_real = df_real["v_stranger_mean"].values
        a_real = df_real["a_mean"].values
        t_real = df_real["t_mean"].values
        z_real = df_real["z_mean"].values

        v_self_sig = self.sigmoid_v_prediction(df_real["T_ms"].values, df_real["P"].values,
                                               np.ones(n), calib_params)
        v_stranger_sig = self.sigmoid_v_prediction(df_real["T_ms"].values, df_real["P"].values,
                                                   np.zeros(n), calib_params)
        a_sig = self.sigmoid_a_prediction(M_vals, calib_params)

        t_sig = np.full(n, np.mean(t_real))
        z_sig = np.full(n, np.mean(z_real))

        self.gp_v_self.fit(X, v_self_real - v_self_sig)
        self.gp_v_stranger.fit(X, v_stranger_real - v_stranger_sig)
        self.gp_a.fit(X, a_real - a_sig)
        self.gp_t.fit(X, t_real - t_sig)
        self.gp_z.fit(X, z_real - z_sig)

        self._fitted = True
        return self

    def predict(self, P, T, W, condition_key, return_std=False):
        """预测 DDM 参数

        Args:
            P, T, W: 实验设计参数 (标量或数组)
            condition_key: 1=self, 0=stranger
            return_std: 是否返回 GP 预测标准差

        Returns:
            如果 return_std=False: (v, a, t, z) 四个参数
            如果 return_std=True:  ((v, a, t, z), (v_std, a_std, t_std, z_std))
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        W_arr = np.atleast_1d(np.asarray(W, dtype=float))
        cond_arr = np.atleast_1d(np.asarray(condition_key))
        n = len(P_arr)

        X = self.normalize_PTW(P_arr, T_arr, W_arr)
        M_arr = T_arr + W_arr

        v_sig = self.sigmoid_v_prediction(T_arr, P_arr, cond_arr, self._calib_params)
        a_sig = self.sigmoid_a_prediction(M_arr, self._calib_params)
        t_sig = np.full(n, self._calib_params.get("t_baseline", 0.35))
        z_sig = np.full(n, self._calib_params.get("z_baseline", 0.55))

        v_self_residual_pred, v_self_residual_std = self.gp_v_self.predict(X, return_std=True)
        v_stranger_residual_pred, v_stranger_residual_std = self.gp_v_stranger.predict(X, return_std=True)
        a_pred, a_std = self.gp_a.predict(X, return_std=True)
        t_pred, t_std = self.gp_t.predict(X, return_std=True)
        z_pred, z_std = self.gp_z.predict(X, return_std=True)

        v_self_sig = self.sigmoid_v_prediction(T_arr, P_arr, np.ones(n), self._calib_params)
        v_stranger_sig = self.sigmoid_v_prediction(T_arr, P_arr, np.zeros(n), self._calib_params)
        v_final = np.where(
            cond_arr == 1,
            v_self_sig + v_self_residual_pred,
            v_stranger_sig + v_stranger_residual_pred,
        )
        v_std = np.where(cond_arr == 1, v_self_residual_std, v_stranger_residual_std)

        a_final = a_sig + a_pred
        t_final = t_sig + t_pred
        z_final = z_sig + z_pred

        a_final = np.maximum(a_final, 0.1)
        t_final = np.maximum(t_final, 0.01)
        z_final = np.clip(z_final, 0.01, a_final - 0.01)

        if return_std:
            return (v_final, a_final, t_final, z_final), (v_std, a_std, t_std, z_std)
        return v_final, a_final, t_final, z_final

    def predict_params_full(self, P, T, W):
        """同时预测 self 和 stranger 的完整 DDM 参数"""
        v_self, a_self, t_self, z_self = self.predict(P, T, W, 1)
        v_stranger, a_stranger, t_stranger, z_stranger = self.predict(P, T, W, 0)
        return v_self, v_stranger, a_self, t_self, z_self

    def predict_with_uncertainty(self, P, T, W, condition_key):
        """预测并返回不确定性"""
        (v, a, t, z), (v_std, a_std, t_std, z_std) = self.predict(P, T, W, condition_key, return_std=True)
        return {"v": v, "a": a, "t": t, "z": z,
                "v_std": v_std, "a_std": a_std, "t_std": t_std, "z_std": z_std}

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "seed": self.seed,
                "_fitted": self._fitted,
                "_X_train": self._X_train,
                "_calib_params": self._calib_params,
                "gp_v_self": self.gp_v_self,
                "gp_v_stranger": self.gp_v_stranger,
                "gp_a": self.gp_a,
                "gp_t": self.gp_t,
                "gp_z": self.gp_z,
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(seed=data["seed"])
        model._fitted = data["_fitted"]
        model._X_train = data["_X_train"]
        model._calib_params = data["_calib_params"]
        model.gp_v_self = data["gp_v_self"]
        model.gp_v_stranger = data["gp_v_stranger"]
        model.gp_a = data["gp_a"]
        model.gp_t = data["gp_t"]
        model.gp_z = data["gp_z"]
        return model


def train_model_from_data(df_real, calib_params=None):
    """便捷函数：从真实 HDDM 参数训练 GP+Sigmoid 模型"""
    model = GPSigmoidHybridModel(seed=42)
    model.fit(df_real, calib_params=calib_params)

    if calib_params is not None:
        t_baseline = np.mean(df_real["t_mean"])
        z_baseline = np.mean(df_real["z_mean"])
        model._calib_params["t_baseline"] = t_baseline
        model._calib_params["z_baseline"] = z_baseline

    return model
