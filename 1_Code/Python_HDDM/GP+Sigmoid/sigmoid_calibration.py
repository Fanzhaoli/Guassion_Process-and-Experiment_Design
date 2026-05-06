import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
HDDM_PARAMS_PATH = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces" / "all_groups_ddm_params.csv"
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def v_P_Function(P, P1=4, k_min=0.1, k_max=0.05, gamma=0.2, P0=32):
    P = np.asarray(P, dtype=float)
    k = k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))
    return 1.0 / (1.0 + np.exp(-k * (P - P1)))


def compute_v_s2(T, P, condition_key, alaph1=1.5, alaph2=-0.4, gamma=0.2,
                 T_0=100.0, k_T=0.01, base_scale=3.0):
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    v_T = 1.0 / (1.0 + np.exp(-k_T * (T - T_0)))
    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=gamma, P0=32)
    v_0 = v_T * v_P * base_scale
    condition_key = np.asarray(condition_key)
    v_1 = np.where(condition_key == 1, v_0 * (1 + alaph1), v_0 * (1 + alaph2))
    return v_1


def compute_a_s2(M, beta1=0.2, beta2=0.0, k=0.01, M_0=600.0, base_scale=3.0):
    M = np.asarray(M, dtype=float)
    a_0 = 1.0 / (1.0 + np.exp(-k * (M - M_0))) * base_scale
    a_1 = np.where(M > M_0, a_0 * (1 + beta1), a_0 * (1 + beta2))
    return a_1


def objective(params, df_real):
    alaph1, alaph2, beta1, beta2, gamma, base_scale_v, base_scale_a = params

    rmse_v_self = 0.0
    rmse_v_stranger = 0.0
    rmse_a = 0.0
    n = len(df_real)

    for _, row in df_real.iterrows():
        P_val = row["P"]
        T_val = row["T_ms"]
        W_val = row["W_ms"]
        M_val = T_val + W_val

        v_self_sig = compute_v_s2(T_val, P_val, 1, alaph1=alaph1, alaph2=alaph2,
                                  gamma=gamma, base_scale=base_scale_v)
        v_stranger_sig = compute_v_s2(T_val, P_val, 0, alaph1=alaph1, alaph2=alaph2,
                                      gamma=gamma, base_scale=base_scale_v)
        a_sig = compute_a_s2(M_val, beta1=beta1, beta2=beta2, base_scale=base_scale_a)

        rmse_v_self += (float(v_self_sig) - row["v_self_mean"]) ** 2
        rmse_v_stranger += (float(v_stranger_sig) - row["v_stranger_mean"]) ** 2
        rmse_a += (float(a_sig) - row["a_mean"]) ** 2

    total_rmse = np.sqrt((rmse_v_self + rmse_v_stranger + rmse_a) / (3 * n))
    return total_rmse


print("=" * 60)
print("Sigmoid 参数校准：最小化 Sigmoid 预测与 HDDM 真实参数的 RMSE")
print("=" * 60)

df_real = pd.read_csv(HDDM_PARAMS_PATH)
print(f"\n加载 HDDM 参数: {len(df_real)} 个条件")
print(df_real[["group_id", "P", "T_ms", "W_ms", "v_self_mean", "v_stranger_mean", "a_mean"]].round(3).to_string(index=False))

bounds = [
    (0.1, 3.0),     # alaph1
    (-2.0, 1.0),    # alaph2
    (-1.0, 2.0),    # beta1
    (-1.0, 1.0),    # beta2
    (0.01, 1.0),    # gamma
    (1.0, 10.0),    # base_scale_v
    (1.0, 10.0),    # base_scale_a
]

print("\n开始差分进化优化...")
result = differential_evolution(
    objective, bounds, args=(df_real,),
    maxiter=500, seed=42, popsize=15, tol=1e-8
)

alaph1_opt, alaph2_opt, beta1_opt, beta2_opt, gamma_opt, bs_v_opt, bs_a_opt = result.x

print(f"\n{'='*40}")
print("优化结果")
print(f"{'='*40}")
print(f"  alaph1 (self增强):      {alaph1_opt:.4f}")
print(f"  alaph2 (stranger调制):   {alaph2_opt:.4f}")
print(f"  beta1 (高M边界):         {beta1_opt:.4f}")
print(f"  beta2 (低M边界):         {beta2_opt:.4f}")
print(f"  gamma (练习效应速率):     {gamma_opt:.4f}")
print(f"  base_scale_v:            {bs_v_opt:.4f}")
print(f"  base_scale_a:            {bs_a_opt:.4f}")
print(f"  最终 RMSE:               {result.fun:.6f}")
print(f"  收敛成功:               {result.success}")

calibrated_params = {
    "alaph1": alaph1_opt, "alaph2": alaph2_opt,
    "beta1": beta1_opt, "beta2": beta2_opt,
    "gamma": gamma_opt, "base_scale_v": bs_v_opt, "base_scale_a": bs_a_opt,
    "final_rmse": result.fun, "converged": result.success,
}

predictions = []
for _, row in df_real.iterrows():
    P_val = row["P"]
    T_val = row["T_ms"]
    W_val = row["W_ms"]
    M_val = T_val + W_val

    v_self_sig = float(compute_v_s2(T_val, P_val, 1, **{k: v for k, v in calibrated_params.items()
                                                         if k in ["alaph1", "alaph2", "gamma"]},
                                    base_scale=bs_v_opt))
    v_stranger_sig = float(compute_v_s2(T_val, P_val, 0, **{k: v for k, v in calibrated_params.items()
                                                             if k in ["alaph1", "alaph2", "gamma"]},
                                        base_scale=bs_v_opt))
    a_sig = float(compute_a_s2(M_val, beta1=beta1_opt, beta2=beta2_opt, base_scale=bs_a_opt))

    predictions.append({
        "group_id": row["group_id"], "P": P_val, "T_ms": T_val, "W_ms": W_val, "M_ms": M_val,
        "v_self_real": row["v_self_mean"], "v_stranger_real": row["v_stranger_mean"],
        "a_real": row["a_mean"], "t_real": row["t_mean"], "z_real": row["z_mean"],
        "v_self_sig": v_self_sig, "v_stranger_sig": v_stranger_sig, "a_sig": a_sig,
        "v_self_residual": row["v_self_mean"] - v_self_sig,
        "v_stranger_residual": row["v_stranger_mean"] - v_stranger_sig,
        "a_residual": row["a_mean"] - a_sig,
    })

df_pred = pd.DataFrame(predictions)
df_pred.to_csv(OUT_DIR / "sigmoid_calibration_results.csv", index=False)

print(f"\n预测 vs 真实:")
cols = ["group_id", "P", "T_ms", "W_ms", "v_self_real", "v_self_sig", "v_stranger_real",
        "v_stranger_sig", "a_real", "a_sig"]
print(df_pred[cols].round(3).to_string(index=False))

print(f"\n校准结果已保存到: {OUT_DIR / 'sigmoid_calibration_results.csv'}")

params_df = pd.DataFrame([calibrated_params])
params_df.to_csv(OUT_DIR / "sigmoid_calibrated_params.csv", index=False)
print(f"校准参数已保存到: {OUT_DIR / 'sigmoid_calibrated_params.csv'}")
print("=" * 60)
