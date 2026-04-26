import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# =========================
# 路径与全局配置
# =========================
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
REAL_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined.csv"
GEN_PATH = BASE_DIR / "2_Data" / "Generate_Data" / "Generate_Data_v2.4.5_checks" / "gp_ddm_v2.4.5_large.csv"
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Generate_Data_v2.4.5_checks"
FIG_DIR = BASE_DIR / "3_Figures" / "Generate_Data_v2.4.5_checks" / "Model_Comparison_v2.4.5"
MODEL_DIR = OUT_DIR / "models_v2.4.5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# 运行配置：可以按需要在这里切换快速版/完整版
FAST_MODE = True
GP_INCLUDE_LABEL = True

if FAST_MODE:
    FIT_TRIAL_CAP = 40
    PRED_TRIAL_CAP = 60
    N_REP_PRED = 2
    N_RANDOM_SEARCH = 40
else:
    FIT_TRIAL_CAP = 120
    PRED_TRIAL_CAP = 240
    N_REP_PRED = 5
    N_RANDOM_SEARCH = 180


@dataclass
class SimConfig:
    v_noise: float = 0.55
    a_noise: float = 0.40
    lapse_max: float = 0.22
    t0: float = 0.20


# =========================
# 与 v2.4.5 一致的机制函数
# =========================
def k_P(P, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    return k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))


def v_P_Function(P, P1=4, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    k = k_P(P, k_min, k_max, gamma, P0)
    return 1 / (1 + np.exp(-k * (P - P1)))


def compute_v_s2(T_ms, P, condition_key, alpha1=1.5, alpha2=-0.4, gamma=0.2):
    T_0 = 100
    k_T = 0.01
    v_T = 1 / (1 + np.exp(-k_T * (T_ms - T_0)))
    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=gamma, P0=32)
    v_0 = v_T * v_P * 3
    if condition_key == 1:
        return v_0 * (1 + alpha1)
    return v_0 * (1 + alpha2)


def compute_a_s2(M_ms, beta1=0.2, beta2=0.0, k=0.01, M_0=600):
    a_0 = 1 / (1 + np.exp(-k * (M_ms - M_0))) * 3
    if M_ms > 600:
        return a_0 * (1 + beta1)
    return a_0 * (1 + beta2)


def normalize_PTW_to_unit(P, T_ms, W_ms):
    P_norm = (P - 75.0) / 75.0
    T_norm = (T_ms - 305.0) / 295.0
    W_norm = (W_ms - 850.0) / 650.0
    return P_norm, T_norm, W_norm


def compute_lapse_omission_prob(T_ms, W_ms, lapse_max=0.35, T_mid=80, W_mid=600, T_scale=25, W_scale=120):
    t_term = 1.0 / (1.0 + np.exp((T_ms - T_mid) / T_scale))
    w_term = 1.0 / (1.0 + np.exp((W_ms - W_mid) / W_scale))
    return float(lapse_max * t_term * w_term)


def sample_a_positive(rng, a_mix, a_noise, a_floor=0.03, max_resample=30):
    for _ in range(max_resample):
        a_candidate = rng.normal(a_mix, a_noise)
        if a_candidate > a_floor:
            return a_candidate
    return max(a_floor + abs(rng.normal(0, a_noise * 0.25)), a_mix * 0.5, a_floor)


def simulate_ddm_with_deadline(rng, v, a, z, t0, deadline_s, dt=0.001):
    decision_budget = deadline_s - t0
    if np.isnan(decision_budget) or decision_budget <= dt:
        return np.nan, 0, 1, "deadline"

    x = float(z)
    time = 0.0
    max_steps = int(decision_budget / dt)
    for _ in range(max_steps):
        dx = v * dt + np.sqrt(dt) * rng.normal()
        x += dx
        time += dt
        if x >= a:
            return t0 + time, 1, 0, "upper"
        if x <= 0:
            return t0 + time, 2, 0, "lower"
    return np.nan, 0, 1, "deadline"


# =========================
# 数据处理
# =========================
def get_match_key(subject_id):
    keys = ["f", "j", "j", "f"]
    return keys[(int(subject_id) - 1) % 4]


def load_real_matching_data():
    real = pd.read_csv(REAL_PATH)
    real = real.rename(columns={"subjectID": "subject", "Label": "label", "Correct": "correct", "RT": "rt_s"})
    real["label"] = real["label"].astype(str).str.lower()
    real["shape"] = real["Shape"].astype(str).str.lower()
    real["CorrectKey"] = real["CorrectKey"].astype(str).str.lower()
    real["Response"] = real["Response"].astype(str).str.lower()
    real["subject_uid"] = real["GroupInfo"].astype(str)

    real["match_key"] = real["subject"].apply(get_match_key)
    real["matching"] = np.where(real["CorrectKey"] == real["match_key"], "Matching", "NonMatching")
    real["T_ms"] = pd.to_numeric(real["T"], errors="coerce") * 1000.0
    real["W_ms"] = pd.to_numeric(real["W"], errors="coerce") * 1000.0
    real["rt_s"] = pd.to_numeric(real["rt_s"], errors="coerce")
    real["correct"] = pd.to_numeric(real["correct"], errors="coerce").fillna(0).astype(int)
    real["responded"] = (~real["rt_s"].isna()) & (real["rt_s"] > 0)
    real["omission"] = (~real["responded"]).astype(int)

    if "stage" in real.columns:
        real = real[real["stage"].astype(str).str.lower() == "formal"].copy()

    real = real[real["matching"] == "Matching"].copy()

    cond = (
        real[["P", "T_ms", "W_ms"]]
        .drop_duplicates()
        .sort_values(["P", "T_ms", "W_ms"])
        .reset_index(drop=True)
    )
    cond["condition_id"] = [f"C{i+1}" for i in range(len(cond))]
    real = real.merge(cond, on=["P", "T_ms", "W_ms"], how="left")

    return real, cond


def load_generated_with_condition_mapping(cond):
    gen = pd.read_csv(GEN_PATH)
    gen["subject_uid"] = gen["subject"].astype(str)
    gen["label"] = gen["label"].astype(str).str.lower()
    gen["T_ms"] = pd.to_numeric(gen["T"], errors="coerce")
    gen["W_ms"] = pd.to_numeric(gen["W"], errors="coerce")
    gen["rt_s"] = pd.to_numeric(gen["RT"], errors="coerce")
    gen["correct"] = pd.to_numeric(gen["correct"], errors="coerce").fillna(0).astype(int)
    gen["responded"] = pd.to_numeric(gen["responded"], errors="coerce").fillna(0).astype(int)
    gen["omission"] = pd.to_numeric(gen["omission"], errors="coerce").fillna(0).astype(int)

    scale_P = max(cond["P"].max() - cond["P"].min(), 1)
    scale_T = max(cond["T_ms"].max() - cond["T_ms"].min(), 1)
    scale_W = max(cond["W_ms"].max() - cond["W_ms"].min(), 1)

    sub_cond = gen[["subject", "P", "T_ms", "W_ms"]].drop_duplicates().copy()

    def nearest_real_condition(row):
        d = (
            ((cond["P"] - row["P"]) / scale_P) ** 2
            + ((cond["T_ms"] - row["T_ms"]) / scale_T) ** 2
            + ((cond["W_ms"] - row["W_ms"]) / scale_W) ** 2
        )
        idx = d.idxmin()
        return cond.loc[idx, ["condition_id", "P", "T_ms", "W_ms"]]

    assigned = sub_cond.apply(nearest_real_condition, axis=1)
    assigned = pd.concat([sub_cond[["subject"]].reset_index(drop=True), assigned.reset_index(drop=True)], axis=1)
    assigned = assigned.rename(columns={"P": "P_target", "T_ms": "T_ms_target", "W_ms": "W_ms_target"})

    gen = gen.merge(assigned, on="subject", how="left")
    return gen


# =========================
# 统计量与 EZ
# =========================
def summarize_behavior(df, p_col="P", t_col="T_ms", w_col="W_ms"):
    grouped = (
        df.groupby(["condition_id", p_col, t_col, w_col, "label"], as_index=False)
        .agg(
            n_trials=("condition_id", "size"),
            n_responded=("responded", "sum"),
            acc=("correct", "mean"),
            omit_rate=("omission", "mean"),
        )
    )

    rt_med = (
        df[df["responded"] == 1]
        .groupby(["condition_id", p_col, t_col, w_col, "label"], as_index=False)
        .agg(rt_median=("rt_s", "median"))
    )
    out = grouped.merge(rt_med, on=["condition_id", p_col, t_col, w_col, "label"], how="left")
    return out


def ez_diffusion_from_group(df_group, rt_col="rt_s", correct_col="correct", s=0.1):
    n_total = len(df_group)
    if n_total < 4:
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})

    pc = (df_group[correct_col].sum() + 0.5) / (n_total + 1.0)
    pc = float(np.clip(pc, 1e-4, 1 - 1e-4))

    correct_rt = df_group.loc[(df_group[correct_col] == 1) & df_group[rt_col].notna(), rt_col]
    if len(correct_rt) < 3:
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})

    mrt = correct_rt.mean()
    vrt = correct_rt.var(ddof=1)
    if (pc <= 0.5) or (vrt <= 0) or np.isnan(vrt):
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})

    logit_p = np.log(pc / (1 - pc))
    x = logit_p * (pc**2 * logit_p - pc * logit_p + pc - 0.5) / vrt
    if x <= 0 or np.isnan(x):
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})

    v = s * np.power(x, 0.25)
    if v == 0 or np.isnan(v):
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})

    a = (s**2) * logit_p / v
    y = (-v * a) / (s**2)
    mdt = (a / (2 * v)) * ((1 - np.exp(y)) / (1 + np.exp(y)))
    ter = mrt - mdt
    z = a / 2.0

    valid = int((not np.any(np.isnan([v, a, ter, z]))) and (a > 0) and (ter >= 0))
    if not valid:
        return pd.Series({"v_est": np.nan, "a_est": np.nan, "t_est": np.nan, "z_est": np.nan, "valid_ez": 0})
    return pd.Series({"v_est": v, "a_est": a, "t_est": ter, "z_est": z, "valid_ez": 1})


def estimate_condition_ez(df, p_col="P", t_col="T_ms", w_col="W_ms"):
    rows = []
    for keys, g in df.groupby(["condition_id", p_col, t_col, w_col, "label"]):
        condition_id, p_val, t_val, w_val, label = keys
        ez_row = ez_diffusion_from_group(g)
        rows.append(
            {
                "condition_id": condition_id,
                "P": float(p_val),
                "T_ms": float(t_val),
                "W_ms": float(w_val),
                "label": str(label),
                "v_real": ez_row["v_est"],
                "a_real": ez_row["a_est"],
                "t_real": ez_row["t_est"],
                "z_real": ez_row["z_est"],
                "valid_ez": ez_row["valid_ez"],
            }
        )
    return pd.DataFrame(rows)


# =========================
# 模型与仿真
# =========================
def condition_label_to_key(label):
    return 1 if str(label).lower() == "self" else 0


def build_gp_features(P, T_ms, W_ms, label):
    Pn, Tn, Wn = normalize_PTW_to_unit(P, T_ms, W_ms)
    if GP_INCLUDE_LABEL:
        return [Pn, Tn, Wn, float(condition_label_to_key(label))]
    return [Pn, Tn, Wn]


def simulate_for_conditions(cond_df, cfg, mode="pure", gp_models=None, pure_params=None, n_rep=1, seed=42):
    rows = []
    for rep_idx in range(n_rep):
        rng = np.random.default_rng(seed + rep_idx)
        for _, row in cond_df.iterrows():
            P = float(row["P"])
            T_ms = float(row["T_ms"])
            W_ms = float(row["W_ms"])
            M_ms = T_ms + W_ms
            label = str(row["label"])
            condition_key = condition_label_to_key(label)
            n_trials = int(row["n_trials"])

            if mode == "pure":
                alpha1, alpha2, gamma, beta1, beta2 = pure_params
                v_base = compute_v_s2(T_ms, P, condition_key, alpha1=alpha1, alpha2=alpha2, gamma=gamma)
                a_base = compute_a_s2(M_ms, beta1=beta1, beta2=beta2)
            else:
                alpha1, alpha2, gamma, beta1, beta2 = pure_params
                v_s2 = compute_v_s2(T_ms, P, condition_key, alpha1=alpha1, alpha2=alpha2, gamma=gamma)
                a_s2 = compute_a_s2(M_ms, beta1=beta1, beta2=beta2)
                X = np.array([build_gp_features(P, T_ms, W_ms, label)])
                v_res = gp_models["gp_v"].predict(X)[0]
                a_res = gp_models["gp_a"].predict(X)[0]
                v_base = v_s2 + v_res
                a_base = a_s2 + a_res

            for _ in range(n_trials):
                v_final = rng.normal(v_base, cfg.v_noise)
                a_final = sample_a_positive(rng, a_base, cfg.a_noise, a_floor=0.03)
                z_final = a_final / 2.0
                p_lapse = compute_lapse_omission_prob(T_ms, W_ms, lapse_max=cfg.lapse_max)
                deadline_s = W_ms / 1000.0

                if rng.random() < p_lapse:
                    rt_s = np.nan
                    response = 0
                    omission = 1
                    correct = 0
                else:
                    rt_s, response, omission, _ = simulate_ddm_with_deadline(
                        rng, v_final, a_final, z_final, cfg.t0, deadline_s
                    )
                    correct = 1 if response == 1 else 0

                rows.append(
                    {
                        "condition_id": row["condition_id"],
                        "P": P,
                        "T_ms": T_ms,
                        "W_ms": W_ms,
                        "label": label,
                        "rt_s": rt_s,
                        "responded": int(omission == 0),
                        "omission": omission,
                        "correct": correct,
                        "v_base": v_base,
                        "a_base": a_base,
                    }
                )
    return pd.DataFrame(rows)


def objective_pure_sigmoid(theta, cond_df, target_df):
    alpha1, alpha2, gamma, beta1, beta2, v_noise, a_noise, lapse_max = theta
    cfg = SimConfig(v_noise=v_noise, a_noise=a_noise, lapse_max=lapse_max)

    sim_trials = simulate_for_conditions(
        cond_df,
        cfg,
        mode="pure",
        pure_params=(alpha1, alpha2, gamma, beta1, beta2),
        n_rep=1,
        seed=1234,
    )
    sim_summary = summarize_behavior(sim_trials)
    sim_summary = sim_summary.rename(
        columns={
            "acc": "acc_sim",
            "omit_rate": "omit_rate_sim",
            "rt_median": "rt_median_sim",
        }
    )

    merged = target_df.merge(
        sim_summary,
        on=["condition_id", "P", "T_ms", "W_ms", "label"],
        how="left",
        suffixes=("_real", "_sim"),
    )

    rt_rmse = np.sqrt(np.nanmean((merged["rt_median_real"] - merged["rt_median_sim"]) ** 2))
    acc_rmse = np.sqrt(np.nanmean((merged["acc_real"] - merged["acc_sim"]) ** 2))
    omit_rmse = np.sqrt(np.nanmean((merged["omit_rate_real"] - merged["omit_rate_sim"]) ** 2))

    if np.isnan(rt_rmse):
        rt_rmse = 10.0
    if np.isnan(acc_rmse):
        acc_rmse = 1.0
    if np.isnan(omit_rmse):
        omit_rmse = 1.0

    # 增加趋势惩罚：避免只拟合均值而忽略条件走势
    trend_penalty = 0.0
    for metric in ["rt_median", "acc", "omit_rate"]:
        for label in sorted(merged["label"].dropna().unique()):
            sub = merged[merged["label"] == label]
            rho = spearman_safe(sub[f"{metric}_real"], sub[f"{metric}_sim"])
            if np.isnan(rho):
                trend_penalty += 0.8
            else:
                trend_penalty += (1.0 - rho)

    score = 1.2 * rt_rmse + 1.0 * acc_rmse + 0.8 * omit_rmse + 0.2 * trend_penalty
    return float(score)


def fit_pure_sigmoid(cond_df, target_df):
    bounds = [
        (0.2, 2.5),
        (-0.9, 0.2),
        (0.05, 0.8),
        (0.0, 0.6),
        (-0.2, 0.2),
        (0.05, 1.5),
        (0.05, 1.0),
        (0.01, 0.45),
    ]

    rng = np.random.default_rng(42)
    best_x = None
    best_score = np.inf
    n_search = N_RANDOM_SEARCH

    for _ in range(n_search):
        x = np.array([rng.uniform(low, high) for (low, high) in bounds], dtype=float)
        score = objective_pure_sigmoid(x, cond_df, target_df)
        if score < best_score:
            best_score = score
            best_x = x.copy()

    return SimpleNamespace(x=best_x, fun=best_score, success=True, message=f"random_search_{n_search}")


def fit_gp_residual(real_ez, pure_param_tuple):
    alpha1, alpha2, gamma, beta1, beta2 = pure_param_tuple
    fit_df = real_ez[real_ez["valid_ez"] == 1].copy()
    fit_df = fit_df.dropna(subset=["v_real", "a_real"])

    if fit_df.empty:
        raise RuntimeError("无法拟合 GP：真实 EZ 参数为空。")

    v_s2_list = []
    a_s2_list = []
    X_rows = []
    for _, row in fit_df.iterrows():
        P = float(row["P"])
        T_ms = float(row["T_ms"])
        W_ms = float(row["W_ms"])
        M_ms = T_ms + W_ms
        key = condition_label_to_key(row["label"])
        v_s2 = compute_v_s2(T_ms, P, key, alpha1=alpha1, alpha2=alpha2, gamma=gamma)
        a_s2 = compute_a_s2(M_ms, beta1=beta1, beta2=beta2)
        X_rows.append(build_gp_features(P, T_ms, W_ms, row["label"]))
        v_s2_list.append(v_s2)
        a_s2_list.append(a_s2)

    X = np.array(X_rows)
    y_v = fit_df["v_real"].values - np.array(v_s2_list)
    y_a = fit_df["a_real"].values - np.array(a_s2_list)

    n_dim = X.shape[1]
    kernel = 1.0 * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 5e1)) + WhiteKernel(
        noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1)
    )
    gp_v = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp_a = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp_v.fit(X, y_v)
    gp_a.fit(X, y_a)

    return {"gp_v": gp_v, "gp_a": gp_a, "n_fit": len(fit_df)}


def compute_metrics(real_summary, model_summary):
    model_summary = model_summary.rename(
        columns={
            "acc": "acc_pred",
            "omit_rate": "omit_rate_pred",
            "rt_median": "rt_median_pred",
        }
    )
    merged = real_summary.merge(
        model_summary,
        on=["condition_id", "P", "T_ms", "W_ms", "label"],
        how="left",
        suffixes=("_real", "_pred"),
    )

    metrics = {
        "rmse_rt_median": float(np.sqrt(np.nanmean((merged["rt_median_real"] - merged["rt_median_pred"]) ** 2))),
        "rmse_acc": float(np.sqrt(np.nanmean((merged["acc_real"] - merged["acc_pred"]) ** 2))),
        "rmse_omit_rate": float(np.sqrt(np.nanmean((merged["omit_rate_real"] - merged["omit_rate_pred"]) ** 2))),
    }
    return metrics, merged


def spearman_safe(a, b):
    x = pd.Series(a)
    y = pd.Series(b)
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    return x[valid].corr(y[valid], method="spearman")


def make_figures(compare_df, out_prefix):
    for metric in ["rt_median", "acc", "omit_rate"]:
        plt.figure(figsize=(10, 5))
        for label in sorted(compare_df["label"].dropna().unique()):
            sub = compare_df[compare_df["label"] == label].sort_values(["P", "T_ms", "W_ms"])
            x = np.arange(len(sub))
            plt.plot(x, sub[f"{metric}_real"], marker="o", linewidth=1.8, label=f"{label}-real")
            plt.plot(x, sub[f"{metric}_pred"], marker="s", linewidth=1.8, linestyle="--", label=f"{label}-pred")
        plt.title(f"{out_prefix} | {metric}")
        plt.grid(alpha=0.25)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{out_prefix}_{metric}.png", dpi=220)
        plt.close()


# =========================
# 主流程
# =========================
def main():
    print("[1/8] 读取并预处理真实 Matching 数据")
    real_match, cond = load_real_matching_data()

    print("[2/8] 读取生成数据并执行最近邻条件映射")
    gen_mapped = load_generated_with_condition_mapping(cond)

    print("[3/8] 计算真实数据行为统计与 EZ 参数")
    real_behavior = summarize_behavior(real_match)
    real_behavior = real_behavior.rename(
        columns={
            "acc": "acc_real",
            "omit_rate": "omit_rate_real",
            "rt_median": "rt_median_real",
        }
    )
    real_ez = estimate_condition_ez(real_match)

    print("[4/8] 估计生成数据的条件统计（用于校验映射）")
    gen_behavior = summarize_behavior(gen_mapped, p_col="P_target", t_col="T_ms_target", w_col="W_ms_target")
    gen_behavior.to_csv(OUT_DIR / "generated_behavior_summary_v2.4.5.csv", index=False, encoding="utf-8-sig")

    print("[5/8] 拟合纯 Sigmoid 模型")
    fit_cond = real_behavior[["condition_id", "P", "T_ms", "W_ms", "label", "n_trials"]].copy()
    fit_cond["n_trials"] = fit_cond["n_trials"].clip(upper=FIT_TRIAL_CAP)
    de_result = fit_pure_sigmoid(fit_cond, real_behavior)
    best = de_result.x

    pure_params = {
        "alpha1": float(best[0]),
        "alpha2": float(best[1]),
        "gamma": float(best[2]),
        "beta1": float(best[3]),
        "beta2": float(best[4]),
        "v_noise": float(best[5]),
        "a_noise": float(best[6]),
        "lapse_max": float(best[7]),
        "objective": float(de_result.fun),
        "success": bool(de_result.success),
        "message": str(de_result.message),
    }
    pd.DataFrame([pure_params]).to_csv(
        OUT_DIR / "pure_sigmoid_fit_summary_v2.4.5.csv", index=False, encoding="utf-8-sig"
    )

    cfg = SimConfig(v_noise=pure_params["v_noise"], a_noise=pure_params["a_noise"], lapse_max=pure_params["lapse_max"])
    pred_cond = fit_cond.copy()
    pred_cond["n_trials"] = pred_cond["n_trials"].clip(upper=PRED_TRIAL_CAP)

    pure_trials = simulate_for_conditions(
        pred_cond,
        cfg,
        mode="pure",
        pure_params=(pure_params["alpha1"], pure_params["alpha2"], pure_params["gamma"], pure_params["beta1"], pure_params["beta2"]),
        n_rep=N_REP_PRED,
        seed=2026,
    )
    pure_summary = summarize_behavior(pure_trials)

    print("[6/8] 训练 GP 残差模型（基于真实 EZ 参数）")
    gp_models = fit_gp_residual(
        real_ez,
        (
            pure_params["alpha1"],
            pure_params["alpha2"],
            pure_params["gamma"],
            pure_params["beta1"],
            pure_params["beta2"],
        ),
    )

    gp_trials = simulate_for_conditions(
        pred_cond,
        cfg,
        mode="gp",
        gp_models=gp_models,
        pure_params=(pure_params["alpha1"], pure_params["alpha2"], pure_params["gamma"], pure_params["beta1"], pure_params["beta2"]),
        n_rep=N_REP_PRED,
        seed=4096,
    )
    gp_summary = summarize_behavior(gp_trials)

    with open(MODEL_DIR / "gp_residual_models_v2.4.5.pkl", "wb") as f:
        pickle.dump(gp_models, f)

    gp_fit_info = {
        "n_fit": int(gp_models["n_fit"]),
        "kernel_v": str(gp_models["gp_v"].kernel_),
        "kernel_a": str(gp_models["gp_a"].kernel_),
    }
    pd.DataFrame([gp_fit_info]).to_csv(OUT_DIR / "gp_sigmoid_fit_summary_v2.4.5.csv", index=False, encoding="utf-8-sig")

    print("[7/8] 计算模型比较指标")
    pure_metrics, pure_compare = compute_metrics(real_behavior, pure_summary)
    gp_metrics, gp_compare = compute_metrics(real_behavior, gp_summary)

    # 参数复原相关（真实 EZ vs 模型条件基线）
    ez_valid = real_ez[real_ez["valid_ez"] == 1].copy()
    pure_v_pred = []
    pure_a_pred = []
    gp_v_pred = []
    gp_a_pred = []
    for _, row in ez_valid.iterrows():
        P = float(row["P"])
        T_ms = float(row["T_ms"])
        W_ms = float(row["W_ms"])
        key = condition_label_to_key(row["label"])
        M_ms = T_ms + W_ms

        v_s2 = compute_v_s2(T_ms, P, key, alpha1=pure_params["alpha1"], alpha2=pure_params["alpha2"], gamma=pure_params["gamma"])
        a_s2 = compute_a_s2(M_ms, beta1=pure_params["beta1"], beta2=pure_params["beta2"])
        pure_v_pred.append(v_s2)
        pure_a_pred.append(a_s2)

        X = np.array([build_gp_features(P, T_ms, W_ms, row["label"])])
        gp_v_pred.append(v_s2 + gp_models["gp_v"].predict(X)[0])
        gp_a_pred.append(a_s2 + gp_models["gp_a"].predict(X)[0])

    param_metrics = {
        "spearman_v_pure": spearman_safe(ez_valid["v_real"], pure_v_pred),
        "spearman_v_gp": spearman_safe(ez_valid["v_real"], gp_v_pred),
        "spearman_a_pure": spearman_safe(ez_valid["a_real"], pure_a_pred),
        "spearman_a_gp": spearman_safe(ez_valid["a_real"], gp_a_pred),
    }

    metrics_df = pd.DataFrame(
        [
            {"model": "pure_sigmoid", **pure_metrics},
            {"model": "gp_sigmoid", **gp_metrics},
        ]
    )
    for k, v in param_metrics.items():
        metrics_df[k] = v

    metrics_df.to_csv(OUT_DIR / "model_compare_metrics_v2.4.5.csv", index=False, encoding="utf-8-sig")

    print("[8/8] 导出对齐结果与图表")
    pure_compare.to_csv(OUT_DIR / "condition_level_predictions_pure_v2.4.5.csv", index=False, encoding="utf-8-sig")
    gp_compare.to_csv(OUT_DIR / "condition_level_predictions_gp_v2.4.5.csv", index=False, encoding="utf-8-sig")
    real_ez.to_csv(OUT_DIR / "real_matching_ez_params_v2.4.5.csv", index=False, encoding="utf-8-sig")

    make_figures(pure_compare, "pure_sigmoid")
    make_figures(gp_compare, "gp_sigmoid")

    print("完成：输出目录 ->", OUT_DIR)
    print("完成：图表目录 ->", FIG_DIR)


if __name__ == "__main__":
    main()
