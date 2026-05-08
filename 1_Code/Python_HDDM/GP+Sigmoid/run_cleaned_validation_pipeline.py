"""
GP+Sigmoid cleaned validation pipeline.

完成 6 个步骤：
1. HDDM 参数诊断
2. Group 7/8 元数据纠正、合并与敏感性分析
3. 重新生成 cleaned HDDM 参数表
4. 重新训练 GP+Sigmoid 并做 LOCV
5. 模拟数据 vs 真实数据的行为层验证
6. 基于 GP 不确定性挑选新实验点
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from gp_sigmoid_hybrid_model import GPSigmoidHybridModel


HDDM_PARAMS_PATH = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces" / "all_groups_ddm_params.csv"
REAL_DATA_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined.csv"
REAL_SUMMARY_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined_summary.csv"

OUT_DATA_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid_Cleaned"
OUT_FIG_DIR = BASE_DIR / "3_Figures" / "GP_Sigmoid_Cleaned"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PipelineConfig:
    seed: int = 42
    sigmoid_maxiter: int = 300
    n_sim_per_identity: int | None = None
    sim_dt: float = 0.002
    candidate_topn: int = 20


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.nanmean((x - y) ** 2)))


def v_P_Function(P, P1=4, k_min=0.1, k_max=0.05, gamma=0.2, P0=32):
    P = np.asarray(P, dtype=float)
    k = k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))
    return 1.0 / (1.0 + np.exp(-k * (P - P1)))


def compute_v_s2(T, P, condition_key, alaph1=1.5, alaph2=-0.4, gamma=0.2, base_scale=3.0):
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    v_T = 1.0 / (1.0 + np.exp(-0.01 * (T - 100.0)))
    v_P = v_P_Function(P=P, gamma=gamma)
    v_0 = v_T * v_P * base_scale
    condition_key = np.asarray(condition_key)
    return np.where(condition_key == 1, v_0 * (1 + alaph1), v_0 * (1 + alaph2))


def compute_a_s2(M, beta1=0.2, beta2=0.0, base_scale=3.0):
    M = np.asarray(M, dtype=float)
    a_0 = 1.0 / (1.0 + np.exp(-0.01 * (M - 600.0))) * base_scale
    return np.where(M > 600, a_0 * (1 + beta1), a_0 * (1 + beta2))


def sigmoid_objective(params, df_data):
    alaph1, alaph2, beta1, beta2, gamma, base_scale_v, base_scale_a = params
    M = df_data["T_ms"].values + df_data["W_ms"].values
    v_self_sig = compute_v_s2(
        df_data["T_ms"].values, df_data["P"].values, np.ones(len(df_data)),
        alaph1=alaph1, alaph2=alaph2, gamma=gamma, base_scale=base_scale_v
    )
    v_stranger_sig = compute_v_s2(
        df_data["T_ms"].values, df_data["P"].values, np.zeros(len(df_data)),
        alaph1=alaph1, alaph2=alaph2, gamma=gamma, base_scale=base_scale_v
    )
    a_sig = compute_a_s2(M, beta1=beta1, beta2=beta2, base_scale=base_scale_a)
    err = np.r_[
        v_self_sig - df_data["v_self_mean"].values,
        v_stranger_sig - df_data["v_stranger_mean"].values,
        a_sig - df_data["a_mean"].values,
    ]
    return float(np.sqrt(np.mean(err ** 2)))


def calibrate_sigmoid(df_data, seed=42, maxiter=300):
    bounds = [
        (0.1, 3.0),    # alaph1
        (-2.0, 1.0),   # alaph2
        (-1.0, 2.0),   # beta1
        (-1.0, 1.0),   # beta2
        (0.01, 1.0),   # gamma
        (1.0, 10.0),   # base_scale_v
        (1.0, 10.0),   # base_scale_a
    ]
    result = differential_evolution(
        sigmoid_objective,
        bounds,
        args=(df_data,),
        maxiter=maxiter,
        seed=seed,
        popsize=12,
        tol=1e-7,
        polish=True,
    )
    names = ["alaph1", "alaph2", "beta1", "beta2", "gamma", "base_scale_v", "base_scale_a"]
    params = dict(zip(names, result.x))
    params["final_rmse"] = float(result.fun)
    params["converged"] = bool(result.success)
    params["t_baseline"] = float(df_data["t_mean"].mean())
    params["z_baseline"] = float(df_data["z_mean"].mean())
    return params


def prepare_real_matching_data():
    real = pd.read_csv(REAL_DATA_PATH)
    real = real[real["stage"] == "formal"].copy()
    if "Matching" not in real.columns:
        real["Matching"] = np.where(
            ((real["Shape"] == "circle") & (real["Label"] == "self"))
            | ((real["Shape"] == "square") & (real["Label"] == "stranger")),
            "Matching",
            "NonMatching",
        )
    real = real[real["Matching"].str.lower() == "matching"].copy()
    real["group_id"] = real["groupID"].astype(int)
    real["identity"] = real["Label"].str.lower()
    real["RT_sec"] = pd.to_numeric(real.get("RT_sec", real["RT"]), errors="coerce")
    real["T_ms"] = pd.to_numeric(real.get("T_ms", real["T"] * 1000), errors="coerce")
    real["W_ms"] = pd.to_numeric(real.get("W_ms", real["W"] * 1000), errors="coerce")
    real["responded"] = real["RT_sec"].notna() & (real["RT_sec"] > 0)
    real["omission"] = (~real["responded"]).astype(int)
    real["ACC"] = real.get("ACC", real["Correct"]).astype(int)
    return real


def summarize_real_matching(real):
    rows = []
    for (gid, identity), g in real.groupby(["group_id", "identity"]):
        correct_rt = g.loc[(g["ACC"] == 1) & g["responded"], "RT_sec"]
        rows.append({
            "group_id": gid,
            "identity": identity,
            "P": float(g["P"].iloc[0]),
            "T_ms": float(g["T_ms"].iloc[0]),
            "W_ms": float(g["W_ms"].iloc[0]),
            "n_subjects": int(g["SubjectUID"].nunique() if "SubjectUID" in g.columns else g["subjectID"].nunique()),
            "n_trials": int(len(g)),
            "acc": float(g["ACC"].mean()),
            "omission_rate": float(g["omission"].mean()),
            "correct_rt_mean_ms": float(correct_rt.mean() * 1000) if len(correct_rt) else np.nan,
            "correct_rt_sd_ms": float(correct_rt.std() * 1000) if len(correct_rt) > 1 else np.nan,
            "correct_rt_q10_ms": float(correct_rt.quantile(0.10) * 1000) if len(correct_rt) else np.nan,
            "correct_rt_q50_ms": float(correct_rt.quantile(0.50) * 1000) if len(correct_rt) else np.nan,
            "correct_rt_q90_ms": float(correct_rt.quantile(0.90) * 1000) if len(correct_rt) else np.nan,
        })
    by_identity = pd.DataFrame(rows)
    spe_rows = []
    for gid, g in by_identity.groupby("group_id"):
        piv = g.pivot(index="group_id", columns="identity", values="correct_rt_mean_ms")
        spe_rt = np.nan
        if {"self", "stranger"}.issubset(set(piv.columns)):
            spe_rt = float(piv.loc[gid, "self"] - piv.loc[gid, "stranger"])
        first = g.iloc[0]
        spe_rows.append({
            "group_id": gid,
            "P": first["P"],
            "T_ms": first["T_ms"],
            "W_ms": first["W_ms"],
            "n_subjects": int(g["n_subjects"].max()),
            "n_trials": int(g["n_trials"].sum()),
            "acc": float(np.average(g["acc"], weights=g["n_trials"])),
            "omission_rate": float(np.average(g["omission_rate"], weights=g["n_trials"])),
            "SPE_RT_ms": spe_rt,
        })
    by_group = pd.DataFrame(spe_rows)
    return by_identity, by_group


def step1_hddm_diagnostics(hddm, real_group_summary, behavior_group_summary):
    h = hddm.copy()
    h["group_id"] = h["group_id"].astype(int)
    expected = behavior_group_summary[["group_id", "P", "T_ms", "W_ms", "n_subjects", "n_trials", "acc", "omission_rate", "SPE_RT_ms"]].copy()
    diag = h.merge(expected, on="group_id", how="left", suffixes=("_hddm", "_behavior"))
    for col in ["P", "T_ms", "W_ms"]:
        diag[f"{col}_matches_behavior"] = np.isclose(diag[f"{col}_hddm"], diag[f"{col}_behavior"], equal_nan=False)
    diag["design_matches_behavior"] = diag[["P_matches_behavior", "T_ms_matches_behavior", "W_ms_matches_behavior"]].all(axis=1)

    diag["v_self_ci_width"] = diag["v_self_q975"] - diag["v_self_q025"]
    diag["v_stranger_ci_width"] = diag["v_stranger_q975"] - diag["v_stranger_q025"]
    diag["a_ci_width"] = diag["a_q975"] - diag["a_q025"]
    diag["t_ci_width"] = diag["t_q975"] - diag["t_q025"]
    diag["z_ci_width"] = diag["z_q975"] - diag["z_q025"]
    diag["SPE_v_abs"] = diag["SPE_v"].abs()
    diag["flag_high_omission"] = diag["omission_rate"] > 0.50
    diag["flag_wide_v_ci"] = (diag["v_self_ci_width"] > 6.0) | (diag["v_stranger_ci_width"] > 6.0)
    diag["flag_a_out_of_range"] = ~diag["a_mean"].between(0.3, 3.5)
    diag["flag_t_out_of_range"] = ~diag["t_mean"].between(0.05, 1.2)
    diag["flag_z_out_of_range"] = ~diag["z_mean"].between(0.05, 0.95)
    diag["flag_metadata_mismatch"] = ~diag["design_matches_behavior"]
    flag_cols = [c for c in diag.columns if c.startswith("flag_")]
    diag["n_flags"] = diag[flag_cols].sum(axis=1)

    design_counts_original = h.groupby(["P", "T_ms", "W_ms"])["group_id"].apply(lambda x: ",".join(map(str, x))).reset_index()
    design_counts_original["n_groups"] = design_counts_original["group_id"].str.split(",").str.len()
    design_counts_behavior = expected.groupby(["P", "T_ms", "W_ms"])["group_id"].apply(lambda x: ",".join(map(str, x))).reset_index()
    design_counts_behavior["n_groups"] = design_counts_behavior["group_id"].str.split(",").str.len()

    diag.to_csv(OUT_DATA_DIR / "step1_hddm_parameter_diagnostics.csv", index=False)
    design_counts_original.to_csv(OUT_DATA_DIR / "step1_design_duplicates_original_hddm.csv", index=False)
    design_counts_behavior.to_csv(OUT_DATA_DIR / "step1_design_duplicates_behavior_metadata.csv", index=False)
    return diag


def aggregate_params_by_design(df, source_name):
    weight_col = "n_subjects"
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    rows = []
    mean_cols = [c for c in df.columns if c.endswith("_mean")] + ["SPE_v"]
    std_cols = [c for c in df.columns if c.endswith("_std")]
    q025_cols = [c for c in df.columns if c.endswith("_q025")]
    q975_cols = [c for c in df.columns if c.endswith("_q975")]
    extra_cols = ["omission_rate", "acc", "SPE_RT_ms"]
    for (P, T, W), g in df.groupby(["P", "T_ms", "W_ms"], sort=True):
        weights = g[weight_col].fillna(1.0).values.astype(float)
        row = {
            "condition_id": len(rows) + 1,
            "source_group_ids": ",".join(map(str, g["group_id"].astype(int).tolist())),
            "P": float(P),
            "T_ms": float(T),
            "W_ms": float(W),
            "M_ms": float(T + W),
            "n_source_groups": int(len(g)),
            "n_subjects": int(g.get("n_subjects", pd.Series([len(g)])).sum()),
            "source_table": source_name,
        }
        for col in mean_cols:
            if col in g:
                row[col] = float(np.average(g[col], weights=weights))
        for col in std_cols:
            if col in g:
                row[col] = float(np.average(g[col], weights=weights))
        for col in q025_cols:
            if col in g:
                row[col] = float(g[col].min())
        for col in q975_cols:
            if col in g:
                row[col] = float(g[col].max())
        for col in extra_cols:
            if col in g:
                row[col] = float(np.average(g[col], weights=weights))
        if "v_self_mean" in row and "v_stranger_mean" in row:
            row["SPE_v"] = row["v_self_mean"] - row["v_stranger_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def step2_step3_clean_hddm_tables(hddm, diagnostics, behavior_group_summary):
    h = hddm.copy()
    h["group_id"] = h["group_id"].astype(int)
    expected = behavior_group_summary[["group_id", "P", "T_ms", "W_ms", "n_subjects", "n_trials", "acc", "omission_rate", "SPE_RT_ms"]].copy()
    corrected = h.drop(columns=["P", "T_ms", "W_ms", "M_ms"], errors="ignore").merge(expected, on="group_id", how="left")
    corrected["M_ms"] = corrected["T_ms"] + corrected["W_ms"]
    corrected["metadata_corrected"] = True

    main = aggregate_params_by_design(corrected, "metadata_corrected_by_behavior")
    main.to_csv(OUT_DATA_DIR / "step3_cleaned_hddm_params_main.csv", index=False)
    corrected.to_csv(OUT_DATA_DIR / "step3_hddm_params_metadata_corrected_by_group.csv", index=False)

    original_keep = h.merge(expected[["group_id", "n_subjects", "n_trials", "acc", "omission_rate", "SPE_RT_ms"]], on="group_id", how="left")
    sensitivity = {
        "original_keep_all": original_keep,
        "original_merge_duplicate_designs": aggregate_params_by_design(original_keep, "original_merge_duplicate_designs"),
        "metadata_corrected_keep_all": corrected,
        "metadata_corrected_aggregated": main,
        "metadata_corrected_drop_high_omission": aggregate_params_by_design(
            corrected[corrected["omission_rate"] <= 0.50].copy(), "metadata_corrected_drop_high_omission"
        ),
        "metadata_corrected_drop_any_flagged": aggregate_params_by_design(
            corrected.merge(diagnostics[["group_id", "n_flags"]], on="group_id", how="left").query("n_flags == 0").copy(),
            "metadata_corrected_drop_any_flagged",
        ),
    }
    for name, table in sensitivity.items():
        table.to_csv(OUT_DATA_DIR / f"step2_sensitivity_{name}.csv", index=False)
    return main, corrected, sensitivity


def train_gp_sigmoid(df_clean, cfg: PipelineConfig, label="main"):
    params = calibrate_sigmoid(df_clean, seed=cfg.seed, maxiter=cfg.sigmoid_maxiter)
    pd.DataFrame([params]).to_csv(OUT_DATA_DIR / f"step4_sigmoid_calibrated_params_{label}.csv", index=False)

    model = GPSigmoidHybridModel(seed=cfg.seed)
    model.fit(df_clean, calib_params=params)
    model_path = OUT_DATA_DIR / f"step4_gp_sigmoid_model_{label}.pkl"
    model.save(model_path)

    preds = []
    for _, row in df_clean.iterrows():
        vs, vo, a, t, z = model.predict_params_full(row["P"], row["T_ms"], row["W_ms"])
        vs = float(np.asarray(vs).item())
        vo = float(np.asarray(vo).item())
        a = float(np.asarray(a).item())
        t = float(np.asarray(t).item())
        z = float(np.asarray(z).item())
        preds.append({
            "condition_id": row.get("condition_id", row.name + 1),
            "source_group_ids": row.get("source_group_ids", row.get("group_id", "")),
            "P": row["P"],
            "T_ms": row["T_ms"],
            "W_ms": row["W_ms"],
            "v_self_real": row["v_self_mean"],
            "v_self_pred": vs,
            "v_stranger_real": row["v_stranger_mean"],
            "v_stranger_pred": vo,
            "a_real": row["a_mean"],
            "a_pred": a,
            "t_real": row["t_mean"],
            "t_pred": t,
            "z_real": row["z_mean"],
            "z_pred": z,
            "SPE_v_real": row["v_self_mean"] - row["v_stranger_mean"],
            "SPE_v_pred": vs - vo,
        })
    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(OUT_DATA_DIR / f"step4_gp_sigmoid_predictions_{label}.csv", index=False)

    metrics = []
    for param in ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]:
        metrics.append({
            "target": param,
            "rmse": rmse(pred_df[f"{param}_real"], pred_df[f"{param}_pred"]),
            "r": safe_corr(pred_df[f"{param}_real"], pred_df[f"{param}_pred"]),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DATA_DIR / f"step4_training_metrics_{label}.csv", index=False)
    return model, params, pred_df, metrics_df


def run_locv(df_clean, cfg: PipelineConfig):
    rows = []
    for held_idx, held in df_clean.iterrows():
        train = df_clean.drop(index=held_idx).copy()
        if len(train) < 3:
            continue
        params = calibrate_sigmoid(train, seed=cfg.seed, maxiter=max(120, cfg.sigmoid_maxiter // 2))
        model = GPSigmoidHybridModel(seed=cfg.seed)
        model.fit(train, calib_params=params)
        vs, vo, a, t, z = model.predict_params_full(held["P"], held["T_ms"], held["W_ms"])
        vs = float(np.asarray(vs).item())
        vo = float(np.asarray(vo).item())
        a = float(np.asarray(a).item())
        t = float(np.asarray(t).item())
        z = float(np.asarray(z).item())
        row = {
            "held_condition_id": held.get("condition_id", held_idx + 1),
            "held_source_group_ids": held.get("source_group_ids", ""),
            "P": held["P"],
            "T_ms": held["T_ms"],
            "W_ms": held["W_ms"],
            "v_self_real": held["v_self_mean"],
            "v_self_pred": vs,
            "v_stranger_real": held["v_stranger_mean"],
            "v_stranger_pred": vo,
            "a_real": held["a_mean"],
            "a_pred": a,
            "t_real": held["t_mean"],
            "t_pred": t,
            "z_real": held["z_mean"],
            "z_pred": z,
            "SPE_v_real": held["v_self_mean"] - held["v_stranger_mean"],
            "SPE_v_pred": vs - vo,
        }
        for param in ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]:
            row[f"{param}_error"] = row[f"{param}_pred"] - row[f"{param}_real"]
        rows.append(row)
    locv = pd.DataFrame(rows)
    locv.to_csv(OUT_DATA_DIR / "step4_locv_results_cleaned.csv", index=False)
    metrics = []
    for param in ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]:
        metrics.append({
            "target": param,
            "rmse": rmse(locv[f"{param}_real"], locv[f"{param}_pred"]),
            "mae": float(np.nanmean(np.abs(locv[f"{param}_real"] - locv[f"{param}_pred"]))),
            "r": safe_corr(locv[f"{param}_real"], locv[f"{param}_pred"]),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DATA_DIR / "step4_locv_metrics_cleaned.csv", index=False)
    return locv, metrics_df


def simulate_ddm_trials(n, v, a, t0, z, deadline, rng, dt=0.002):
    # HDDM 的 z 通常是边界比例；若 z>1，则按绝对起点处理。
    start = z * a if 0 < z <= 1 else z
    start = float(np.clip(start, 0.01, a - 0.01))
    max_steps = max(1, int(np.ceil(max(deadline - t0, dt) / dt)))
    rows = []
    noise_scale = np.sqrt(dt)
    for _ in range(n):
        x = start
        response = 0
        rt = np.nan
        for step in range(1, max_steps + 1):
            x += v * dt + rng.normal(0, noise_scale)
            if x >= a:
                response = 1
                rt_candidate = t0 + step * dt
                if rt_candidate <= deadline:
                    rt = rt_candidate
                break
            if x <= 0:
                response = 0
                rt_candidate = t0 + step * dt
                if rt_candidate <= deadline:
                    rt = rt_candidate
                break
        omission = int(np.isnan(rt))
        rows.append({
            "RT_sec": rt,
            "responded": int(not omission),
            "omission": omission,
            "ACC": int((not omission) and response == 1),
            "response_boundary": response if not omission else np.nan,
        })
    return rows


def behavior_validation(model, real, df_clean, cfg: PipelineConfig):
    rng = np.random.default_rng(cfg.seed)
    sim_rows = []
    for _, cond in df_clean.iterrows():
        source_groups = [int(x) for x in str(cond.get("source_group_ids", cond.get("group_id", ""))).split(",") if x != ""]
        if not source_groups:
            continue
        real_cond = real[real["group_id"].isin(source_groups)]
        for identity, condition_key in [("self", 1), ("stranger", 0)]:
            n_real = int((real_cond["identity"] == identity).sum())
            if cfg.n_sim_per_identity is not None:
                n_real = min(n_real, cfg.n_sim_per_identity)
            if n_real <= 0:
                continue
            v, a, t0, z = model.predict(cond["P"], cond["T_ms"], cond["W_ms"], condition_key)
            v = float(np.asarray(v).item())
            a = float(np.asarray(a).item())
            t0 = float(np.asarray(t0).item())
            z = float(np.asarray(z).item())
            deadline = float((cond["T_ms"] + cond["W_ms"]) / 1000.0)
            trials = simulate_ddm_trials(n_real, v, a, t0, z, deadline, rng, dt=cfg.sim_dt)
            for trial in trials:
                trial.update({
                    "condition_id": cond.get("condition_id", np.nan),
                    "source_group_ids": cond.get("source_group_ids", ""),
                    "P": cond["P"],
                    "T_ms": cond["T_ms"],
                    "W_ms": cond["W_ms"],
                    "identity": identity,
                    "v": v,
                    "a": a,
                    "t": t0,
                    "z": z,
                })
                sim_rows.append(trial)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT_DATA_DIR / "step5_simulated_matching_trials.csv", index=False)

    real_by_id, real_by_group = summarize_real_matching(real)
    sim_by_id, sim_by_group = summarize_simulated_behavior(sim)
    comp_id = real_by_id.merge(
        sim_by_id,
        on=["identity", "P", "T_ms", "W_ms"],
        how="outer",
        suffixes=("_real", "_sim"),
    )
    for col in ["acc", "omission_rate", "correct_rt_mean_ms", "correct_rt_q10_ms", "correct_rt_q50_ms", "correct_rt_q90_ms"]:
        comp_id[f"{col}_diff_sim_minus_real"] = comp_id[f"{col}_sim"] - comp_id[f"{col}_real"]
    comp_id.to_csv(OUT_DATA_DIR / "step5_behavior_validation_by_identity.csv", index=False)

    comp_group = real_by_group.merge(
        sim_by_group,
        on=["P", "T_ms", "W_ms"],
        how="outer",
        suffixes=("_real", "_sim"),
    )
    for col in ["acc", "omission_rate", "SPE_RT_ms"]:
        comp_group[f"{col}_diff_sim_minus_real"] = comp_group[f"{col}_sim"] - comp_group[f"{col}_real"]
    comp_group.to_csv(OUT_DATA_DIR / "step5_behavior_validation_by_condition.csv", index=False)

    metrics = []
    for col in ["acc", "omission_rate", "correct_rt_mean_ms"]:
        metrics.append({
            "level": "identity",
            "target": col,
            "rmse": rmse(comp_id[f"{col}_real"], comp_id[f"{col}_sim"]),
            "mae": float(np.nanmean(np.abs(comp_id[f"{col}_real"] - comp_id[f"{col}_sim"]))),
            "r": safe_corr(comp_id[f"{col}_real"], comp_id[f"{col}_sim"]),
        })
    for col in ["acc", "omission_rate", "SPE_RT_ms"]:
        metrics.append({
            "level": "condition",
            "target": col,
            "rmse": rmse(comp_group[f"{col}_real"], comp_group[f"{col}_sim"]),
            "mae": float(np.nanmean(np.abs(comp_group[f"{col}_real"] - comp_group[f"{col}_sim"]))),
            "r": safe_corr(comp_group[f"{col}_real"], comp_group[f"{col}_sim"]),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DATA_DIR / "step5_behavior_validation_metrics.csv", index=False)
    plot_behavior_validation(comp_group, comp_id)
    return sim, comp_group, metrics_df


def summarize_simulated_behavior(sim):
    rows = []
    for (cond_id, identity), g in sim.groupby(["condition_id", "identity"]):
        correct_rt = g.loc[(g["ACC"] == 1) & (g["omission"] == 0), "RT_sec"]
        rows.append({
            "condition_id": cond_id,
            "group_id": int(cond_id),  # 仅用于和主表对齐；后续会按设计变量合并
            "identity": identity,
            "P": float(g["P"].iloc[0]),
            "T_ms": float(g["T_ms"].iloc[0]),
            "W_ms": float(g["W_ms"].iloc[0]),
            "n_trials": int(len(g)),
            "acc": float(g["ACC"].mean()),
            "omission_rate": float(g["omission"].mean()),
            "correct_rt_mean_ms": float(correct_rt.mean() * 1000) if len(correct_rt) else np.nan,
            "correct_rt_sd_ms": float(correct_rt.std() * 1000) if len(correct_rt) > 1 else np.nan,
            "correct_rt_q10_ms": float(correct_rt.quantile(0.10) * 1000) if len(correct_rt) else np.nan,
            "correct_rt_q50_ms": float(correct_rt.quantile(0.50) * 1000) if len(correct_rt) else np.nan,
            "correct_rt_q90_ms": float(correct_rt.quantile(0.90) * 1000) if len(correct_rt) else np.nan,
        })
    by_identity = pd.DataFrame(rows)
    spe_rows = []
    for _, g in by_identity.groupby(["P", "T_ms", "W_ms"]):
        means = g.set_index("identity")["correct_rt_mean_ms"].to_dict()
        spe_rt = np.nan
        if "self" in means and "stranger" in means:
            spe_rt = float(means["self"] - means["stranger"])
        first = g.iloc[0]
        spe_rows.append({
            "group_id": int(first["condition_id"]),
            "P": first["P"],
            "T_ms": first["T_ms"],
            "W_ms": first["W_ms"],
            "n_trials": int(g["n_trials"].sum()),
            "acc": float(np.average(g["acc"], weights=g["n_trials"])),
            "omission_rate": float(np.average(g["omission_rate"], weights=g["n_trials"])),
            "SPE_RT_ms": spe_rt,
        })
    by_group = pd.DataFrame(spe_rows)
    return by_identity, by_group


def plot_behavior_validation(comp_group, comp_id):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
        axes,
        ["acc", "omission_rate", "SPE_RT_ms"],
        ["Accuracy", "Omission Rate", "SPE RT (ms)"],
    ):
        ax.scatter(comp_group[f"{col}_real"], comp_group[f"{col}_sim"], s=70, color="steelblue", edgecolors="black")
        vals = pd.concat([comp_group[f"{col}_real"], comp_group[f"{col}_sim"]]).dropna()
        if len(vals):
            lo, hi = vals.min(), vals.max()
            pad = (hi - lo) * 0.10 if hi > lo else 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", alpha=0.6)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Real")
        ax.set_ylabel("Simulated")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "step5_behavior_validation_condition_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = comp_id.sort_values(["P", "T_ms", "W_ms", "identity"])
    labels = [f"{int(r.P)}-{int(r.T_ms)}-{int(r.W_ms)}-{r.identity[0]}" for _, r in plot_df.iterrows()]
    x = np.arange(len(plot_df))
    ax.bar(x - 0.2, plot_df["correct_rt_mean_ms_real"], width=0.4, label="Real", color="steelblue")
    ax.bar(x + 0.2, plot_df["correct_rt_mean_ms_sim"], width=0.4, label="Sim", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Correct RT mean (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "step5_behavior_validation_rt_bars.png", dpi=200, bbox_inches="tight")
    plt.close()


def find_candidate_design_points(model, df_clean, cfg: PipelineConfig):
    P_grid = np.linspace(0, 120, 25)
    T_grid = np.linspace(30, 500, 25)
    W_grid = np.linspace(300, 1500, 25)
    Pg, Tg, Wg = np.meshgrid(P_grid, T_grid, W_grid, indexing="ij")
    Pf, Tf, Wf = Pg.ravel(), Tg.ravel(), Wg.ravel()
    X = model.normalize_PTW(Pf, Tf, Wf)
    _, vs_std = model.gp_v_self.predict(X, return_std=True)
    _, vo_std = model.gp_v_stranger.predict(X, return_std=True)
    _, a_std = model.gp_a.predict(X, return_std=True)
    _, t_std = model.gp_t.predict(X, return_std=True)
    _, z_std = model.gp_z.predict(X, return_std=True)
    total_unc = vs_std + vo_std + a_std + t_std + z_std

    existing = model.normalize_PTW(df_clean["P"].values, df_clean["T_ms"].values, df_clean["W_ms"].values)
    candidate_norm = model.normalize_PTW(Pf, Tf, Wf)
    too_close = np.zeros(len(Pf), dtype=bool)
    for point in existing:
        dist = np.sqrt(((candidate_norm - point) ** 2).sum(axis=1))
        too_close |= dist < 0.18
    score = total_unc.copy()
    score[too_close] = -np.inf
    top_idx = np.argsort(score)[::-1][:cfg.candidate_topn]
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        rows.append({
            "rank": rank,
            "P": int(round(Pf[idx])),
            "T_ms": int(round(Tf[idx])),
            "W_ms": int(round(Wf[idx])),
            "M_ms": int(round(Tf[idx] + Wf[idx])),
            "total_uncertainty": float(total_unc[idx]),
            "v_self_std": float(vs_std[idx]),
            "v_stranger_std": float(vo_std[idx]),
            "a_std": float(a_std[idx]),
            "t_std": float(t_std[idx]),
            "z_std": float(z_std[idx]),
        })
    candidates = pd.DataFrame(rows)
    candidates.to_csv(OUT_DATA_DIR / "step6_candidate_design_points.csv", index=False)
    plot_candidate_points(candidates, df_clean)
    return candidates


def plot_candidate_points(candidates, df_clean):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(df_clean["P"], df_clean["T_ms"], c="red", marker="x", s=90, label="Existing")
    ax.scatter(candidates["P"], candidates["T_ms"], c=candidates["total_uncertainty"], cmap="viridis", s=70, label="Candidates")
    for _, r in candidates.head(8).iterrows():
        ax.annotate(f"#{int(r['rank'])}", (r["P"], r["T_ms"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("P")
    ax.set_ylabel("T (ms)")
    ax.set_title("Candidate points in P-T space")
    ax.legend(fontsize=8)

    ax = axes[1]
    top = candidates.sort_values("total_uncertainty", ascending=True)
    labels = [f"#{int(r['rank'])}: P{int(r['P'])},T{int(r['T_ms'])},W{int(r['W_ms'])}" for _, r in top.iterrows()]
    ax.barh(labels, top["total_uncertainty"], color="steelblue")
    ax.set_xlabel("Total GP uncertainty")
    ax.set_title("Top candidate ranking")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "step6_candidate_design_points.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_step4_results(pred_df, locv_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, param in zip(axes.ravel(), ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]):
        ax.scatter(pred_df[f"{param}_real"], pred_df[f"{param}_pred"], color="steelblue", edgecolors="black", s=70)
        vals = pd.concat([pred_df[f"{param}_real"], pred_df[f"{param}_pred"]]).dropna()
        if len(vals):
            lo, hi = vals.min(), vals.max()
            pad = (hi - lo) * 0.10 if hi > lo else 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", alpha=0.6)
        ax.set_title(f"{param}: RMSE={rmse(pred_df[f'{param}_real'], pred_df[f'{param}_pred']):.3f}")
        ax.set_xlabel("HDDM")
        ax.set_ylabel("GP+Sigmoid")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "step4_training_fit_cleaned.png", dpi=200, bbox_inches="tight")
    plt.close()

    if locv_df is not None and len(locv_df):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, param in zip(axes.ravel(), ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]):
            ax.scatter(locv_df[f"{param}_real"], locv_df[f"{param}_pred"], color="coral", edgecolors="black", s=70)
            vals = pd.concat([locv_df[f"{param}_real"], locv_df[f"{param}_pred"]]).dropna()
            if len(vals):
                lo, hi = vals.min(), vals.max()
                pad = (hi - lo) * 0.10 if hi > lo else 0.1
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", alpha=0.6)
            ax.set_title(f"LOCV {param}: RMSE={rmse(locv_df[f'{param}_real'], locv_df[f'{param}_pred']):.3f}")
            ax.set_xlabel("HDDM")
            ax.set_ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(OUT_FIG_DIR / "step4_locv_fit_cleaned.png", dpi=200, bbox_inches="tight")
        plt.close()


def write_summary_report(diag, clean, train_metrics, locv_metrics, behavior_metrics, candidates):
    report = {
        "outputs_dir": str(OUT_DATA_DIR),
        "figures_dir": str(OUT_FIG_DIR),
        "n_hddm_groups": int(len(diag)),
        "n_clean_conditions": int(len(clean)),
        "metadata_mismatch_groups": diag.loc[diag["flag_metadata_mismatch"], "group_id"].astype(int).tolist(),
        "high_omission_groups": diag.loc[diag["flag_high_omission"], "group_id"].astype(int).tolist(),
        "training_metrics": train_metrics.to_dict(orient="records"),
        "locv_metrics": locv_metrics.to_dict(orient="records"),
        "behavior_metrics": behavior_metrics.to_dict(orient="records"),
        "top_candidate_points": candidates.head(8).to_dict(orient="records"),
    }
    with open(OUT_DATA_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# GP+Sigmoid Cleaned Pipeline Summary",
        "",
        f"- HDDM groups diagnosed: {len(diag)}",
        f"- Cleaned design conditions: {len(clean)}",
        f"- Metadata mismatch groups: {report['metadata_mismatch_groups']}",
        f"- High omission groups (>50%): {report['high_omission_groups']}",
        "",
        "## Training Metrics",
        train_metrics.round(4).to_markdown(index=False),
        "",
        "## LOCV Metrics",
        locv_metrics.round(4).to_markdown(index=False),
        "",
        "## Behavior Validation Metrics",
        behavior_metrics.round(4).to_markdown(index=False),
        "",
        "## Top Candidate Points",
        candidates.head(10).round(4).to_markdown(index=False),
        "",
    ]
    (OUT_DATA_DIR / "pipeline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(cfg: PipelineConfig):
    print("=" * 72)
    print("GP+Sigmoid cleaned validation pipeline")
    print("=" * 72)

    print("[Step 0] Loading data...")
    hddm = pd.read_csv(HDDM_PARAMS_PATH)
    real = prepare_real_matching_data()
    real_by_identity, behavior_group_summary = summarize_real_matching(real)
    real_by_identity.to_csv(OUT_DATA_DIR / "step0_real_matching_summary_by_identity.csv", index=False)
    behavior_group_summary.to_csv(OUT_DATA_DIR / "step0_real_matching_summary_by_condition.csv", index=False)

    print("[Step 1] HDDM diagnostics...")
    diag = step1_hddm_diagnostics(hddm, None, behavior_group_summary)
    print(f"  Metadata mismatch groups: {diag.loc[diag['flag_metadata_mismatch'], 'group_id'].astype(int).tolist()}")
    print(f"  High omission groups: {diag.loc[diag['flag_high_omission'], 'group_id'].astype(int).tolist()}")

    print("[Step 2/3] Cleaning HDDM parameter table and writing sensitivity tables...")
    clean, corrected, sensitivity = step2_step3_clean_hddm_tables(hddm, diag, behavior_group_summary)
    print(f"  Cleaned conditions: {len(clean)}")

    print("[Step 4] Re-running GP+Sigmoid on cleaned HDDM table...")
    model, params, pred_df, train_metrics = train_gp_sigmoid(clean, cfg, label="cleaned_main")
    locv_df, locv_metrics = run_locv(clean, cfg)
    plot_step4_results(pred_df, locv_df)
    print(train_metrics.round(4).to_string(index=False))
    print(locv_metrics.round(4).to_string(index=False))

    print("[Step 5] Behavior-level validation...")
    sim, comp_group, behavior_metrics = behavior_validation(model, real, clean, cfg)
    print(behavior_metrics.round(4).to_string(index=False))

    print("[Step 6] Candidate design points from uncertainty...")
    candidates = find_candidate_design_points(model, clean, cfg)
    print(candidates.head(8).round(4).to_string(index=False))

    write_summary_report(diag, clean, train_metrics, locv_metrics, behavior_metrics, candidates)
    print("=" * 72)
    print("Pipeline complete")
    print(f"Data outputs: {OUT_DATA_DIR}")
    print(f"Figures:      {OUT_FIG_DIR}")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmoid-maxiter", type=int, default=300)
    parser.add_argument("--n-sim-per-identity", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--candidate-topn", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PipelineConfig(
        seed=args.seed,
        sigmoid_maxiter=args.sigmoid_maxiter,
        n_sim_per_identity=args.n_sim_per_identity,
        sim_dt=args.sim_dt,
        candidate_topn=args.candidate_topn,
    )
    run_pipeline(config)
