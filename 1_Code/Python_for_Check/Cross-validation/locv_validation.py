"""
阶段 B: 留一条件交叉验证 (Leave-One-Condition-Out Cross-Validation)

每次留一个实验条件 (group) 作为验证集，用其余 7 个条件训练 GP+Sigmoid 混合模型，
检验模型对未参与训练条件的预测精度。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "1_Code" / "Python_HDDM" / "GP+Sigmoid"))
from gp_sigmoid_hybrid_model import GPSigmoidHybridModel, train_model_from_data

BASE_DIR = Path(__file__).resolve().parents[3]
HDDM_PARAMS_PATH = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces" / "all_groups_ddm_params.csv"
CALIB_PARAMS_PATH = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid" / "sigmoid_calibrated_params.csv"
OUT_DATA_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Cross_Validation"
OUT_FIG_DIR = BASE_DIR / "3_Figures" / "Cross_Validation"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("阶段 B: 留一条件交叉验证 (LOCV)")
print("=" * 60)

df_all = pd.read_csv(HDDM_PARAMS_PATH)
print(f"\n加载 {len(df_all)} 个条件的数据")

if CALIB_PARAMS_PATH.exists():
    calib_df = pd.read_csv(CALIB_PARAMS_PATH)
    calib_params = calib_df.iloc[0].to_dict()
    print(f"使用校准参数: alaph1={calib_params['alaph1']:.3f}, alaph2={calib_params['alaph2']:.3f}")
else:
    calib_params = {"alaph1": 1.5, "alaph2": -0.4, "beta1": 0.2, "beta2": 0.0,
                    "gamma": 0.2, "base_scale_v": 3.0, "base_scale_a": 3.0}
    print("使用默认 Sigmoid 参数")

group_ids = sorted(df_all["group_id"].unique())
print(f"实验条件: Group {group_ids}")

all_results = []

for held_out_group in group_ids:
    print(f"\n{'─'*40}")
    print(f"留出 Group {held_out_group}, 用剩余 {len(group_ids)-1} 组训练")

    df_train = df_all[df_all["group_id"] != held_out_group].copy()
    df_test = df_all[df_all["group_id"] == held_out_group].copy()

    model = train_model_from_data(df_train, calib_params=calib_params)

    test_row = df_test.iloc[0]
    v_self_pred, v_stranger_pred, a_pred, t_pred, z_pred = model.predict_params_full(
        test_row["P"], test_row["T_ms"], test_row["W_ms"]
    )
    v_self_pred = np.asarray(v_self_pred).item()
    v_stranger_pred = np.asarray(v_stranger_pred).item()
    a_pred = np.asarray(a_pred).item()
    t_pred = np.asarray(t_pred).item()
    z_pred = np.asarray(z_pred).item()

    result = {
        "held_out_group": held_out_group,
        "P": test_row["P"], "T_ms": test_row["T_ms"], "W_ms": test_row["W_ms"],
        "v_self_real": test_row["v_self_mean"], "v_self_pred": v_self_pred,
        "v_stranger_real": test_row["v_stranger_mean"], "v_stranger_pred": v_stranger_pred,
        "a_real": test_row["a_mean"], "a_pred": a_pred,
        "t_real": test_row["t_mean"], "t_pred": t_pred,
        "z_real": test_row["z_mean"], "z_pred": z_pred,
    }
    result["SPE_v_real"] = result["v_self_real"] - result["v_stranger_real"]
    result["SPE_v_pred"] = result["v_self_pred"] - result["v_stranger_pred"]

    for param in ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]:
        result[f"{param}_error"] = result[f"{param}_real"] - result[f"{param}_pred"]
        result[f"{param}_abs_error"] = abs(result[f"{param}_error"])

    all_results.append(result)

    print(f"  v_self:  真实={result['v_self_real']:.3f}, 预测={result['v_self_pred']:.3f}, "
          f"误差={result['v_self_error']:.3f}")
    print(f"  v_stranger: 真实={result['v_stranger_real']:.3f}, 预测={result['v_stranger_pred']:.3f}, "
          f"误差={result['v_stranger_error']:.3f}")
    print(f"  a:   真实={result['a_real']:.3f}, 预测={result['a_pred']:.3f}, "
          f"误差={result['a_error']:.3f}")
    print(f"  SPE_v: 真实={result['SPE_v_real']:.3f}, 预测={result['SPE_v_pred']:.3f}, "
          f"误差={result['SPE_v_error']:.3f}")

df_results = pd.DataFrame(all_results)
df_results.to_csv(OUT_DATA_DIR / "locv_results.csv", index=False)

print(f"\n{'='*60}")
print("LOCV 汇总结果")
print(f"{'='*60}")

metrics = {}
for param in ["v_self", "v_stranger", "a", "t", "z", "SPE_v"]:
    real = df_results[f"{param}_real"]
    pred = df_results[f"{param}_pred"]
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    mae = np.mean(np.abs(real - pred))
    try:
        corr = np.corrcoef(real, pred)[0, 1]
    except Exception:
        corr = np.nan
    metrics[param] = {"RMSE": rmse, "MAE": mae, "r": corr}
    print(f"  {param:12s}: RMSE={rmse:.4f}, MAE={mae:.4f}, r={corr:.3f}")

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(OUT_DATA_DIR / "locv_metrics.csv")

print(f"\n结果已保存到: {OUT_DATA_DIR}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, (param, ax, ylabel) in enumerate([
    ("v_self", axes[0, 0], "v (Self)"), ("v_stranger", axes[0, 1], "v (Stranger)"),
    ("a", axes[0, 2], "a"), ("t", axes[1, 0], "t (s)"),
    ("z", axes[1, 1], "z"), ("SPE_v", axes[1, 2], "SPE_v"),
]):
    real = df_results[f"{param}_real"]
    pred = df_results[f"{param}_pred"]
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    corr = np.corrcoef(real, pred)[0, 1]

    ax.scatter(real, pred, s=100, c="steelblue", edgecolors="black", zorder=5)
    for j, (_, r) in enumerate(df_results.iterrows()):
        ax.annotate(f"G{r['held_out_group']:.0f}", (real.iloc[j], pred.iloc[j]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    all_vals = np.concatenate([real.values, pred.values])
    lim_min = min(all_vals) * 1.3 if min(all_vals) < 0 else min(all_vals) * 0.7
    lim_max = max(all_vals) * 1.3 if max(all_vals) > 0 else max(all_vals) * 0.7
    lim_min = min(lim_min, lim_max - 0.1)
    lim_max = max(lim_max, lim_min + 0.1)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.5, label="y=x")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Real HDDM")
    ax.set_ylabel("LOCV Predicted")
    ax.set_title(f"{ylabel}  (RMSE={rmse:.3f}, r={corr:.3f})")
    ax.legend(loc="upper left", fontsize=7)

plt.suptitle("Leave-One-Condition-Out Cross-Validation: Real vs Predicted", fontsize=14, y=1.01)
plt.tight_layout()
fig_path = OUT_FIG_DIR / "locv_scatter.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"散点图已保存: {fig_path}")

fig2, ax2 = plt.subplots(figsize=(8, 6))
params = list(metrics.keys())
rmse_vals = [metrics[p]["RMSE"] for p in params]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))
bars = ax2.bar(params, rmse_vals, color=colors, edgecolor="black")
for bar, val in zip(bars, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("RMSE")
ax2.set_title("LOCV Prediction RMSE by Parameter")
plt.tight_layout()
fig2_path = OUT_FIG_DIR / "locv_rmse.png"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"RMSE图已保存: {fig2_path}")

print(f"\n{'='*60}")
print("阶段 B LOCV 完成!")
print(f"{'='*60}")
