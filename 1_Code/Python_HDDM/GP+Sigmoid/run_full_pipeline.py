import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gp_sigmoid_hybrid_model import GPSigmoidHybridModel, train_model_from_data

BASE_DIR = Path(__file__).resolve().parents[3]
HDDM_PARAMS_PATH = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces" / "all_groups_ddm_params.csv"
CALIB_PARAMS_PATH = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid" / "sigmoid_calibrated_params.csv"
OUT_DATA_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid"
OUT_FIG_DIR = BASE_DIR / "3_Figures" / "GP_Sigmoid"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DATA_DIR / "gp_sigmoid_hybrid_model.pkl"

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("运行 GP+Sigmoid 混合生成模型完整管线")
print("=" * 60)

print("\n[1/5] 加载真实 HDDM 参数...")
df_real = pd.read_csv(HDDM_PARAMS_PATH)
print(f"  加载 {len(df_real)} 个条件:")
print(df_real[["group_id", "P", "T_ms", "W_ms", "v_self_mean", "v_stranger_mean",
              "a_mean", "t_mean", "SPE_v"]].round(3).to_string(index=False))

print("\n[2/5] 加载校准后的 Sigmoid 参数...")
if CALIB_PARAMS_PATH.exists():
    calib_df = pd.read_csv(CALIB_PARAMS_PATH)
    calib_params = calib_df.iloc[0].to_dict()
    print(f"  已加载校准参数: alaph1={calib_params['alaph1']:.3f}, "
          f"alaph2={calib_params['alaph2']:.3f}, gamma={calib_params['gamma']:.3f}")
else:
    print("  未找到校准参数，使用默认参数。请先运行 sigmoid_calibration.py")
    calib_params = {"alaph1": 1.5, "alaph2": -0.4, "beta1": 0.2, "beta2": 0.0,
                    "gamma": 0.2, "base_scale_v": 3.0, "base_scale_a": 3.0}

print("\n[3/5] 训练 GP+Sigmoid 混合模型...")
model = train_model_from_data(df_real, calib_params=calib_params)

print(f"  GP v_self kernel: {model.gp_v_self.kernel_}")
print(f"  GP a kernel:      {model.gp_a.kernel_}")
print(f"  GP t kernel:      {model.gp_t.kernel_}")

model.save(MODEL_PATH)
print(f"  模型已保存到: {MODEL_PATH}")

print("\n[4/5] 在训练点预测并计算残差...")
predictions = []
for _, row in df_real.iterrows():
    v_self_pred, v_stranger_pred, a_pred, t_pred, z_pred = model.predict_params_full(
        row["P"], row["T_ms"], row["W_ms"]
    )
    v_self_pred = np.asarray(v_self_pred).item()
    v_stranger_pred = np.asarray(v_stranger_pred).item()
    a_pred = np.asarray(a_pred).item()
    t_pred = np.asarray(t_pred).item()
    z_pred = np.asarray(z_pred).item()
    spe_v_pred = v_self_pred - v_stranger_pred
    spe_v_real = row["v_self_mean"] - row["v_stranger_mean"]

    predictions.append({
        "group_id": row["group_id"], "P": row["P"], "T_ms": row["T_ms"], "W_ms": row["W_ms"],
        "v_self_real": row["v_self_mean"], "v_self_pred": v_self_pred,
        "v_stranger_real": row["v_stranger_mean"], "v_stranger_pred": v_stranger_pred,
        "a_real": row["a_mean"], "a_pred": a_pred,
        "t_real": row["t_mean"], "t_pred": t_pred,
        "z_real": row["z_mean"], "z_pred": z_pred,
        "SPE_v_real": spe_v_real, "SPE_v_pred": spe_v_pred,
    })

df_pred = pd.DataFrame(predictions)
df_pred.to_csv(OUT_DATA_DIR / "gp_sigmoid_predictions.csv", index=False)

for param in ["v_self", "v_stranger", "a", "t", "z"]:
    real = df_pred[f"{param}_real"]
    pred = df_pred[f"{param}_pred"]
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    corr = np.corrcoef(real, pred)[0, 1]
    print(f"  {param}: RMSE={rmse:.4f}, r={corr:.3f}")

spe_rmse = np.sqrt(np.mean((df_pred["SPE_v_real"] - df_pred["SPE_v_pred"]) ** 2))
spe_corr = np.corrcoef(df_pred["SPE_v_real"], df_pred["SPE_v_pred"])[0, 1]
print(f"  SPE_v: RMSE={spe_rmse:.4f}, r={spe_corr:.3f}")

print("\n[5/5] 生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (param, ax, ylabel) in enumerate([
    ("v_self", axes[0, 0], "v (Self)"), ("v_stranger", axes[0, 1], "v (Stranger)"),
    ("a", axes[0, 2], "a"), ("t", axes[1, 0], "t (s)"),
    ("z", axes[1, 1], "z"), ("SPE_v", axes[1, 2], "SPE_v"),
]):
    real = df_pred[f"{param}_real"] if param != "SPE_v" else df_pred["SPE_v_real"]
    pred = df_pred[f"{param}_pred"] if param != "SPE_v" else df_pred["SPE_v_pred"]
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    corr = np.corrcoef(real, pred)[0, 1]

    ax.scatter(real, pred, s=80, c="steelblue", edgecolors="black", zorder=5)
    for j, (_, r) in enumerate(df_pred.iterrows()):
        ax.annotate(f"G{r['group_id']:.0f}", (real.iloc[j], pred.iloc[j]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    lim_min = min(real.min(), pred.min()) * 1.2
    lim_max = max(real.max(), pred.max()) * 1.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.5, label="y=x")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Real HDDM")
    ax.set_ylabel("GP+Sigmoid Predicted")
    ax.set_title(f"{ylabel}  (RMSE={rmse:.3f}, r={corr:.3f})")
    ax.legend(loc="upper left", fontsize=7)

plt.suptitle("GP+Sigmoid Hybrid Model: Real vs Predicted DDM Parameters", fontsize=14, y=1.01)
plt.tight_layout()
fig_path = OUT_FIG_DIR / "gp_sigmoid_real_vs_pred.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  散点图: {fig_path}")

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

P_vals = df_pred["P"].values
T_vals = df_pred["T_ms"].values
W_vals = df_pred["W_ms"].values

P_grid = np.array([0, 8, 30, 60, 90, 120])
T_grid = np.array([30, 50, 100, 200, 300, 500])
W_grid = np.array([300, 500, 600, 800, 1100, 1500])

for i, (param, ax, ylabel) in enumerate([
    ("v_self", axes2[0, 0], "v (Self)"), ("v_stranger", axes2[0, 1], "v (Stranger)"),
    ("a", axes2[0, 2], "a"), ("t", axes2[1, 0], "t (s)"),
    ("z", axes2[1, 1], "z"), ("SPE_v", axes2[1, 2], "SPE_v"),
]):
    real_vals = df_pred[f"{param}_real"] if param != "SPE_v" else df_pred["SPE_v_real"]
    ax.scatter(P_vals, real_vals, s=50, c="red", marker="o", label="Real", alpha=0.7, edgecolors="black")

    P_test, T_test, W_test = np.meshgrid(P_grid, T_grid, W_grid, indexing="ij")
    P_flat, T_flat, W_flat = P_test.ravel(), T_test.ravel(), W_test.ravel()
    M_flat = T_flat + W_flat

    try:
        v_self_gp, v_stranger_gp, a_gp, t_gp, z_gp = model.predict_params_full(P_flat, T_flat, W_flat)

        if param == "v_self":
            gp_vals = v_self_gp
        elif param == "v_stranger":
            gp_vals = v_stranger_gp
        elif param == "a":
            gp_vals = a_gp
        elif param == "t":
            gp_vals = t_gp
        elif param == "z":
            gp_vals = z_gp
        elif param == "SPE_v":
            gp_vals = v_self_gp - v_stranger_gp

        ax.scatter(P_flat, gp_vals, s=5, c="steelblue", alpha=0.3, label="GP+Sigmoid")
    except Exception as e:
        print(f"  ⚠️ GP预测{param}时出错: {e}")

    ax.set_xlabel("P (Practice)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=7)

plt.suptitle("GP+Sigmoid Response Surface: DDM Parameters vs Practice (P)", fontsize=14, y=1.01)
plt.tight_layout()
fig2_path = OUT_FIG_DIR / "gp_response_surface_P.png"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  响应面图: {fig2_path}")

fig3, ax3 = plt.subplots(figsize=(8, 5))
labels = [f"G{r['group_id']:.0f}" for _, r in df_pred.iterrows()]
x = np.arange(len(labels))
w = 0.35
ax3.bar(x - w/2, df_pred["SPE_v_real"], w, label="Real HDDM", color="steelblue", alpha=0.8)
ax3.bar(x + w/2, df_pred["SPE_v_pred"], w, label="GP+Sigmoid", color="coral", alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45)
ax3.set_ylabel("SPE_v (v_self - v_stranger)")
ax3.set_title(f"SPE_v: Real vs GP+Sigmoid Prediction (RMSE={spe_rmse:.3f})")
ax3.legend()
plt.tight_layout()
fig3_path = OUT_FIG_DIR / "spe_v_comparison.png"
plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  SPE对比图: {fig3_path}")

print(f"\n{'='*60}")
print("GP+Sigmoid 混合模型管线完成!")
print(f"  模型: {MODEL_PATH}")
print(f"  预测: {OUT_DATA_DIR / 'gp_sigmoid_predictions.csv'}")
print(f"  图表: {OUT_FIG_DIR}")
print(f"{'='*60}")
