"""
阶段 C: GP 响应面不确定性最大化 — 确定新实验设计点

基于已训练的 GP+Sigmoid 混合模型，在 (P,T,W) 设计空间中搜索 GP 预测
不确定性最大的区域，作为下一轮数据收集的候选实验条件。

策略:
  1. 在全空间 (P,T,W) 上构建密集网格
  2. 对每个点的 self 条件预测总不确定性 (sum of stds across v, a, t, z)
  3. 排除已有 8 个实验条件附近区域
  4. 输出不确定性最大的 Top-N 候选点
  5. 可视化 GP 不确定性响应面
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
from gp_sigmoid_hybrid_model import GPSigmoidHybridModel

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "2_Data" / "Generate_Data" / "GP_Sigmoid" / "gp_sigmoid_hybrid_model.pkl"
HDDM_PARAMS_PATH = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces" / "all_groups_ddm_params.csv"
OUT_DATA_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "External_Verification"
OUT_FIG_DIR = BASE_DIR / "3_Figures" / "External_Verification"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("阶段 C: GP 响应面不确定性最大化")
print("=" * 60)

print("\n[1/4] 加载 GP+Sigmoid 混合模型...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}\n"
                            f"请先运行 GP+Sigmoid/run_full_pipeline.py")
model = GPSigmoidHybridModel.load(MODEL_PATH)
print(f"  模型已加载 (训练于 {len(model._X_train)} 个条件)")

print("\n[2/4] 加载已有实验条件坐标...")
df_existing = pd.read_csv(HDDM_PARAMS_PATH)
existing_points = df_existing[["P", "T_ms", "W_ms"]].values
print(f"  已有 {len(existing_points)} 个实验条件:")
for _, r in df_existing.iterrows():
    print(f"    Group {r['group_id']:.0f}: P={r['P']:.0f}, T={r['T_ms']:.0f}ms, W={r['W_ms']:.0f}ms")

print("\n[3/4] 在 (P,T,W) 空间中密集采样并计算 GP 不确定性...")

P_range = np.linspace(0, 120, 15)
T_range = np.linspace(30, 500, 15)
W_range = np.linspace(300, 1500, 15)

P_grid, T_grid, W_grid = np.meshgrid(P_range, T_range, W_range, indexing="ij")
P_flat = P_grid.ravel()
T_flat = T_grid.ravel()
W_flat = W_grid.ravel()
n_points = len(P_flat)
print(f"  网格采样: {len(P_range)}×{len(T_range)}×{len(W_range)} = {n_points} 个点")

M_flat = T_flat + W_flat

batch_size = 500
all_uncertainty = np.zeros(n_points)
all_v_std = np.zeros(n_points)
all_a_std = np.zeros(n_points)
all_t_std = np.zeros(n_points)
all_z_std = np.zeros(n_points)

for start in range(0, n_points, batch_size):
    end = min(start + batch_size, n_points)
    batch_P = P_flat[start:end]
    batch_T = T_flat[start:end]
    batch_W = W_flat[start:end]
    n_batch = end - start

    X_batch = model.normalize_PTW(batch_P, batch_T, batch_W)

    _, v_std = model.gp_v_self.predict(X_batch, return_std=True)
    _, a_std = model.gp_a.predict(X_batch, return_std=True)
    _, t_std = model.gp_t.predict(X_batch, return_std=True)
    _, z_std = model.gp_z.predict(X_batch, return_std=True)

    all_v_std[start:end] = v_std
    all_a_std[start:end] = a_std
    all_t_std[start:end] = t_std
    all_z_std[start:end] = z_std
    all_uncertainty[start:end] = v_std + a_std + t_std + z_std

print(f"  不确定性范围: {all_uncertainty.min():.4f} ~ {all_uncertainty.max():.4f}")
print(f"  平均不确定性: {all_uncertainty.mean():.4f}")

dist_threshold = 0.15
existing_norm = model.normalize_PTW(
    existing_points[:, 0], existing_points[:, 1], existing_points[:, 2]
)
all_norm = model.normalize_PTW(P_flat, T_flat, W_flat)

too_close = np.zeros(n_points, dtype=bool)
for ep in existing_norm:
    dists = np.sqrt(np.sum((all_norm - ep) ** 2, axis=1))
    too_close |= (dists < dist_threshold)

n_excluded = too_close.sum()
print(f"  排除已有条件附近点: {n_excluded} / {n_points}")

masked_uncertainty = all_uncertainty.copy()
masked_uncertainty[too_close] = -np.inf

top_n = 12
top_indices = np.argsort(masked_uncertainty)[::-1][:top_n]

print(f"\n[4/4] 不确定性最大的 {top_n} 个候选新实验条件:")

candidates = []
for idx in top_indices:
    candidates.append({
        "P": P_flat[idx],
        "T_ms": T_flat[idx],
        "W_ms": W_flat[idx],
        "M_ms": M_flat[idx],
        "total_uncertainty": all_uncertainty[idx],
        "v_std": all_v_std[idx],
        "a_std": all_a_std[idx],
        "t_std": all_t_std[idx],
        "z_std": all_z_std[idx],
    })

df_candidates = pd.DataFrame(candidates)
df_candidates["rank"] = range(1, len(df_candidates) + 1)
df_candidates = df_candidates[["rank", "P", "T_ms", "W_ms", "M_ms",
                                "total_uncertainty", "v_std", "a_std", "t_std", "z_std"]]
df_candidates.to_csv(OUT_DATA_DIR / "optimal_design_points.csv", index=False)

for _, r in df_candidates.iterrows():
    print(f"  #{r['rank']:.0f}: P={r['P']:.0f}, T={r['T_ms']:.0f}ms, W={r['W_ms']:.0f}ms "
          f"(M={r['M_ms']:.0f}ms) → 不确定性={r['total_uncertainty']:.4f}")

print(f"\n完整候选表已保存到: {OUT_DATA_DIR / 'optimal_design_points.csv'}")

print("\n生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax_idx, (var, x_var, y_var, title, ax) in enumerate([
    ("W_flat", "P_flat", "T_flat", "W={W:.0f}ms 截面", axes[0, 0]),
    ("W_flat", "P_flat", "T_flat", "W≈W_median 截面", axes[0, 1]),
    ("T_flat", "P_flat", "W_flat", "T≈T_median 截面", axes[1, 0]),
    ("T_flat", "P_flat", "W_flat", "全局投影", axes[1, 1]),
]):
    if ax_idx == 0:
        w_target = 600.0
        w_mask = np.abs(W_flat - w_target) < 50
        x_vals = P_flat[w_mask]
        y_vals = T_flat[w_mask]
        z_vals = all_uncertainty[w_mask]
        ax.set_xlabel("P"); ax.set_ylabel("T (ms)")
        ax.set_title(f"GP Uncertainty at W≈{w_target:.0f}ms")

    elif ax_idx == 1:
        w_target = np.median(W_range)
        w_mask = np.abs(W_flat - w_target) < 60
        x_vals = P_flat[w_mask]
        y_vals = T_flat[w_mask]
        z_vals = all_uncertainty[w_mask]
        ax.set_xlabel("P"); ax.set_ylabel("T (ms)")
        ax.set_title(f"GP Uncertainty at W≈{w_target:.0f}ms (median)")

    elif ax_idx == 2:
        t_target = np.median(T_range)
        t_mask = np.abs(T_flat - t_target) < 30
        x_vals = P_flat[t_mask]
        y_vals = W_flat[t_mask]
        z_vals = all_uncertainty[t_mask]
        ax.set_xlabel("P"); ax.set_ylabel("W (ms)")
        ax.set_title(f"GP Uncertainty at T≈{t_target:.0f}ms (median)")

    else:
        x_vals = P_flat
        y_vals = W_flat
        z_vals = all_uncertainty
        ax.set_xlabel("P"); ax.set_ylabel("W (ms)")
        ax.set_title("GP Uncertainty in (P, W) space (all T)")

    if len(x_vals) > 0:
        sc = ax.scatter(x_vals, y_vals, c=z_vals, cmap="viridis", s=30, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Total Uncertainty")

    ax.scatter(existing_points[:, 0], existing_points[:, 1] if ax_idx < 2 else existing_points[:, 2],
               c="red", marker="X", s=100, edgecolors="black", linewidths=1.5,
               label="Existing conditions", zorder=10)
    for i, (_, r) in enumerate(df_existing.iterrows()):
        y_pos = r["T_ms"] if ax_idx < 2 else r["W_ms"]
        ax.annotate(f"G{r['group_id']:.0f}", (r["P"], y_pos),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color="red")

    if len(df_candidates) > 0:
        for _, r in df_candidates.head(4).iterrows():
            y_pos = r["T_ms"] if ax_idx < 2 else r["W_ms"]
            ax.scatter(r["P"], y_pos, c="cyan", marker="D", s=80, edgecolors="black",
                      linewidths=1.5, zorder=11)
            ax.annotate(f"#{r['rank']:.0f}", (r["P"], y_pos),
                       textcoords="offset points", xytext=(-10, -10), fontsize=8, color="cyan")

    ax.legend(fontsize=7, loc="upper right")

plt.suptitle("GP Response Surface: Uncertainty Map for External Validation Design", fontsize=14, y=1.01)
plt.tight_layout()
fig_path = OUT_FIG_DIR / "uncertainty_surface.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  不确定性响应面: {fig_path}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
df_sorted = df_candidates.sort_values("total_uncertainty", ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_sorted)))
bars = ax2.barh(
    [f"#{r['rank']:.0f}: P={r['P']:.0f},T={r['T_ms']:.0f},W={r['W_ms']:.0f}" for _, r in df_sorted.iterrows()],
    df_sorted["total_uncertainty"].values, color=colors, edgecolor="black"
)
ax2.set_xlabel("Total GP Uncertainty")
ax2.set_title("Top Candidate Design Points for External Validation")
plt.tight_layout()
fig2_path = OUT_FIG_DIR / "candidate_points_ranking.png"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  候选点排序: {fig2_path}")

print(f"\n{'='*60}")
print("阶段 C 完成!")
print(f"  候选设计点: {OUT_DATA_DIR / 'optimal_design_points.csv'}")
print(f"  图表: {OUT_FIG_DIR}")
print(f"{'='*60}")
