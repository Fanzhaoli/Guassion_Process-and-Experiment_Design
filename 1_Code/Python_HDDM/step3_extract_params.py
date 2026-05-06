import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parents[2]
TRACE_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces"
FIG_DIR = BASE_DIR / "3_Figures" / "HDDM_Results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 3: 提取 HDDM 参数后验分布")
print("=" * 60)

stats_files = sorted(TRACE_DIR.glob("*_stats.csv"))
pkl_files = sorted(TRACE_DIR.glob("*_traces.pkl"))
npz_files = sorted(TRACE_DIR.glob("*_traces.npz"))

if not stats_files:
    raise FileNotFoundError(
        f"未找到迹线文件！\n"
        f"请先在 Docker 中运行 step2_hddm_fit.py\n"
        f"预期位置: {TRACE_DIR}"
    )

print(f"\n发现 {len(stats_files)} 个 stats 文件")
print(f"Trace 文件: {len(pkl_files)} pkl, {len(npz_files)} npz")

all_params = []

for stats_path in stats_files:
    fname = stats_path.stem.replace("_stats", "")
    print(f"\n处理: {fname}")

    match = re.search(r"group(\d+)_P(\d+)_T(\d+)_W(\d+)", fname)
    if not match:
        print(f"  ⚠️ 无法解析文件名，跳过")
        continue
    group_id = int(match.group(1))
    P_val = int(match.group(2))
    T_val = int(match.group(3))
    W_val = int(match.group(4))

    stats = pd.read_csv(stats_path, index_col=0)

    row = {
        "group_id": group_id,
        "P": P_val,
        "T_ms": T_val,
        "W_ms": W_val,
        "M_ms": T_val + W_val,
    }

    param_map = {
        "v(0)": "v_stranger",
        "v(1)": "v_self",
        "a": "a",
        "t": "t",
        "z": "z",
    }

    for stat_key, label in param_map.items():
        if stat_key in stats.index:
            row[f"{label}_mean"] = float(stats.loc[stat_key, "mean"])
            row[f"{label}_std"] = float(stats.loc[stat_key, "std"])
            row[f"{label}_q025"] = float(stats.loc[stat_key, "2.5q"])
            row[f"{label}_q975"] = float(stats.loc[stat_key, "97.5q"])

    if "v_self_mean" in row and "v_stranger_mean" in row:
        row["SPE_v"] = row["v_self_mean"] - row["v_stranger_mean"]

    pkl_path = TRACE_DIR / f"{fname}_traces.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                traces = pickle.load(f)
            trace_keys = list(traces.keys())
            print(f"  Trace 键名: {trace_keys}")

            for key in trace_keys:
                samples = traces[key]
                key_clean = key.replace("(", "_").replace(")", "").replace(".", "_")
                if key in ["v(0)", "v(1)", "a", "t", "z"]:
                    continue
                if "subj" in key.lower():
                    pass

        except Exception as e:
            print(f"  ⚠️ pkl 加载失败: {e}")

    subj_params = {}
    for col in stats.index:
        col_str = str(col)
        for prefix, param in [("a_subj", "a"), ("t_subj", "t"), ("z_subj", "z"),
                               ("v_subj(0)", "v_stranger"), ("v_subj(1)", "v_self")]:
            if col_str.startswith(prefix):
                subj_params.setdefault(param, []).append(float(stats.loc[col, "mean"]))

    for param, vals in subj_params.items():
        if vals:
            row[f"{param}_subj_mean"] = np.mean(vals)
            row[f"{param}_subj_std"] = np.std(vals)

    if "v_self_mean" in row:
        print(f"  v_self={row['v_self_mean']:.3f}, v_stranger={row['v_stranger_mean']:.3f}")
        print(f"  SPE_v={row.get('SPE_v', np.nan):.3f}")
        print(f"  a={row.get('a_mean', np.nan):.3f}, t={row.get('t_mean', np.nan):.3f}, z={row.get('z_mean', np.nan):.3f}")

    all_params.append(row)

df_params = pd.DataFrame(all_params).sort_values("group_id")
params_path = TRACE_DIR / "all_groups_ddm_params.csv"
df_params.to_csv(params_path, index=False)
print(f"\n参数汇总表已保存到: {params_path}")

print("\n" + "=" * 60)
print("DDM 参数汇总 (后验均值)")
print("=" * 60)
disp_cols = ["group_id", "P", "T_ms", "W_ms", "v_self_mean", "v_stranger_mean",
             "SPE_v", "a_mean", "t_mean", "z_mean"]
avail = [c for c in disp_cols if c in df_params.columns]
print(df_params[avail].round(4).to_string(index=False))

print("\n" + "=" * 60)
print("绘制 DDM 参数可视化...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
if "SPE_v" in df_params.columns:
    valid = df_params.dropna(subset=["SPE_v"])
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid)))
    for i, (_, r) in enumerate(valid.iterrows()):
        lo = r["v_self_mean"] - r["v_self_q025"]
        hi = r["v_self_q975"] - r["v_self_mean"]
        ax.errorbar(i, r["SPE_v"], yerr=[[lo], [hi]],
                    fmt="o", color=colors[i], capsize=5, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("SPE_v (v_self - v_stranger)")
    ax.set_title("SPE in Drift Rate by Group")

ax = axes[0, 1]
if "v_self_mean" in df_params.columns:
    valid = df_params.dropna(subset=["v_self_mean"])
    x = np.arange(len(valid))
    w = 0.35
    ax.bar(x - w/2, valid["v_self_mean"], w, label="Self", alpha=0.8)
    ax.bar(x + w/2, valid["v_stranger_mean"], w, label="Stranger", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("Drift Rate v")
    ax.set_title("Drift Rate by Condition")
    ax.legend()

ax = axes[1, 0]
if "a_mean" in df_params.columns:
    valid = df_params.dropna(subset=["a_mean"])
    x = np.arange(len(valid))
    ax.bar(x, valid["a_mean"], color="steelblue", alpha=0.8)
    ax.errorbar(x, valid["a_mean"],
                yerr=[valid["a_mean"]-valid["a_q025"], valid["a_q975"]-valid["a_mean"]],
                fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("Boundary a")
    ax.set_title("Decision Boundary by Group")

ax = axes[1, 1]
if "t_mean" in df_params.columns:
    valid = df_params.dropna(subset=["t_mean"])
    x = np.arange(len(valid))
    ax.bar(x, valid["t_mean"], color="coral", alpha=0.8)
    ax.errorbar(x, valid["t_mean"],
                yerr=[valid["t_mean"]-valid["t_q025"], valid["t_q975"]-valid["t_mean"]],
                fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("Nondecision Time t (s)")
    ax.set_title("Nondecision Time by Group")

plt.tight_layout()
fig_path = FIG_DIR / "ddm_params_by_group.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"参数图已保存到: {fig_path}")

if "SPE_v" in df_params.columns and "P" in df_params.columns:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, (var, xlabel) in enumerate([
        ("P", "Practice P"),
        ("T_ms", "Presentation Time T (ms)"),
        ("W_ms", "Response Window W (ms)"),
    ]):
        ax = axes2[i]
        valid = df_params.dropna(subset=["SPE_v", var])
        ax.scatter(valid[var], valid["SPE_v"], s=80, c="steelblue", edgecolors="black")
        for _, r in valid.iterrows():
            ax.annotate(f"G{r['group_id']:.0f}", (r[var], r["SPE_v"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SPE_v")
        ax.set_title(f"SPE_v vs {xlabel}")
    plt.tight_layout()
    fig2_path = FIG_DIR / "spe_vs_design_params.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"SPE vs 设计参数图已保存到: {fig2_path}")

print("\n" + "=" * 60)
print("Step 3 完成!")
print("=" * 60)
