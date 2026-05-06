import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
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
npz_files = sorted(TRACE_DIR.glob("*_traces.npz"))

print(f"\n发现 {len(stats_files)} 个 stats 文件, {len(npz_files)} 个 trace 文件")

all_params = []

for stats_path, npz_path in zip(stats_files, npz_files):
    fname = stats_path.stem.replace("_stats", "")
    print(f"\n处理: {fname}")

    match = re.search(r"group(\d+)_P(\d+)_T(\d+)_W(\d+)", fname)
    if match:
        group_id = int(match.group(1))
        P_val = int(match.group(2))
        T_val = int(match.group(3))
        W_val = int(match.group(4))
    else:
        print(f"  ⚠️ 无法从文件名解析参数，跳过")
        continue

    stats = pd.read_csv(stats_path, index_col=0)
    print(f"  Stats 中参数: {list(stats.index)}")

    traces = dict(np.load(npz_path, allow_pickle=True))
    trace_keys = list(traces.keys())
    # 扁平化嵌套数组
    for k in trace_keys:
        arr = traces[k]
        if arr.ndim == 0:
            arr = arr.item()
            if isinstance(arr, np.ndarray):
                traces[k] = arr.flatten()
            else:
                traces[k] = np.atleast_1d(arr)
        elif arr.ndim > 1:
            traces[k] = arr.flatten()

    row = {
        "group_id": group_id,
        "P": P_val,
        "T_ms": T_val,
        "W_ms": W_val,
        "M_ms": T_val + W_val,
    }

    for param_name in ["v", "a", "t", "z"]:
        for key in trace_keys:
            key_lower = key.lower()
            if param_name == key_lower and "subj" not in key_lower:
                samples = traces[key]
                row[f"{param_name}_mean"] = float(np.mean(samples))
                row[f"{param_name}_std"] = float(np.std(samples))
                row[f"{param_name}_q025"] = float(np.percentile(samples, 2.5))
                row[f"{param_name}_q975"] = float(np.percentile(samples, 97.5))
                break
        else:
            for col in stats.index:
                col_lower = str(col).lower()
                if param_name == col_lower and "subj" not in col_lower:
                    row[f"{param_name}_mean"] = float(stats.loc[col, "mean"])
                    row[f"{param_name}_std"] = float(stats.loc[col, "std"])
                    row[f"{param_name}_q025"] = float(
                        stats.loc[col, "2.5q"]
                        if "2.5q" in stats.columns
                        else np.nan
                    )
                    row[f"{param_name}_q975"] = float(
                        stats.loc[col, "97.5q"]
                        if "97.5q" in stats.columns
                        else np.nan
                    )
                    break

    for param_name in ["v", "a", "t"]:
        v_self_keys = [
            k for k in trace_keys if param_name in k.lower() and "subj" in k.lower()
        ]
        if v_self_keys:
            all_subj_samples = []
            for k in v_self_keys:
                samples = traces[k]
                mean_sample = np.mean(samples)
                all_subj_samples.append(mean_sample)
            if all_subj_samples:
                row[f"{param_name}_subj_mean_mean"] = float(np.mean(all_subj_samples))
                row[f"{param_name}_subj_mean_std"] = float(np.std(all_subj_samples))

    if any(p in trace_keys for p in ["v(identity_1)", "v(identity_0)"]):
        for identity_key in ["v(identity_1)", "v(identity_0)"]:
            if identity_key in traces:
                label = "v_self" if "1" in identity_key else "v_stranger"
                samples = traces[identity_key]
                row[f"{label}_mean"] = float(np.mean(samples))
                row[f"{label}_std"] = float(np.std(samples))
                row[f"{label}_q025"] = float(np.percentile(samples, 2.5))
                row[f"{label}_q975"] = float(np.percentile(samples, 97.5))

    n_v_keys = [k for k in trace_keys if k.startswith("v") and "subj" not in k]
    has_depends = any("(" in k or "identity" in k.lower() for k in n_v_keys)

    if has_depends and "v_self_mean" not in row:
        v_keys_self = [k for k in n_v_keys if "1" in k or "self" in k.lower()]
        v_keys_stranger = [k for k in n_v_keys if "0" in k or "stranger" in k.lower()]

        if v_keys_self:
            samples = traces[v_keys_self[0]]
            row["v_self_mean"] = float(np.mean(samples))
            row["v_self_q025"] = float(np.percentile(samples, 2.5))
            row["v_self_q975"] = float(np.percentile(samples, 97.5))
        if v_keys_stranger:
            samples = traces[v_keys_stranger[0]]
            row["v_stranger_mean"] = float(np.mean(samples))
            row["v_stranger_q025"] = float(np.percentile(samples, 2.5))
            row["v_stranger_q975"] = float(np.percentile(samples, 97.5))

    if "v_self_mean" not in row and "v_subj_mean_mean" in row:
        row["v_self_mean"] = row.get("v_subj_mean_mean", np.nan)

    if "v_self_mean" in row and "v_stranger_mean" in row:
        row["SPE_v"] = row["v_self_mean"] - row["v_stranger_mean"]

    print(f"  Group{group_id} (P={P_val},T={T_val},W={W_val}): ", end="")
    if "v_self_mean" in row:
        print(f"v_self={row['v_self_mean']:.3f}, v_stranger={row['v_stranger_mean']:.3f}, SPE_v={row.get('SPE_v', np.nan):.3f}")
    else:
        print(f"v={row.get('v_mean', np.nan):.3f}, a={row.get('a_mean', np.nan):.3f}")
    print(f"  trace keys: {trace_keys}")

    all_params.append(row)

df_params = pd.DataFrame(all_params)
df_params = df_params.sort_values("group_id")
params_path = TRACE_DIR / "all_groups_ddm_params.csv"
df_params.to_csv(params_path, index=False)
print(f"\n参数汇总表已保存到: {params_path}")

print("\n" + "=" * 60)
print("DDM 参数汇总 (各条件后验均值)")
print("=" * 60)

display_cols = [
    "group_id",
    "P",
    "T_ms",
    "W_ms",
    "v_self_mean",
    "v_stranger_mean",
    "SPE_v",
    "a_mean",
    "t_mean",
]
available_cols = [c for c in display_cols if c in df_params.columns]
print(df_params[available_cols].to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
if "SPE_v" in df_params.columns:
    valid = df_params.dropna(subset=["SPE_v"])
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid)))
    for i, (_, r) in enumerate(valid.iterrows()):
        ax.errorbar(
            i,
            r["SPE_v"],
            yerr=[
                [r["v_self_mean"] - r["v_self_q025"]],
                [r["v_self_q975"] - r["v_self_mean"]],
            ],
            fmt="o",
            color=colors[i],
            capsize=5,
            markersize=8,
        )
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(
        [f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45
    )
    ax.set_ylabel("SPE_v (v_self - v_stranger)")
    ax.set_title("SPE in Drift Rate by Group")

ax = axes[0, 1]
if "v_self_mean" in df_params.columns:
    valid = df_params.dropna(subset=["v_self_mean"])
    x = np.arange(len(valid))
    w = 0.35
    ax.bar(x - w / 2, valid["v_self_mean"], w, label="Self", alpha=0.8)
    ax.bar(x + w / 2, valid["v_stranger_mean"], w, label="Stranger", alpha=0.8)
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
    ax.errorbar(
        x,
        valid["a_mean"],
        yerr=[valid["a_mean"] - valid["a_q025"], valid["a_q975"] - valid["a_mean"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("Boundary a")
    ax.set_title("Decision Boundary by Group")

ax = axes[1, 1]
if "t_mean" in df_params.columns:
    valid = df_params.dropna(subset=["t_mean"])
    x = np.arange(len(valid))
    ax.bar(x, valid["t_mean"], color="coral", alpha=0.8)
    ax.errorbar(
        x,
        valid["t_mean"],
        yerr=[valid["t_mean"] - valid["t_q025"], valid["t_q975"] - valid["t_mean"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
    ax.set_ylabel("Nondecision Time t0 (s)")
    ax.set_title("Nondecision Time by Group")

plt.tight_layout()
fig_path = FIG_DIR / "ddm_params_by_group.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n参数汇总图已保存到: {fig_path}")

if "SPE_v" in df_params.columns and "P" in df_params.columns:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for i, (var, xlabel) in enumerate(
        [("P", "Practice P"), ("T_ms", "Presentation Time T (ms)"), ("W_ms", "Response Window W (ms)")]
    ):
        ax = axes2[i]
        valid = df_params.dropna(subset=["SPE_v", var])
        ax.scatter(valid[var], valid["SPE_v"], s=80, c="steelblue", edgecolors="black")
        for _, r in valid.iterrows():
            ax.annotate(
                f"G{r['group_id']:.0f}",
                (r[var], r["SPE_v"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
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
