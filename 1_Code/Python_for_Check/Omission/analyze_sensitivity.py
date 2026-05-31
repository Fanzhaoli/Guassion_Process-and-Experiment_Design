import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parents[3]

CENSOR_TRACE_DIR_NEW = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity" / "censor_traces"
CENSOR_TRACE_DIR_OLD = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces"
DROP_TRACE_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity" / "drop_traces"
DATA_SUMMARY_PATH = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity" / "data_summary.csv"

FIG_DIR = BASE_DIR / "3_Figures" / "Omission_Sensitivity"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Omission 敏感性分析 - 参数比较")
print("=" * 60)

data_summary = pd.read_csv(DATA_SUMMARY_PATH)


def load_params_from_stats(stats_dir, prefix_pattern):
    """从 stats 文件提取 DDM 参数"""
    stats_files = sorted(Path(stats_dir).glob("*_stats.csv"))
    if not stats_files:
        return None

    all_params = []
    for stats_path in stats_files:
        fname = stats_path.stem.replace("_stats", "")

        match_dict = {}
        for prefix in prefix_pattern:
            m = re.search(prefix, fname)
            if m:
                match_dict.update(m.groupdict())
        if not match_dict:
            continue

        stats = pd.read_csv(stats_path, index_col=0)

        param_map = {
            "v(0)": "v_stranger",
            "v(1)": "v_self",
            "a": "a",
            "t": "t",
            "z": "z",
        }

        row = {}
        for k, v in match_dict.items():
            try:
                row[k] = int(v)
            except ValueError:
                row[k] = v

        for stat_key, label in param_map.items():
            if stat_key in stats.index:
                row[f"{label}_mean"] = float(stats.loc[stat_key, "mean"])
                row[f"{label}_std"] = float(stats.loc[stat_key, "std"])
                row[f"{label}_q025"] = float(stats.loc[stat_key, "2.5q"])
                row[f"{label}_q975"] = float(stats.loc[stat_key, "97.5q"])

        if "v_self_mean" in row and "v_stranger_mean" in row:
            row["SPE_v"] = row["v_self_mean"] - row["v_stranger_mean"]

        all_params.append(row)

    if not all_params:
        return None

    df = pd.DataFrame(all_params).sort_values("group_id")
    return df


print("\n加载 Censor 方案参数...")

censor_patterns = {
    str(CENSOR_TRACE_DIR_NEW): [r"hddm_data_group(?P<group_id>\d+)_P(?P<P>\d+)_T(?P<T_ms>\d+)_W(?P<W_ms>\d+)_censor"],
    str(CENSOR_TRACE_DIR_OLD): [r"hddm_data_group(?P<group_id>\d+)_P(?P<P>\d+)_T(?P<T_ms>\d+)_W(?P<W_ms>\d+)"],
}

censor_df = None
censor_source = None
for dir_path, patterns in censor_patterns.items():
    censor_df = load_params_from_stats(Path(dir_path), patterns)
    if censor_df is not None:
        censor_source = dir_path
        print(f"  从 {dir_path} 加载了 {len(censor_df)} 组 Censor 参数")
        break

if censor_df is None:
    raise FileNotFoundError(
        f"未找到 Censor 方案结果！\n"
        f"  请先运行: python 1_Code/Python_for_Check/Omission/run_censor_fit.py (Docker)\n"
        f"  或确保 {CENSOR_TRACE_DIR_OLD} 中存在 HDDM 拟合结果"
    )

censor_df["M_ms"] = censor_df["T_ms"] + censor_df["W_ms"]
print(f"  加载 {len(censor_df)} 组 Censor 参数")

print("\n加载 Drop 方案参数...")
drop_df = load_params_from_stats(
    DROP_TRACE_DIR,
    [r"hddm_data_group(?P<group_id>\d+)_P(?P<P>\d+)_T(?P<T_ms>\d+)_W(?P<W_ms>\d+)_drop"],
)

if drop_df is None:
    print("\n" + "!" * 60)
    print("未找到 Drop 方案拟合结果！")
    print("请在 Docker 容器中先运行 run_drop_fit.py：")
    print("  python 1_Code/Python_for_Check/Omission/run_drop_fit.py")
    print("!" * 60)
    drop_df = None

if drop_df is not None:
    drop_df["M_ms"] = drop_df["T_ms"] + drop_df["W_ms"]
    print(f"  加载 {len(drop_df)} 组 Drop 参数")

PARAM_KEYS = ["v_self", "v_stranger", "SPE_v", "a", "t", "z"]
PARAM_LABELS = {
    "v_self": "Drift Rate v_self",
    "v_stranger": "Drift Rate v_stranger",
    "SPE_v": "SPE in Drift Rate (v_self - v_stranger)",
    "a": "Decision Boundary a",
    "t": "Nondecision Time t (s)",
    "z": "Starting Point z",
}

cmp_data = censor_df.merge(
    data_summary[["group_id", "omission_rate", "n_omission", "n_total_trials", "n_drop_trials"]],
    on="group_id",
    how="left",
)

if drop_df is not None:
    cmp_data = cmp_data.merge(
        drop_df,
        on=["group_id", "P", "T_ms", "W_ms", "M_ms"],
        how="inner",
        suffixes=("_censor", "_drop"),
    )

    comparison_rows = []
    for _, row in cmp_data.iterrows():
        gid = row["group_id"]
        for key in PARAM_KEYS:
            c_mean = row.get(f"{key}_mean_censor", np.nan)
            d_mean = row.get(f"{key}_mean_drop", np.nan)
            c_std = row.get(f"{key}_std_censor", np.nan)
            d_std = row.get(f"{key}_std_drop", np.nan)

            c_lo = row.get(f"{key}_q025_censor", np.nan)
            c_hi = row.get(f"{key}_q975_censor", np.nan)
            d_lo = row.get(f"{key}_q025_drop", np.nan)
            d_hi = row.get(f"{key}_q975_drop", np.nan)

            delta = d_mean - c_mean

            ci_overlap = not ((c_hi < d_lo) or (d_hi < c_lo))

            pooled_se = np.sqrt(c_std**2 + d_std**2) if not np.isnan(c_std) and not np.isnan(d_std) else np.nan
            cohens_d = delta / pooled_se if pooled_se and not np.isnan(pooled_se) and pooled_se > 0 else np.nan

            comparison_rows.append({
                "group_id": gid,
                "P": row["P"],
                "T_ms": row["T_ms"],
                "W_ms": row["W_ms"],
                "omission_rate": row["omission_rate"],
                "parameter": key,
                "censor_mean": c_mean,
                "censor_q025": c_lo,
                "censor_q975": c_hi,
                "drop_mean": d_mean,
                "drop_q025": d_lo,
                "drop_q975": d_hi,
                "delta": delta,
                "abs_delta": abs(delta),
                "ci_overlap": ci_overlap,
                "cohens_d": cohens_d,
            })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = OUT_DIR / "sensitivity_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n参数比较表已保存到: {comp_path}")

    print("\n" + "=" * 60)
    print("Omission 处理方法对比: Censor vs Drop")
    print("=" * 60)
    for param in ["v_self", "v_stranger", "SPE_v", "a", "t", "z"]:
        pdata = comp_df[comp_df["parameter"] == param].copy()
        if len(pdata) == 0:
            continue
        print(f"\n--- {PARAM_LABELS[param]} ---")
        print(f"  {'Group':>5s} {'Censor':>8s} {'Drop':>8s} {'Delta':>8s} {'|d|>0.5?':>9s} {'95%CI Overlap':>13s}")
        for _, r in pdata.iterrows():
            flag_big = "YES" if abs(r["delta"]) > 0.5 else ""
            ci_str = "YES" if r["ci_overlap"] else "NO ***"
            print(f"  G{r['group_id']:>4.0f} {r['censor_mean']:>8.3f} {r['drop_mean']:>8.3f} {r['delta']:>8.3f} {flag_big:>9s} {ci_str:>13s}")

    print("\n" + "=" * 60)
    print("Cohen's d 汇总 (Drop - Censor)")
    print("=" * 60)
    for param in ["v_self", "v_stranger", "SPE_v", "a", "t", "z"]:
        pdata = comp_df[comp_df["parameter"] == param]
        d_vals = pdata["cohens_d"].dropna()
        if len(d_vals) > 0:
            print(f"  {PARAM_LABELS[param]}: mean d={d_vals.mean():.3f}, max |d|={d_vals.abs().max():.3f}")


print("\n" + "=" * 60)
print("绘制敏感性分析图表...")
print("=" * 60)

if drop_df is not None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, key in enumerate(PARAM_KEYS):
        ax = axes[i]
        c_mean = cmp_data[f"{key}_mean_censor"].values
        d_mean = cmp_data[f"{key}_mean_drop"].values
        c_lo = cmp_data[f"{key}_q025_censor"].values
        c_hi = cmp_data[f"{key}_q975_censor"].values
        d_lo = cmp_data[f"{key}_q025_drop"].values
        d_hi = cmp_data[f"{key}_q975_drop"].values
        groups = cmp_data["group_id"].values

        x = np.arange(len(groups))
        width = 0.35

        ax.bar(x - width / 2, c_mean, width,
               yerr=[c_mean - c_lo, c_hi - c_mean],
               label="Censor (rt=deadline)", color="#4472C4", alpha=0.85,
               capsize=4, error_kw={"linewidth": 1.5})
        ax.bar(x + width / 2, d_mean, width,
               yerr=[d_mean - d_lo, d_hi - d_mean],
               label="Drop (remove omission)", color="#ED7D31", alpha=0.85,
               capsize=4, error_kw={"linewidth": 1.5})

        ax.set_xticks(x)
        ax.set_xticklabels([f"G{g:.0f}" for g in groups], fontsize=9)
        ax.set_ylabel(PARAM_LABELS[key], fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Omission 敏感性分析: Censor vs Drop 方案下 DDM 参数比较",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "sensitivity_censor_vs_drop_params.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  参数对比图 → {fig_path}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()

    for i, key in enumerate(PARAM_KEYS):
        ax = axes2[i]
        c_mean = cmp_data[f"{key}_mean_censor"].values
        d_mean = cmp_data[f"{key}_mean_drop"].values
        groups = cmp_data["group_id"].values
        omission_rates = cmp_data["omission_rate"].values

        scatter = ax.scatter(c_mean, d_mean, c=omission_rates, s=100,
                             cmap="RdYlGn_r", edgecolors="black", linewidth=0.8,
                             vmin=0, vmax=80)

        for j, gid in enumerate(groups):
            ax.annotate(f"G{gid:.0f}", (c_mean[j], d_mean[j]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        alpha=0.8)

        all_vals = np.concatenate([c_mean, d_mean])
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.1
        vmin -= pad
        vmax += pad
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(f"Censor {key}", fontsize=9)
        ax.set_ylabel(f"Drop {key}", fontsize=9)
        ax.set_title(PARAM_LABELS[key], fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    cbar = fig2.colorbar(scatter, ax=axes2, orientation="vertical",
                          fraction=0.02, pad=0.04)
    cbar.set_label("Omission Rate (%)", fontsize=9)

    fig2.suptitle("Omission 敏感性分析: Censor vs Drop 一致性散点图\n(颜色=遗漏率)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2_path = FIG_DIR / "sensitivity_scatter_censor_vs_drop.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  一致性散点图 → {fig2_path}")

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
    axes3 = axes3.flatten()

    for i, key in enumerate(PARAM_KEYS):
        ax = axes3[i]
        delta_vals = cmp_data[f"{key}_mean_drop"].values - cmp_data[f"{key}_mean_censor"].values
        omission_rates = cmp_data["omission_rate"].values
        groups = cmp_data["group_id"].values

        colors_delta = ["#4472C4" if abs(d) < 0.5 else "#ED7D31" for d in delta_vals]
        ax.bar(np.arange(len(groups)), delta_vals, color=colors_delta, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([f"G{g:.0f}\n{or_:.0f}%" for g, or_ in zip(groups, omission_rates)], fontsize=8)
        ax.set_ylabel(f"Δ {key}", fontsize=10)
        ax.set_title(f"{PARAM_LABELS[key]} (Drop - Censor)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig3.suptitle("Omission 敏感性分析: 参数差异 Δ = Drop - Censor\n(橙色=|Δ|>0.5)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig3_path = FIG_DIR / "sensitivity_delta_params.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  参数差异图 → {fig3_path}")

    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
    axes4 = axes4.flatten()

    for i, key in enumerate(PARAM_KEYS):
        ax = axes4[i]
        delta_vals = cmp_data[f"{key}_mean_drop"].values - cmp_data[f"{key}_mean_censor"].values
        omission_rates = cmp_data["omission_rate"].values
        groups = cmp_data["group_id"].values

        ax.scatter(omission_rates, delta_vals, s=120, c="#4472C4",
                   edgecolors="black", linewidth=0.8, alpha=0.85)

        for j, gid in enumerate(groups):
            ax.annotate(f"G{gid:.0f}", (omission_rates[j], delta_vals[j]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        if len(omission_rates) >= 2:
            z = np.polyfit(omission_rates, delta_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(omission_rates), max(omission_rates), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=1, alpha=0.5,
                    label=f"Trend (slope={z[0]:.4f})")
            ax.legend(fontsize=7)

        ax.set_xlabel("Omission Rate (%)", fontsize=9)
        ax.set_ylabel(f"Δ {key}", fontsize=10)
        ax.set_title(f"{PARAM_LABELS[key]}", fontsize=10)
        ax.grid(alpha=0.3)

    fig4.suptitle("Omission 敏感性分析: 参数差异与遗漏率的关系",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig4_path = FIG_DIR / "sensitivity_delta_vs_omission_rate.png"
    plt.savefig(fig4_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  遗漏率关系图 → {fig4_path}")


fig5, ax5 = plt.subplots(figsize=(12, 5))

groups = np.arange(1, 9)
n_total = []
n_drop = []
n_omission_list = []
omission_rates = []

for _, row in cmp_data.iterrows():
    n_total.append(row["n_total_trials"])
    n_drop.append(row["n_drop_trials"])
    n_omission_list.append(row["n_omission"])
    omission_rates.append(row["omission_rate"])

x = np.arange(len(groups))
width = 0.4

bars_keep = ax5.bar(x - width/2, n_drop, width, label="保留试次 (Drop方案)", color="#4472C4", alpha=0.85)
bars_drop = ax5.bar(x - width/2, n_omission_list, width, bottom=n_drop,
                     label="丢弃试次 (Omission)", color="#D62728", alpha=0.65)
bars_total = ax5.bar(x + width/2, n_total, width, label="总试次 (Censor方案)", color="#7F7F7F", alpha=0.85)

for i, (k, o) in enumerate(zip(n_drop, n_omission_list)):
    total = k + o
    ax5.text(i - width/2, total + 30, f"{o/total*100:.0f}%", ha="center", fontsize=8, fontweight="bold")

ax5.set_xticks(x)
ax5.set_xticklabels([f"G{g}\nP={r['P']:.0f},T={r['T_ms']:.0f},W={r['W_ms']:.0f}"
                      for g, r in zip(groups, [cmp_data.iloc[i] for i in range(len(cmp_data))])], fontsize=8)
ax5.set_ylabel("试次数", fontsize=11)
ax5.set_title("各组 Omission 率与试次分布", fontsize=13, fontweight="bold")
ax5.legend(fontsize=9, loc="upper right")
ax5.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig5_path = FIG_DIR / "omission_summary_by_group.png"
plt.savefig(fig5_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  遗漏率汇总图 → {fig5_path}")


print("\n" + "=" * 60)
print("敏感性分析完成!")
print(f"图表保存于: {FIG_DIR}")
print(f"数据保存于: {OUT_DIR}")
print("=" * 60)

if drop_df is not None:
    print("\n" + "=" * 60)
    print("关键结论速览")
    print("=" * 60)
    for key, label in PARAM_LABELS.items():
        delta_col = f"{key}_mean_drop" if key != "SPE_v" else None
        if key == "SPE_v":
            deltas = cmp_data["SPE_v_drop"].values - cmp_data["SPE_v_censor"].values
        else:
            deltas = cmp_data[f"{key}_mean_drop"].values - cmp_data[f"{key}_mean_censor"].values

        mean_d = np.mean(deltas)
        max_d = np.max(np.abs(deltas))
        sig_count = np.sum(np.abs(deltas) > 0.5)

        print(f"  {label}: 平均Δ={mean_d:.4f}, 最大|Δ|={max_d:.4f}, |Δ|>0.5的组数={sig_count}/{len(deltas)}")