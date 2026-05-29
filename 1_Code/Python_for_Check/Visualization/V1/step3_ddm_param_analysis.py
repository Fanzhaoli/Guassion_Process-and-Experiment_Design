import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")
DATA_DIR = PROJECT_ROOT / "2_Data" / "Generate_Data" / "CRF_Analysis"
TRACES_DIR = PROJECT_ROOT / "2_Data" / "Real_Data" / "HDDM_Traces"
FIG_DIR = PROJECT_ROOT / "3_Figures" / "CRF_Analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "figure.dpi": 150,
})

QUALITY_COLORS = {"good": "#2E7D32", "caution": "#F9A825", "exclude": "#C62828"}

QUALITY_LABELS = {
    1: "exclude", 2: "exclude", 3: "caution",
    4: "good", 5: "good", 6: "good", 7: "good", 8: "good",
}

DESIGN_MAP = {
    1: {"P": 0, "T": 30, "W": 300, "M": 330, "Label": "G1 | P0_T30_W300"},
    2: {"P": 0, "T": 30, "W": 600, "M": 630, "Label": "G2 | P0_T30_W600"},
    3: {"P": 120, "T": 30, "W": 600, "M": 630, "Label": "G3 | P120_T30_W600"},
    4: {"P": 120, "T": 80, "W": 600, "M": 680, "Label": "G4 | P120_T80_W600"},
    5: {"P": 8, "T": 100, "W": 1100, "M": 1200, "Label": "G5 | P8_T100_W1100"},
    6: {"P": 120, "T": 500, "W": 1500, "M": 2000, "Label": "G6 | P120_T500_W1500"},
    7: {"P": 120, "T": 80, "W": 800, "M": 880, "Label": "G7 | P120_T80_W800"},
    8: {"P": 120, "T": 80, "W": 800, "M": 880, "Label": "G8 | P120_T80_W800"},
}


def load_params():
    df = pd.read_csv(TRACES_DIR / "all_groups_ddm_params.csv")
    for gid in range(1, 9):
        df.loc[df["group_id"] == gid, "P"] = DESIGN_MAP[gid]["P"]
        df.loc[df["group_id"] == gid, "T_ms"] = DESIGN_MAP[gid]["T"]
        df.loc[df["group_id"] == gid, "W_ms"] = DESIGN_MAP[gid]["W"]
        df.loc[df["group_id"] == gid, "M_ms"] = DESIGN_MAP[gid]["M"]
        df.loc[df["group_id"] == gid, "Label"] = DESIGN_MAP[gid]["Label"]
        df.loc[df["group_id"] == gid, "data_quality"] = QUALITY_LABELS[gid]

    df["zr_mean"] = df["z_mean"] / df["a_mean"]
    df["zr_q025"] = df["z_q025"] / df["a_q025"]
    df["zr_q975"] = df["z_q975"] / df["a_q975"]
    df["zr_bias"] = df["zr_mean"] - 0.5
    df["v_stranger_abs"] = df["v_stranger_mean"].abs()
    df["v_self_abs"] = df["v_self_mean"].abs()
    return df


def style_ax(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold")


def plot_fig4_v_param_forest(params):
    """Fig 4: v_self 和 v_stranger 森林图（含95% CI）"""
    df = params.copy()
    groups = sorted(df["group_id"].unique())
    colors = [QUALITY_COLORS[QUALITY_LABELS[g]] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, var, title in [
        (axes[0], "v_stranger", "v (Stranger)"),
        (axes[1], "v_self", "v (Self)"),
    ]:
        y_pos = np.arange(len(groups))
        means = df[f"{var}_mean"].values
        q025 = df[f"{var}_q025"].values
        q975 = df[f"{var}_q975"].values
        err_low = means - q025
        err_high = q975 - means

        for i in range(len(groups)):
            ax.errorbar(means[i], y_pos[i], xerr=[[max(0, err_low[i])], [max(0, err_high[i])]],
                        fmt='o', capsize=4, markersize=10, linewidth=2,
                        color=colors[i], alpha=0.8)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_yticks(y_pos)
        labels = [f"G{g}" for g in groups]
        for i, (g, c) in enumerate(zip(groups, colors)):
            labels[i] = f"G{g} [{QUALITY_LABELS[g][0].upper()}]"
        ax.set_yticklabels(labels)
        style_ax(ax, title=title)
        ax.set_xlabel("Drift Rate v (s$^{-1}$)", fontweight="bold")

    fig.suptitle("Fig 4: DDM Drift Rate Parameters — Forest Plot with 95% CI",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig4_v_param_forest.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig5_z_param_forest(params):
    """Fig 5: z 参数森林图 + zr_bias 分析

    z参数解读：
    - z 是DDM中的绝对起始点，范围在 [0, a] 之间
    - zr = z/a 是相对起始点（relative starting point）
    - zr = 0.5 表示无偏向（起始点居中）
    - zr > 0.5 表示起始点偏向 Matching（上边界）
    - zr < 0.5 表示起始点偏向 NonMatching（下边界）
    - 当前HDDM模型中 z 不依赖 identity（Self/Stranger 共享同一 z）
    - 因此 z-bias 分析限于组间比较，无法区分 Self/Stranger 内的偏向差异
    """
    df = params.copy()
    groups = sorted(df["group_id"].unique())
    colors = [QUALITY_COLORS[QUALITY_LABELS[g]] for g in groups]
    y_pos = np.arange(len(groups))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # Panel A: z absolute value forest plot
    ax = axes[0]
    means = df["z_mean"].values
    q025 = df["z_q025"].values
    q975 = df["z_q975"].values

    for i in range(len(groups)):
        ax.errorbar(means[i], y_pos[i], xerr=[[max(0, means[i] - q025[i])], [max(0, q975[i] - means[i])]],
                    fmt='o', capsize=4, markersize=10, linewidth=2,
                    color=colors[i], alpha=0.8)
    a_vals = df["a_mean"].values
    a_2_vals = a_vals / 2
    ax.scatter(a_vals, y_pos, marker='|', color='gray', s=100, alpha=0.4, zorder=2)
    ax.scatter(a_2_vals, y_pos, marker='|', color='black', s=80, alpha=0.3, zorder=2,
               label='a/2 (neutral)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"G{g}" for g in groups])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_ax(ax, title="z (Absolute Starting Point)")
    ax.set_xlabel("Starting Point z", fontweight="bold")

    # Panel B: zr = z/a forest plot
    ax = axes[1]
    zr_mean = df["zr_mean"].values
    zr_q025 = df["zr_q025"].values
    zr_q975 = df["zr_q975"].values

    for i in range(len(groups)):
        ax.errorbar(zr_mean[i], y_pos[i],
                    xerr=[[max(0, zr_mean[i] - zr_q025[i])], [max(0, zr_q975[i] - zr_mean[i])]],
                    fmt='o', capsize=4, markersize=10, linewidth=2,
                    color=colors[i], alpha=0.8)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
               label='zr=0.5 (no bias)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"G{g}" for g in groups])
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, title="zr = z/a (Relative Start Point)")
    ax.set_xlabel("Relative Starting Point zr", fontweight="bold")

    # Panel C: zr_bias + SPE_v bar chart
    ax = axes[2]
    x = np.arange(len(groups))
    width = 0.35
    bars1 = ax.bar(x - width / 2, df["SPE_v"].values, width,
                   color="#4A90E2", alpha=0.85, label="SPE_v (v_self−v_stranger)")
    bars2 = ax.bar(x + width / 2, df["zr_bias"].values, width,
                   color="#FF8C38", alpha=0.85, label="zr_bias (zr−0.5)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}" for g in groups])
    ax.legend(frameon=False, fontsize=10)
    style_ax(ax, title="SPE_v vs zr_bias by Group")
    ax.set_ylabel("Bias Magnitude", fontweight="bold")

    for bar, val in zip(bars1, df["SPE_v"].values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03 * np.sign(bar.get_height()),
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=7, fontweight='bold')
    for bar, val in zip(bars2, df["zr_bias"].values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * np.sign(bar.get_height()),
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=7, fontweight='bold')

    fig.suptitle("Fig 5: DDM Starting Point (z) Analysis\n"
                 "Note: z is estimated at group level (shared across Self/Stranger identity)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig5_z_param_forest.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig6_spe_vs_zr(params):
    """Fig 6: SPE_v vs zr_bias 散点图"""
    df = params.copy()
    colors = [QUALITY_COLORS[QUALITY_LABELS[g]] for g in df["group_id"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df["SPE_v"], df["zr_bias"], s=250, c=colors, alpha=0.85,
                    edgecolors="black", linewidths=0.8, zorder=5)

    for _, row in df.iterrows():
        gid = int(row["group_id"])
        ax.annotate(f'G{gid}', (row["SPE_v"], row["zr_bias"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=11,
                    fontweight="bold", color=QUALITY_COLORS[QUALITY_LABELS[gid]])

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between([df["SPE_v"].min() - 0.2, df["SPE_v"].max() + 0.2],
                    -0.3, 0, alpha=0.03, color="red", label="Bias toward NonMatching")
    ax.fill_between([df["SPE_v"].min() - 0.2, df["SPE_v"].max() + 0.2],
                    0, 0.3, alpha=0.03, color="blue", label="Bias toward Matching")

    style_ax(ax, title="SPE_v (Drift Bias) vs zr_bias (Start-Point Bias)")
    ax.set_xlabel("SPE_v = v_self − v_stranger", fontweight="bold")
    ax.set_ylabel("zr_bias = z/a − 0.5", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Add quadrant labels
    ax.text(df["SPE_v"].max() * 0.65, 0.18, "Q1: +v, +z\n(Matching biased)", 
            fontsize=9, ha='center', alpha=0.5)
    ax.text(df["SPE_v"].min() * 0.65, 0.18, "Q2: −v, +z\n(Conflicting)", 
            fontsize=9, ha='center', alpha=0.5)
    ax.text(df["SPE_v"].max() * 0.65, -0.22, "Q3: +v, −z\n(Conflicting)", 
            fontsize=9, ha='center', alpha=0.5)
    ax.text(df["SPE_v"].min() * 0.65, -0.22, "Q4: −v, −z\n(NonMatching biased)", 
            fontsize=9, ha='center', alpha=0.5)

    out = FIG_DIR / "fig6_spe_vs_zr_scatter.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def export_param_table(params):
    """导出增强版参数表"""
    cols = [
        "group_id", "Label", "data_quality",
        "v_self_mean", "v_self_q025", "v_self_q975",
        "v_stranger_mean", "v_stranger_q025", "v_stranger_q975",
        "SPE_v", "a_mean", "t_mean", "z_mean",
        "zr_mean", "zr_q025", "zr_q975", "zr_bias",
        "P", "T_ms", "W_ms", "M_ms",
    ]
    out = DATA_DIR / "ddm_params_enhanced.csv"
    params[cols].to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


def main():
    print("=" * 60)
    print("Step 3: DDM Parameter Bias Analysis")
    print("=" * 60)
    params = load_params()
    print(f"Loaded parameters for {len(params)} groups")

    plot_fig4_v_param_forest(params)
    plot_fig5_z_param_forest(params)
    plot_fig6_spe_vs_zr(params)
    export_param_table(params)

    print("\nDDM parameter analysis complete.")


if __name__ == "__main__":
    main()
