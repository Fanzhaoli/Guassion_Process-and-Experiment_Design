import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PROJECT_ROOT = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")
DATA_DIR = PROJECT_ROOT / "2_Data" / "Generate_Data" / "CRF_Analysis"
FIG_DIR = PROJECT_ROOT / "3_Figures" / "CRF_Analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

QUALITY_COLORS = {"good": "#2E7D32", "caution": "#F9A825", "exclude": "#C62828"}
IDENTITY_COLORS = {"Self": "#E65100", "Stranger": "#1565C0"}
IDENTITY_MARKERS = {"Self": "o", "Stranger": "s"}


def load_data():
    df = pd.read_csv(DATA_DIR / "trial_level_combined.csv")
    df_valid = df[df["omission"] == 0].copy()
    return df_valid


def process_crf_condition(subset, n_quantiles=5):
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    labels = np.arange(n_quantiles)
    subset = subset.copy()
    subset["rt_bin"] = pd.qcut(subset["rt"], q=n_quantiles, labels=labels)
    grouped = (
        subset.groupby("rt_bin", observed=False)
        .agg(
            rt_mean=("rt", "mean"),
            response_mean=("response", "mean"),
            response_std=("response", "std"),
            response_n=("response", "count"),
        )
        .reset_index()
    )
    grouped["response_sem"] = grouped["response_std"] / np.sqrt(grouped["response_n"])
    return grouped


def style_ax(ax, title=None, xlabel="RT (s)", ylabel="Prop. Upper Boundary (Matching)"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")


def plot_fig1_crf_by_group(data, quality_filter=None):
    """Fig 1A: 8组子图矩阵，每组 Self(橙色) vs Stranger(蓝色) CRF"""
    groups_for_plot = sorted(data["group_id"].unique())
    if quality_filter is not None:
        groups_for_plot = [
            g
            for g in groups_for_plot
            if data[data["group_id"] == g]["data_quality"].iloc[0] in quality_filter
        ]

    n_groups = len(groups_for_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for idx, group_id in enumerate(groups_for_plot):
        ax = axes[idx]
        group_data = data[data["group_id"] == group_id]
        quality = group_data["data_quality"].iloc[0]
        label_info = group_data["GroupLabel"].iloc[0]
        edge_color = QUALITY_COLORS.get(quality, "gray")

        for identity, id_label in [(0, "Self"), (1, "Stranger")]:
            subset = group_data[group_data["identity"] == identity]
            if len(subset) < 20:
                ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
                continue
            crf = process_crf_condition(subset)
            ax.errorbar(
                crf["rt_mean"], crf["response_mean"], yerr=crf["response_sem"],
                fmt=f'-{IDENTITY_MARKERS[id_label]}', color=IDENTITY_COLORS[id_label],
                label=id_label, capsize=3, alpha=0.85, markersize=7, linewidth=2,
            )

        style_ax(ax, title=f"{label_info}  [{quality.upper()}]")
        ax.legend(loc="best", frameon=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2.5)
        ax.patch.set_edgecolor(edge_color)
        ax.patch.set_linewidth(2.5)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Conditional Response Functions: Self vs Stranger by Group",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig1_crf_by_group.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig2_crf_overlay(data, quality_filter=None):
    """Fig 2: 汇总对比 CRF - Self和Stranger分别叠加"""
    df = data.copy()
    if quality_filter is not None:
        df = df[df["data_quality"].isin(quality_filter)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for id_val, id_label, ax in [(0, "Self", axes[0]), (1, "Stranger", axes[1])]:
        subset = df[df["identity"] == id_val]
        groups = sorted(subset["group_id"].unique())
        cmap = plt.cm.viridis
        colors = [cmap(i / max(len(groups) - 1, 1)) for i in range(len(groups))]

        for gi, group_id in enumerate(groups):
            grp_data = subset[subset["group_id"] == group_id]
            if len(grp_data) < 20:
                continue
            crf = process_crf_condition(grp_data)
            quality = grp_data["data_quality"].iloc[0]
            label = grp_data["GroupLabel"].iloc[0]
            ls = "-" if quality == "good" else ("--" if quality == "caution" else ":")
            alpha = 0.9 if quality == "good" else (0.6 if quality == "caution" else 0.4)
            ax.plot(crf["rt_mean"], crf["response_mean"], f'{ls}o',
                    color=colors[gi], label=label, alpha=alpha, markersize=6, linewidth=1.5)

        style_ax(ax, title=f"{id_label} Condition  (across all groups)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("CRF Overlay: Self vs Stranger Across Groups",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig2_crf_overlay.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig3_two_bias_crf(data):
    """Fig 3: 图1风格 - 两种偏差（v-bias vs z-bias）的条件反应函数对比
    
    图1 文献风格说明：
    - 起始点偏差(z-bias)：主要影响较快反应（最短RT分位点处曲线明显偏离基线）
    - 漂移偏差(v-bias)：持续作用于全过程（曲线整体上移/下移，各分位点偏移幅度相近）
    
    在真实数据中的对应：
    - Self vs Stranger 的差异体现 v-bias（漂移率偏差）
    - 不同 z 值的组间差异体现 z-bias（起始点偏差）
    """
    df = data.copy()

    # --- Panel A: v-bias demonstration (Self vs Stranger within G5, the best-quality group) ---
    grp_vbias = 5
    df_vbias = df[df["group_id"] == grp_vbias]
    
    # --- Panel B: z-bias demonstration ---
    # Compare G5 (zr≈0.337, lowest) vs G6 (zr≈0.482, high z), both good quality
    # G5: lower z → bias toward NonMatching in fast RTs
    # G6: higher z → more balanced or Matching-biased in fast RTs
    grp_z_low = 5   # z=0.452, zr≈0.337 → lower starting point
    grp_z_high = 6  # z=0.714, zr≈0.482 → higher starting point
    
    df_z_low = df[df["group_id"] == grp_z_low]
    df_z_high = df[df["group_id"] == grp_z_high]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel A: Drift Bias (v-bias)
    ax = axes[0]
    crf_self_a = process_crf_condition(df_vbias[df_vbias["identity"] == 0])
    crf_stranger_a = process_crf_condition(df_vbias[df_vbias["identity"] == 1])

    ax.errorbar(crf_stranger_a["rt_mean"], crf_stranger_a["response_mean"],
                yerr=crf_stranger_a["response_sem"],
                fmt='-o', color='#9E9E9E', label='Stranger (≈Baseline)',
                capsize=3, alpha=0.8, markersize=8, linewidth=2.5)
    ax.errorbar(crf_self_a["rt_mean"], crf_self_a["response_mean"],
                yerr=crf_self_a["response_sem"],
                fmt='-o', color='#4A90E2', label='Self (Drift Bias)',
                capsize=3, alpha=0.85, markersize=8, linewidth=2.5)
    style_ax(ax, title="A. Drift-rate Bias (v-bias)")
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    # Panel B: Starting Point Bias (z-bias)
    ax = axes[1]
    crf_z_low = process_crf_condition(df_z_low)
    crf_z_high = process_crf_condition(df_z_high)
    
    ax.errorbar(crf_z_low["rt_mean"], crf_z_low["response_mean"],
                yerr=crf_z_low["response_sem"],
                fmt='-o', color='#9E9E9E',
                label=f'G5: Low z (zr≈0.34, Baseline)',
                capsize=3, alpha=0.8, markersize=8, linewidth=2.5)
    ax.errorbar(crf_z_high["rt_mean"], crf_z_high["response_mean"],
                yerr=crf_z_high["response_sem"],
                fmt='-o', color='#FF8C38',
                label=f'G6: High z (zr≈0.48, Start-Pt Bias)',
                capsize=3, alpha=0.85, markersize=8, linewidth=2.5)
    style_ax(ax, title="B. Starting-point Bias (z-bias)")
    ax.legend(loc="lower right", frameon=False, fontsize=12)

    fig.suptitle("Two Types of Decision Bias in CRF Space\n"
                 "Drift Bias (blue): sustained effect on all RT bins  |  "
                 "Start-Pt Bias (orange): strongest at fastest RT bins",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = FIG_DIR / "fig3_two_bias_crf_style.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig3b_spe_crf(data, quality_filter=None):
    """Fig 3B: SPE差值CRF - Self减Stranger的差异曲线"""
    df = data.copy()
    if quality_filter is not None:
        df = df[df["data_quality"].isin(quality_filter)]

    groups = sorted(df["group_id"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for idx, group_id in enumerate(groups):
        ax = axes[idx]
        group_data = df[df["group_id"] == group_id]
        quality = group_data["data_quality"].iloc[0]
        label = group_data["GroupLabel"].iloc[0]
        edge_color = QUALITY_COLORS.get(quality, "gray")

        crf_self = process_crf_condition(group_data[group_data["identity"] == 0])
        crf_stranger = process_crf_condition(group_data[group_data["identity"] == 1])

        merged = pd.merge(crf_self, crf_stranger, on="rt_bin", suffixes=("_self", "_stranger"))
        merged["delta"] = merged["response_mean_self"] - merged["response_mean_stranger"]
        merged["rt_mid"] = (merged["rt_mean_self"] + merged["rt_mean_stranger"]) / 2
        merged["delta_sem"] = np.sqrt(
            merged["response_sem_self"] ** 2 + merged["response_sem_stranger"] ** 2
        )

        ax.fill_between(merged["rt_mid"], -0.5, 0.5, color="gray", alpha=0.05)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.errorbar(merged["rt_mid"], merged["delta"], yerr=merged["delta_sem"],
                    fmt='-o', color='#6A1B9A', capsize=3, alpha=0.85,
                    markersize=7, linewidth=2, label="PSelf−PStranger")
        ax.set_ylim(-0.35, 0.35)
        style_ax(ax, title=f"{label}  [{quality.upper()}]",
                 ylabel="Δ Prop. Matching (Self − Stranger)")
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2.5)
        ax.patch.set_edgecolor(edge_color)
        ax.patch.set_linewidth(2.5)

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("SPE Difference CRF: Self − Stranger Matching Proportion by RT Quantile",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig3b_spe_crf_difference.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_overview_dashboard(data):
    """综合总览面板：将 Self vs Stranger CRF 在同一图中展示（用透明度和颜色区分数据质量）"""
    df = data.copy()
    groups = sorted(df["group_id"].unique())
    n = len(groups)
    n_cols = 4
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 11))
    axes = axes.flatten()

    for idx, group_id in enumerate(groups):
        ax = axes[idx]
        grp = df[df["group_id"] == group_id]
        quality = grp["data_quality"].iloc[0]
        label = grp["GroupLabel"].iloc[0]
        edge_c = QUALITY_COLORS.get(quality, "gray")

        for id_val, id_lbl in [(0, "Self"), (1, "Stranger")]:
            sub = grp[grp["identity"] == id_val]
            if len(sub) < 20:
                continue
            crf = process_crf_condition(sub)
            ax.errorbar(crf["rt_mean"], crf["response_mean"], yerr=crf["response_sem"],
                        fmt=f'-{IDENTITY_MARKERS[id_lbl]}', color=IDENTITY_COLORS[id_lbl],
                        label=id_lbl, capsize=3, alpha=0.85, markersize=7, linewidth=2)

        style_ax(ax, title=label)
        ax.legend(loc="best", frameon=False, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_c)
            spine.set_linewidth(2)
        ax.patch.set_edgecolor(edge_c)
        ax.patch.set_linewidth(2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("CRF Dashboard: Self vs Stranger — All Groups\n"
                 "Border: Green=Good  |  Yellow=Caution  |  Red=Exclude",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig_dashboard_overview.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    print("=" * 60)
    print("Step 2: CRF Visualization")
    print("=" * 60)
    data = load_data()
    print(f"Loaded {len(data)} valid trials (excl. omissions)")

    # Fig 1: 每组 Self vs Stranger CRF (全部8组，颜色区分质量)
    plot_fig1_crf_by_group(data)

    # Fig 2: 汇总叠加 CRF (仅 good+caution 组)
    plot_fig2_crf_overlay(data, quality_filter=["good", "caution"])

    # Fig 3: 图1风格 - 两种偏差对比
    plot_fig3_two_bias_crf(data)

    # Fig 3B: SPE 差值 CRF
    plot_fig3b_spe_crf(data)

    # 总览面板
    plot_overview_dashboard(data)

    print("\nAll CRF figures generated successfully.")


if __name__ == "__main__":
    main()
