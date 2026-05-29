import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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

QUALITY_LABELS = {
    1: "exclude", 2: "exclude", 3: "caution",
    4: "good", 5: "good", 6: "good", 7: "good", 8: "good",
}
QUALITY_COLORS = {"good": "#2E7D32", "caution": "#F9A825", "exclude": "#C62828"}
QUALITY_EDGE = {"good": "#1B5E20", "caution": "#F57F17", "exclude": "#B71C1C"}


def load_params():
    df = pd.read_csv(TRACES_DIR / "all_groups_ddm_params.csv")
    for gid in range(1, 9):
        df.loc[df["group_id"] == gid, "P"] = [0, 0, 120, 120, 8, 120, 120, 120][gid - 1]
        df.loc[df["group_id"] == gid, "T_ms"] = [30, 30, 30, 80, 100, 500, 80, 80][gid - 1]
        df.loc[df["group_id"] == gid, "W_ms"] = [300, 600, 600, 600, 1100, 1500, 800, 800][gid - 1]
        df.loc[df["group_id"] == gid, "quality"] = QUALITY_LABELS[gid]
    df["zr"] = df["z_mean"] / df["a_mean"]
    df["zr_bias"] = df["zr"] - 0.5
    return df


def style_ax(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold")


def plot_fig7a_tw_space(params):
    """T × W 设计空间中的 SPE_v 和 zr_bias 分布"""
    df = params.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin_v, vmax_v = df["SPE_v"].min() - 0.15, df["SPE_v"].max() + 0.15
    vmin_z, vmax_z = df["zr_bias"].min() - 0.05, df["zr_bias"].max() + 0.05
    cmap_v = plt.cm.RdBu_r
    cmap_z = plt.cm.RdYlGn

    for ax, var, vmin, vmax, cmap, title, clabel in [
        (axes[0], "SPE_v", vmin_v, vmax_v, cmap_v,
         "SPE_v (Drift Bias) in T×W Space", "SPE_v = v_self − v_stranger"),
        (axes[1], "zr_bias", vmin_z, vmax_z, cmap_z,
         "zr_bias (Start-Point Bias) in T×W Space", "zr_bias = z/a − 0.5"),
    ]:
        sc = ax.scatter(
            df["T_ms"], df["W_ms"],
            s=df["P"] * 3 + 200,
            c=df[var], cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors="black", linewidths=0.8, alpha=0.9, zorder=5,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(clabel, fontweight="bold", fontsize=11)

        # Annotate each point
        for _, row in df.iterrows():
            gid = int(row["group_id"])
            quality = row["quality"]
            offset_x = 15 if gid not in [7, 8] else -60
            offset_y = 25 if gid not in [7, 8] else -60
            ax.annotate(
                f'G{gid}', (row["T_ms"], row["W_ms"]),
                textcoords="offset points", xytext=(offset_x, offset_y),
                fontsize=9, fontweight="bold",
                color=QUALITY_COLORS[quality],
                arrowprops=dict(arrowstyle="->", color=QUALITY_COLORS[quality],
                                alpha=0.5, lw=0.8),
            )

        # Add P reference via a legend entry
        p_unique = sorted(df["P"].unique())
        for pu in p_unique:
            ax.scatter([], [], s=pu * 3 + 200, edgecolors="gray",
                       facecolors="none", alpha=0.6, label=f"P={int(pu)}")
        ax.legend(title="Practice (P)", loc="upper left", frameon=False, fontsize=9,
                  title_fontsize=10)

        style_ax(ax, title=title)
        ax.set_xlabel("Stimulus Duration T (ms)", fontweight="bold")
        ax.set_ylabel("Response Window W (ms)", fontweight="bold")
        ax.set_xlim(-50, 600)
        ax.set_ylim(100, 1700)

    fig.suptitle("Fig 7A: Experimental Design Space — T×W Projection\n"
                 "Bubble size ∝ Practice (P)  |  Color = Parameter Value",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig7a_design_space_tw.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig7b_param_vs_design(params):
    """DDM参数 vs 实验设计参数的趋势面板"""
    df = params.copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    param_configs = [
        ("v_self_mean", "v_self", "T_ms", "T (ms)", "b", axes[0, 0],
         "v_self vs T"),
        ("v_stranger_mean", "v_stranger", "T_ms", "T (ms)", "b", axes[0, 1],
         "v_stranger vs T"),
        ("SPE_v", "SPE_v", "T_ms", "T (ms)", "b", axes[0, 2],
         "SPE_v vs T"),
        ("z_mean", "z", "W_ms", "W (ms)", "b", axes[1, 0],
         "z vs W"),
        ("zr_bias", "zr_bias", "W_ms", "W (ms)", "b", axes[1, 1],
         "zr_bias vs W"),
        ("SPE_v", "SPE_v", "P", "P (trials)", "b", axes[1, 2],
         "SPE_v vs P"),
    ]

    for var, label, xvar, xlabel, color, ax, title in param_configs:
        for _, row in df.iterrows():
            quality = row["quality"]
            c = QUALITY_COLORS[quality]
            marker = "o" if quality == "good" else ("s" if quality == "caution" else "x")
            alpha = 0.9 if quality == "good" else (0.7 if quality == "caution" else 0.4)
            size = 120 if quality == "good" else (80 if quality == "caution" else 60)
            ax.scatter(row[xvar], row[var], s=size, c=c, marker=marker,
                       alpha=alpha, edgecolors="black", linewidths=0.5, zorder=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        style_ax(ax, title=title)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(label, fontweight="bold")

    # Add legend
    for quality, color in [("good", "#2E7D32"), ("caution", "#F9A825"), ("exclude", "#C62828")]:
        axes[1, 2].scatter([], [], c=color, s=80, label=quality, edgecolors="black", linewidths=0.5)
    axes[1, 2].legend(frameon=False, fontsize=9, loc="upper left", title="Data Quality")

    fig.suptitle("Fig 7B: DDM Parameters vs Experimental Design Variables\n"
                 "Marker: ○=Good, □=Caution, ×=Exclude",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig7b_param_vs_design.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_fig7c_3d_summary(params):
    """Fig 7C: 综合设计空间概览——T×W气泡图同时展示SPE_v和zr_bias方向"""
    df = params.copy()

    fig, ax = plt.subplots(figsize=(11, 8))

    for _, row in df.iterrows():
        quality = row["quality"]
        gid = int(row["group_id"])
        c = QUALITY_COLORS[quality]
        spe = row["SPE_v"]
        zr_b = row["zr_bias"]

        # Arrow direction: x=SPE_v, y=zr_bias
        angle = np.arctan2(zr_b, spe) * 180 / np.pi
        color_arrow = "#4A90E2" if spe > 0 else "#C62828"

        ax.annotate(
            "", xy=(row["T_ms"] + spe * 50, row["W_ms"] + zr_b * 300),
            xytext=(row["T_ms"], row["W_ms"]),
            arrowprops=dict(arrowstyle="->", color=color_arrow, lw=2.5, alpha=0.6),
        )
        ax.scatter(row["T_ms"], row["W_ms"], s=row["P"] * 4 + 300,
                   facecolors=c, edgecolors=QUALITY_EDGE[quality],
                   linewidths=1.5, alpha=0.85, zorder=5)
        ax.annotate(f'G{gid}', (row["T_ms"], row["W_ms"]),
                    textcoords="offset points", xytext=(12, -15), fontsize=10,
                    fontweight="bold", color=QUALITY_EDGE[quality])

    # Legend for SPE
    ax.annotate("", xy=(550, 1400), xytext=(450, 1400),
                arrowprops=dict(arrowstyle="->", color="#4A90E2", lw=2.5))
    ax.text(520, 1380, "SPE_v > 0\n(Self faster)", fontsize=8, color="#4A90E2", ha="center")
    ax.annotate("", xy=(150, 1400), xytext=(250, 1400),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=2.5))
    ax.text(200, 1380, "SPE_v < 0\n(Self slower)", fontsize=8, color="#C62828", ha="center")

    style_ax(ax, title="Design Space Overview: T×W with v-bias & z-bias Vectors")
    ax.set_xlabel("Stimulus Duration T (ms)", fontweight="bold")
    ax.set_ylabel("Response Window W (ms)", fontweight="bold")
    ax.set_xlim(-80, 650)
    ax.set_ylim(100, 1750)

    out = FIG_DIR / "fig7c_design_space_3d_summary.png"
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    print("=" * 60)
    print("Step 4: Design Space Visualization")
    print("=" * 60)
    params = load_params()
    print(f"Loaded parameters for {len(params)} groups")

    plot_fig7a_tw_space(params)
    plot_fig7b_param_vs_design(params)
    plot_fig7c_3d_summary(params)

    print("\nDesign space visualization complete.")


if __name__ == "__main__":
    main()
