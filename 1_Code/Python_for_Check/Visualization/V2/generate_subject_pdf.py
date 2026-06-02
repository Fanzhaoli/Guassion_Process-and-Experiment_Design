"""
Self-Matching Task 全被试可视化 PDF 生成器

遍历所有 88 个被试数据，为每个被试生成以下图表：
  1. RT 时序分布图 (Self vs Stranger)
  2. RT 直方图/KDE (Condition × Identity)
  3. 条件分解柱状图 (ACC / RT / Omission)
  4. CRF-SPE 曲线
  5. Response 键偏好热力图

然后生成跨被试汇总图表：
  6. 各组别 SPE 指标小提琴图
  7. T×W 设计空间气泡图

最后将所有图表整合输出为一个完整 PDF 文件。
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.font_manager as fm

# 配置中文字体（Windows 系统使用 Microsoft YaHei）
for font_name in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)

# ========== 路径配置 ==========
PROJECT_ROOT = Path(r"D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")
RAW_DIR = PROJECT_ROOT / "2_Data" / "Real_Data" / "UnExtact" / "raw"
V2_DIR = PROJECT_ROOT / "1_Code" / "Python_for_Check" / "Visualization" / "V2"
OUT_DIR = V2_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "all_subjects_visualization.pdf"

# ========== 实验参数常量 ==========
CONDITIONS = {
    1: {"P": 0, "T": 0.03, "W": 0.3, "label": "G1_P0_T30_W300"},
    2: {"P": 0, "T": 0.03, "W": 0.6, "label": "G2_P0_T30_W600"},
    3: {"P": 120, "T": 0.03, "W": 0.6, "label": "G3_P120_T30_W600"},
    4: {"P": 120, "T": 0.08, "W": 0.6, "label": "G4_P120_T80_W600"},
    5: {"P": 8, "T": 0.10, "W": 1.1, "label": "G5_P8_T100_W1100"},
    6: {"P": 120, "T": 0.50, "W": 1.5, "label": "G6_P120_T500_W1500"},
    7: {"P": 0, "T": 0.10, "W": 1.1, "label": "G7_P0_T100_W1100"},
    8: {"P": 120, "T": 0.03, "W": 0.8, "label": "G8_P120_T30_W800"},
}
QUALITY_MAP = {1: "exclude", 2: "exclude", 3: "caution", 4: "good",
               5: "good", 6: "good", 7: "good", 8: "good"}
QUALITY_COLORS = {"good": "#4caf50", "caution": "#ff9800", "exclude": "#f44336"}
GROUP_PALETTE = ['#e91e63', '#2196f3', '#4caf50', '#ff9800',
                 '#9c27b0', '#00bcd4', '#ffeb3b', '#795548']


# ========== 数据加载函数 (来自 notebook) ==========
def get_pairing_rules(subject_id):
    mod = subject_id % 4
    rules = {
        0: {"square": {"self": "f", "stranger": "j"},
            "circle": {"self": "j", "stranger": "f"}},
        1: {"square": {"self": "j", "stranger": "f"},
            "circle": {"self": "f", "stranger": "j"}},
        2: {"square": {"self": "j", "stranger": "f"},
            "circle": {"self": "f", "stranger": "j"}},
        3: {"square": {"self": "f", "stranger": "j"},
            "circle": {"self": "j", "stranger": "f"}},
    }
    return rules[mod]


def get_match_key(subject_id):
    return ["f", "j", "j", "f"][(subject_id - 1) % 4]


def get_correct_order(subject_id):
    if subject_id % 2 == 0:
        return {"square": "self", "circle": "stranger"}
    return {"square": "stranger", "circle": "self"}


def compute_condition(shape, label, subject_id):
    expected = get_correct_order(subject_id)[shape]
    return "Matching" if label == expected else "NonMatching"


def parse_file_ids(path):
    gid, sid = path.stem.replace("EXP_data_group", "").split("_")[:2]
    return int(gid), int(sid)


def load_one_file(path):
    gid_f, sid_f = parse_file_ids(path)
    df = pd.read_csv(path)
    df["source_file"] = path.name
    df["groupID"] = df["groupID"].astype(int)
    df["subjectID"] = df["subjectID"].astype(int)
    df["trialID"] = df["trialID"].astype(int)
    df["Shape"] = df["Shape"].astype(str).str.strip().str.lower()
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["Response"] = df["Response"].astype(str).str.strip().str.lower()
    df["CorrectKey"] = df["CorrectKey"].astype(str).str.strip().str.lower()
    df["stage"] = df["stage"].fillna("formal").astype(str)
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")
    df["responded"] = df["RT"].notna() & (~df["Response"].isin(["na", "nan", ""]))

    df["Condition"] = [compute_condition(sh, lb, int(sid))
                       for sh, lb, sid in zip(df["Shape"], df["Label"], df["subjectID"])]
    df["Identity"] = np.where(df["Label"].eq("self"), "Self", "Stranger")
    df["MatchKey"] = [get_match_key(int(sid)) for sid in df["subjectID"]]
    df["ResponseIsMatch"] = np.where(
        df["responded"], df["Response"].eq(df["MatchKey"]), np.nan
    )

    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["W"] = pd.to_numeric(df["W"], errors="coerce")
    df["T_ms"] = df["T"] * 1000
    df["W_ms"] = df["W"] * 1000
    df["M_ms"] = df["T_ms"] + df["W_ms"]
    df["RT_ms"] = df["RT"] * 1000
    df["quality"] = df["groupID"].map(QUALITY_MAP)
    df["groupID_from_file"] = gid_f
    df["subjectID_from_file"] = sid_f
    return df


def load_all_raw():
    files = sorted(RAW_DIR.glob("EXP_data_group*.csv"))
    if not files:
        raise FileNotFoundError(f"No EXP_data_group*.csv found in {RAW_DIR}")
    return pd.concat([load_one_file(f) for f in files], ignore_index=True)


# ========== 汇总与分析函数 ==========
def subject_summary_table(df):
    f = df[df["stage"].eq("formal")].copy()
    rows = []
    for keys, g in f.groupby(["Identity", "Condition"], observed=True):
        rr = g[g["responded"]]
        cc = rr[rr["Correct"].eq(1)]
        rows.append({
            "Identity": keys[0],
            "Condition": keys[1],
            "n_trials": len(g),
            "n_resp": len(rr),
            "omission_rate": 1 - len(rr) / max(len(g), 1),
            "accuracy": rr["Correct"].mean() if len(rr) else np.nan,
            "rt_mean_ms": rr["RT_ms"].mean(),
            "rt_median_ms": rr["RT_ms"].median(),
            "correct_rt_mean_ms": cc["RT_ms"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["Identity", "Condition"])


def compute_crf(trials, n_quantiles=5):
    d = trials[
        (trials["stage"].eq("formal"))
        & (trials["responded"])
        & trials["RT"].notna()
    ].copy()
    d = d.sort_values("RT").reset_index(drop=True)
    if len(d) < n_quantiles * 2:
        return pd.DataFrame()
    bins = []
    q_size = len(d) // n_quantiles
    start = 0
    for i in range(n_quantiles):
        end = len(d) if i == n_quantiles - 1 else start + q_size
        b = d.iloc[start:end]
        p = b["ResponseIsMatch"].astype(float).mean()
        sd = b["ResponseIsMatch"].astype(float).std(ddof=1)
        bins.append({
            "bin": i + 1,
            "n": len(b),
            "rt_mean": b["RT"].mean(),
            "rt_mean_ms": b["RT_ms"].mean(),
            "upper_prop": p,
            "sem": sd / math.sqrt(len(b)) if len(b) > 1 else 0,
        })
        start = end
    return pd.DataFrame(bins)


def compute_spe_crf(trials, n_quantiles=5):
    crf_s = compute_crf(trials[trials["Identity"].eq("Self")], n_quantiles)
    crf_st = compute_crf(trials[trials["Identity"].eq("Stranger")], n_quantiles)
    m = min(len(crf_s), len(crf_st))
    if m == 0:
        return crf_s, crf_st, pd.DataFrame()
    spe = pd.DataFrame({
        "bin": np.arange(1, m + 1),
        "rt_mean_ms": (crf_s["rt_mean_ms"].iloc[:m].to_numpy()
                       + crf_st["rt_mean_ms"].iloc[:m].to_numpy()) / 2,
        "spe_upper_prop": (crf_s["upper_prop"].iloc[:m].to_numpy()
                           - crf_st["upper_prop"].iloc[:m].to_numpy()),
        "spe_sem": np.sqrt(crf_s["sem"].iloc[:m].to_numpy() ** 2
                           + crf_st["sem"].iloc[:m].to_numpy() ** 2),
    })
    return crf_s, crf_st, spe


def safe_mean(s):
    return s.mean() if len(s) else np.nan


def summarize_subject(g):
    formal = g[g["stage"].eq("formal")]
    responded = formal[formal["responded"]]
    correct = responded[responded["Correct"].eq(1)]
    acc = (
        correct.groupby("Identity")["Correct"]
        .apply(lambda x: x.sum() / max(len(responded[responded["Identity"].eq(x.name)]), 1))
        if len(responded) else pd.Series(dtype=float)
    )
    rt = (
        responded.groupby("Identity")["RT_ms"].mean()
        if len(responded) else pd.Series(dtype=float)
    )
    return pd.Series({
        "groupID": g["groupID"].iloc[0],
        "subjectID": g["subjectID"].iloc[0],
        "quality": g["quality"].iloc[0],
        "P": g["P"].iloc[0],
        "T_ms": g["T_ms"].iloc[0],
        "W_ms": g["W_ms"].iloc[0],
        "M_ms": g["M_ms"].iloc[0],
        "n_resp": len(responded),
        "n_formal": len(formal),
        "omission_rate": 1 - len(responded) / max(len(formal), 1),
        "accuracy": responded["Correct"].mean(),
        "acc_self": acc.get("Self", np.nan),
        "acc_stranger": acc.get("Stranger", np.nan),
        "SPE_ACC": acc.get("Self", np.nan) - acc.get("Stranger", np.nan),
        "rt_self": rt.get("Self", np.nan),
        "rt_stranger": rt.get("Stranger", np.nan),
        "SPE_RT_ms": rt.get("Stranger", np.nan) - rt.get("Self", np.nan),
    })


# ========== 图表绘制函数 ==========
def plot_rt_timeseries(ax, responded, title):
    """Chart 1: RT 时序分布 (Self vs Stranger)"""
    for identity, color, marker in [
        ("Self", "#ff9800", "o"),
        ("Stranger", "#2196f3", "s"),
    ]:
        d = responded[responded["Identity"].eq(identity)]
        ax.scatter(d["trialID"], d["RT_ms"], s=20, alpha=0.7,
                   label=identity, c=color, marker=marker, edgecolors="none")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("RT (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_rt_histogram(axes, responded, title_prefix):
    """Chart 2: RT 直方图/kde (Condition × Identity)"""
    conditions = ["Matching", "NonMatching"]
    colors = {"Self": "#ff9800", "Stranger": "#2196f3"}
    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        sub = responded[responded["Condition"].eq(cond)]
        for identity in ["Self", "Stranger"]:
            d = sub[sub["Identity"].eq(identity)]["RT_ms"]
            if len(d) > 0:
                ax.hist(d, bins=25, alpha=0.45, label=identity,
                        color=colors[identity], density=True)
                mu, sig = d.mean(), d.std()
                x = np.linspace(max(0, mu - 3.5 * sig), mu + 3.5 * sig, 100)
                y = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                ax.plot(x, y, color=colors[identity], lw=1.8)
        ax.set_title(f"{title_prefix} - {cond}", fontsize=9)
        ax.set_xlabel("RT (ms)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)


def plot_condition_bars(axes, subj_summary, title):
    """Chart 3: 条件分解柱状图 (ACC / RT / Omission)"""
    plot_df = subj_summary.copy()
    plot_df["cell"] = plot_df["Condition"] + "\n" + plot_df["Identity"]
    metrics = [
        (0, "accuracy", 100, "Accuracy (%)", "#4caf50"),
        (1, "rt_mean_ms", 1, "RT Mean (ms)", "#64b5f6"),
        (2, "omission_rate", 100, "Omission (%)", "#f44336"),
    ]
    for idx, col, scale, ylabel, color in metrics:
        ax = axes[idx]
        vals = plot_df[col] * scale
        # Mask NaN
        valid = ~vals.isna()
        ax.bar(plot_df.loc[valid, "cell"], vals[valid],
               color=color, alpha=0.8, edgecolor="white", linewidth=0.8)
        for i, v in enumerate(vals[valid]):
            ax.text(i, v + max(vals[valid]) * 0.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals[valid]) * 1.2 if len(vals[valid]) > 0 else 100)
    axes[0].set_title(title, fontsize=10, fontweight="bold")


def plot_crf_spe(axes, crf_s, crf_st, spe_curve, title):
    """Chart 4: CRF + SPE 曲线"""
    # CRF panel
    ax = axes[0]
    if len(crf_s):
        ax.errorbar(crf_s["rt_mean_ms"], crf_s["upper_prop"], yerr=crf_s["sem"],
                    marker="o", color="#ff9800", label="Self",
                    capsize=3, markersize=6, lw=1.5)
    if len(crf_st):
        ax.errorbar(crf_st["rt_mean_ms"], crf_st["upper_prop"], yerr=crf_st["sem"],
                    marker="o", color="#2196f3", label="Stranger",
                    capsize=3, markersize=6, lw=1.5)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("RT bin mean (ms)", fontsize=8)
    ax.set_ylabel("P(Match Key)", fontsize=8)
    ax.set_title("CRF", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    if len(crf_s) or len(crf_st):
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                        0.5, 1, alpha=0.05, color="#4caf50")
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                        0, 0.5, alpha=0.05, color="#f44336")

    # SPE panel
    ax = axes[1]
    if len(spe_curve):
        ax.errorbar(spe_curve["rt_mean_ms"], spe_curve["spe_upper_prop"],
                    yerr=1.96 * spe_curve["spe_sem"], marker="o", color="#9c27b0",
                    capsize=3, markersize=6, lw=1.5)
        ax.fill_between(
            spe_curve["rt_mean_ms"],
            spe_curve["spe_upper_prop"] - 1.96 * spe_curve["spe_sem"],
            spe_curve["spe_upper_prop"] + 1.96 * spe_curve["spe_sem"],
            alpha=0.15, color="#9c27b0",
        )
    ax.axhline(0, ls="--", color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("RT bin mean (ms)", fontsize=8)
    ax.set_ylabel("Self - Stranger", fontsize=8)
    ax.set_title("CRF-SPE", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25)

    fig = axes[0].figure
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)


def plot_response_heatmap(ax, responded, match_key, title):
    """Chart 5: Response 键偏好热力图"""
    conditions = ["Matching", "NonMatching"]
    identities = ["Self", "Stranger"]
    keys = ["f", "j"]
    data = np.zeros((len(conditions) * len(identities), len(keys) + 1))
    row_labels = []
    for ci, cond in enumerate(conditions):
        for ii, ident in enumerate(identities):
            sub = responded[
                (responded["Condition"].eq(cond))
                & (responded["Identity"].eq(ident))
            ]
            row_idx = ci * 2 + ii
            row_labels.append(f"{cond[:4]}-{ident[:3]}")
            for ki, k in enumerate(keys):
                data[row_idx, ki] = (sub["Response"].eq(k)).sum()
            data[row_idx, 2] = (~sub["responded"]).sum()

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(keys) + 1))
    ax.set_xticklabels(keys + ["miss"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            ax.text(j, i, str(v) if v > 0 else "",
                    ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > data.max() * 0.5 else "black")
    ax.set_title(f"{title}\nMatchKey = {match_key.upper()}", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(axis="x", labelsize=7.5)


def plot_spe_by_group(fig, axes, subject_level):
    """汇总图 1: 各组 SPE 指标小提琴图"""
    colors = {"good": "#4caf50", "caution": "#ff9800", "exclude": "#f44336"}
    metrics = [
        ("SPE_ACC", "SPE Accuracy"),
        ("SPE_RT_ms", "SPE RT (ms)"),
        ("omission_rate", "Omission Rate"),
    ]
    for i, (col, ylabel) in enumerate(metrics):
        ax = axes[i]
        if HAS_SNS:
            sns.violinplot(data=subject_level, x="groupID", y=col,
                          inner=None, color="#d7e8ff", ax=ax, linewidth=0.5)
            sns.stripplot(data=subject_level, x="groupID", y=col, hue="quality",
                         dodge=False, size=5, alpha=0.7, palette=colors, ax=ax,
                         legend=False)
        else:
            for q, c in colors.items():
                d = subject_level[subject_level["quality"].eq(q)]
                ax.scatter(d["groupID"] + np.random.uniform(-0.25, 0.25, len(d)),
                          d[col], s=30, alpha=0.6, c=c, label=q)
                for gid in d["groupID"].unique():
                    gd = d[d["groupID"].eq(gid)][col]
                    if len(gd) > 0:
                        ax.plot([gid - 0.2, gid + 0.2], [gd.median(), gd.median()],
                                color=c, lw=2)
            ax.legend(fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Group ID", fontsize=8)
        ax.axhline(0, ls="--", color="gray", lw=0.8, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Subject-Level SPE & Quality by Design Group",
                 fontsize=11, fontweight="bold")


def plot_design_space(ax, subject_level):
    """汇总图 2: T×W 设计空间气泡图"""
    group_agg = subject_level.groupby("groupID").agg(
        P=("P", "first"),
        T_ms=("T_ms", "first"),
        W_ms=("W_ms", "first"),
        SPE_ACC_mean=("SPE_ACC", "mean"),
        SPE_RT_mean=("SPE_RT_ms", "mean"),
        n=("subjectID", "count"),
        quality=("quality", "first"),
    ).reset_index()

    for _, g in group_agg.iterrows():
        qc = QUALITY_COLORS.get(g["quality"], "#888")
        ax.scatter(g["T_ms"], g["W_ms"],
                   s=max(60, abs(g["SPE_RT_mean"]) * 1.2 + 30),
                   c=qc, alpha=0.7, edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(f"G{int(g['groupID'])}", (g["T_ms"], g["W_ms"]),
                    fontsize=7.5, fontweight="bold", ha="center", va="center",
                    color="white" if qc == "#f44336" else "black")

    ax.set_xlabel("T (ms)", fontsize=9)
    ax.set_ylabel("W (ms)", fontsize=9)
    ax.set_title("Design Space T×W — Group-Level SPE\nBubble size ∝ |SPE_RT|",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, label=q)
        for q, c in QUALITY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left")


def plot_3d_design_space(fig, ax, subject_level):
    """汇总图 3: 三维设计空间 (T, W, SPE)"""
    x = subject_level["T_ms"]
    y = subject_level["W_ms"]
    z = subject_level["SPE_RT_ms"]
    colors = [QUALITY_COLORS.get(q, "#888") for q in subject_level["quality"]]

    sc = ax.scatter(x, y, z, c=colors, s=40, alpha=0.7, edgecolors="white",
                    linewidth=0.5, depthshade=True)
    ax.set_xlabel("T (ms)", fontsize=8)
    ax.set_ylabel("W (ms)", fontsize=8)
    ax.set_zlabel("SPE RT (ms)", fontsize=8)
    ax.set_title("3D Design Space (T × W × SPE_RT)", fontsize=10, fontweight="bold")
    ax.view_init(elev=25, azim=-55)


# ========== 主流程 ==========
def generate_all_charts_pdf():
    print("=" * 60)
    print("  全被试可视化 PDF 生成器")
    print("=" * 60)

    # --- 1. 加载全部数据 ---
    print("\n[1/4] 加载数据...")
    all_df = load_all_raw()
    print(f"  已加载 {len(all_df)} 条试次记录")

    # 获取所有 (groupID, subjectID) 组合
    file_manifest = (
        all_df.groupby(["groupID", "subjectID", "quality"])
        .agg(source_file=("source_file", "first"))
        .reset_index()
        .sort_values(["groupID", "subjectID"])
    )
    print(f"  共 {len(file_manifest)} 个被试文件")

    # --- 预热: 计算所有被试级汇总 (用于汇总图) ---
    print("\n  计算所有被试级汇总指标...")
    subject_level = (
        all_df.groupby(["groupID", "subjectID"], group_keys=False)
        .apply(summarize_subject)
        .reset_index(drop=True)
    )

    # --- 2. 打开 PDF ---
    print("\n[2/4] 开始生成 PDF...")
    with PdfPages(PDF_PATH) as pdf:

        # ----- 封面 -----
        fig = plt.figure(figsize=(10, 7))
        fig.text(0.5, 0.6,
                 "Self-Matching Task\n全被试数据可视化报告",
                 ha="center", va="center", fontsize=26, fontweight="bold")
        fig.text(0.5, 0.45,
                 "88 Subjects × 8 Design Groups\nShape-Label Association Task (Sui et al., 2012)",
                 ha="center", va="center", fontsize=14, color="gray")
        fig.text(0.5, 0.35,
                 f"Generated from {RAW_DIR}",
                 ha="center", va="center", fontsize=10, color="gray")
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ----- 数据总览页 -----
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.5)
        # Group summary table as text
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis("off")
        group_counts = file_manifest.groupby("groupID").size()
        table_data = []
        for gid in range(1, 9):
            cond = CONDITIONS[gid]
            n = group_counts.get(gid, 0)
            q = QUALITY_MAP[gid]
            table_data.append([
                f"G{gid}", f"P={cond['P']}", f"T={cond['T']*1000:.0f}ms",
                f"W={cond['W']*1000:.0f}ms", f"M={cond['W']*1000+cond['T']*1000:.0f}ms",
                str(n), q,
            ])
        tab = ax_table.table(
            cellText=table_data,
            colLabels=["组别", "P(练习)", "T(刺激)", "W(窗口)", "M(总限)", "被试数", "质量"],
            cellLoc="center", loc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1, 1.4)
        ax_table.set_title("实验设计参数总览", fontsize=12, fontweight="bold", y=1.15)

        # Summary histograms
        for i, (col, label) in enumerate([
            ("accuracy", "Accuracy"), ("omission_rate", "Omission Rate"),
            ("SPE_ACC", "SPE Accuracy"), ("SPE_RT_ms", "SPE RT (ms)")
        ]):
            ax = fig.add_subplot(gs[1 + i // 2, i % 2])
            ax.hist(subject_level[col].dropna(), bins=20, color=GROUP_PALETTE[i % 8],
                    edgecolor="white", alpha=0.8)
            ax.axvline(0 if "SPE" in col else subject_level[col].median(),
                      ls="--", color="red" if "SPE" in col else "black", lw=1)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel(label)
            ax.set_ylabel("N subjects")
            ax.grid(True, alpha=0.25)

        for i, (col, label) in enumerate([
            ("SPE_ACC", "SPE Accuracy"),
            ("SPE_RT_ms", "SPE RT (ms)"),
            ("omission_rate", "Omission Rate"),
            ("accuracy", "Accuracy"),
        ]):
            ax = fig.add_subplot(gs[2, i])
            for gid in range(1, 9):
                d = subject_level[subject_level["groupID"].eq(gid)][col].dropna()
                ax.boxplot(
                    [d], positions=[gid], widths=0.5,
                    boxprops=dict(color=GROUP_PALETTE[gid - 1]),
                    whiskerprops=dict(color=GROUP_PALETTE[gid - 1]),
                    medianprops=dict(color="red"), flierprops=dict(markersize=3),
                )
            ax.axhline(0 if "SPE" in col else np.nan, ls="--",
                      color="gray", lw=0.8, alpha=0.5)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Group")
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle("数据总览 — 88 名被试分布", fontsize=13, fontweight="bold",
                     y=1.01)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ----- 跨被试汇总图 1: 小提琴图 -----
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plot_spe_by_group(fig, axes, subject_level)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ----- 跨被试汇总图 2: 设计空间气泡图 -----
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_design_space(ax, subject_level)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ----- 跨被试汇总图 3: 3D 设计空间 -----
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_design_space(fig, ax, subject_level)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ----- 3. 遍历每个被试生成个体图表 -----
        print("\n[3/4] 生成个体被试图表...")
        total = len(file_manifest)
        for idx, (_, row) in enumerate(file_manifest.iterrows(), 1):
            gid = int(row["groupID"])
            sid = int(row["subjectID"])
            fname = row["source_file"]
            quality = row["quality"]

            print(f"  [{idx}/{total}] G{gid}-S{sid} ({quality})", end="")

            # 提取被试数据
            subj = all_df[
                (all_df["groupID"] == gid) & (all_df["subjectID"] == sid)
            ].copy()
            formal = subj[subj["stage"].eq("formal")].copy()
            responded = formal[formal["responded"]].copy()

            if len(formal) == 0:
                print(" — 无正式试次，跳过")
                continue

            title_prefix = f"G{gid}-S{sid}"
            omission = (1 - len(responded) / max(len(formal), 1)) * 100
            acc_val = responded["Correct"].mean() * 100 if len(responded) else 0

            # ----- 个体页面: 2x2 网格布局 -----
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(5, 10, hspace=0.7, wspace=0.5,
                                  height_ratios=[0.3, 1.5, 1.8, 1.8, 1.6])

            # 标题 & 统计信息栏
            ax_info = fig.add_subplot(gs[0, :])
            ax_info.axis("off")
            info_text = (
                f"G{gid}-S{sid}  |  File: {fname}  |  Quality: {quality.upper()}"
                f"  |  MatchKey: {get_match_key(sid).upper()}"
                f"  |  CorrectOrder: {get_correct_order(sid)}"
            )
            stats_text = (
                f"Formal={len(formal)}  |  Responses={len(responded)}  "
                f"|  Omission={omission:.1f}%  |  ACC={acc_val:.1f}%"
            )
            ax_info.text(0.01, 0.7, info_text, transform=ax_info.transAxes,
                        fontsize=8.5, fontweight="bold")
            ax_info.text(0.01, 0.2, stats_text, transform=ax_info.transAxes,
                        fontsize=8, color="gray")

            # Chart 1: RT timeline (top-left, spans 2 cols)
            ax1 = fig.add_subplot(gs[1, :5])
            plot_rt_timeseries(ax1, responded, f"{title_prefix}: RT Timeseries")

            # Chart 2: Response heatmap (top-right)
            ax2 = fig.add_subplot(gs[1, 5:])
            plot_response_heatmap(ax2, responded, get_match_key(sid), title_prefix)

            # Chart 3: RT histogram (mid-left, spans 2 cols)
            ax3_left = fig.add_subplot(gs[2, :3])
            ax3_right = fig.add_subplot(gs[2, 3:6])
            try:
                plot_rt_histogram([ax3_left, ax3_right], responded, title_prefix)
            except Exception as e:
                ax3_left.text(0.5, 0.5, f"RT Hist Error: {e}", transform=ax3_left.transAxes, ha="center")
                ax3_right.text(0.5, 0.5, "Error", transform=ax3_right.transAxes, ha="center")

            # Chart 4: Condition bars (mid-right, spans 2 cols)
            ax4_left = fig.add_subplot(gs[2, 6:8])
            ax4_mid = fig.add_subplot(gs[2, 8])
            ax4_right = fig.add_subplot(gs[2, 9])
            try:
                subj_summ = subject_summary_table(subj)
                plot_condition_bars([ax4_left, ax4_mid, ax4_right],
                                   subj_summ, f"{title_prefix}: Conditions")
            except Exception as e:
                ax4_left.text(0.5, 0.5, f"Bar Error: {e}", transform=ax4_left.transAxes, ha="center")

            # Chart 5: CRF + SPE (bottom, full width)
            ax5_left = fig.add_subplot(gs[3:, :5])
            ax5_right = fig.add_subplot(gs[3:, 5:])
            try:
                crf_s, crf_st, spe_curve = compute_spe_crf(
                    trials_override(subj), n_quantiles=5
                )
                plot_crf_spe([ax5_left, ax5_right], crf_s, crf_st, spe_curve,
                            f"{title_prefix}: CRF & SPE (Q=5)")
                if len(spe_curve) > 0:
                    mean_spe = spe_curve["spe_upper_prop"].mean()
                    print(f" SPE={mean_spe*100:.1f}%", end="")
            except Exception as e:
                ax5_left.text(0.5, 0.5, f"CRF/SPE Error: {e}",
                            transform=ax5_left.transAxes, ha="center", fontsize=9)

            print(" ✓")

            # 保存当前页
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\n[4/4] PDF 生成完毕！")
    print(f"  输出路径: {PDF_PATH}")
    file_size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")
    print("=" * 60)


def trials_override(subj_df):
    """适配 compute_spe_crf 的输入"""
    return subj_df


# ========== 入口 ==========
if __name__ == "__main__":
    generate_all_charts_pdf()
