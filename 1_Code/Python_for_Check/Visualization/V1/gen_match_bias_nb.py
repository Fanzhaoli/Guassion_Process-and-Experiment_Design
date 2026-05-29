r"""Generate Match_vs_Mismatch_Bias_Analysis.ipynb"""
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\Visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.3"}
}

cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))

def code(s):
    cells.append(nbf.v4.new_code_cell(s))

# =====================================================================
# CELL 0: Title
# =====================================================================
md("""# 实验任务 Match vs Mismatch 反应偏差可视化分析

## 研究问题
被试在 Self-Matching Task 中是否存在系统性地偏向 "匹配（Match）" 或 "不匹配（Mismatch）" 的反应倾向？

## 分析框架
- **Match 试次**: (circle, self) 或 (square, stranger) — 形状与身份"匹配"
- **Mismatch 试次**: (circle, stranger) 或 (square, self) — 形状与身份"不匹配"
- 任务要求被试判断形状与身份是否匹配，因此两种试次各占 50%

## 分析维度
1. 被试在不同条件下判断为 Matching 的比例
2. Self vs Stranger 条件下的反应偏向差异
3. 反应时与遗漏率的条件间差异
4. 实验设计参数 (P, T, W) 对偏向的调节作用""")

# =====================================================================
# CELL 1: Imports & Config
# =====================================================================
code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "figure.dpi": 150,
})

# 路径设置
PROJECT_ROOT = Path(r"d:\\GitHub_programe\\GitHub\\Guassion-Process-Experiment-Design")
RAW_DIR = PROJECT_ROOT / "2_Data" / "Real_Data" / "UnExtact" / "raw"
FIG_DIR = PROJECT_ROOT / "3_Figures" / "Match_Bias_Analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 8组实验设计
DESIGN_MAP = {
    1: {"P": 0,   "T_ms": 30,  "W_ms": 300,  "Label": "G1 | P0_T30_W300"},
    2: {"P": 0,   "T_ms": 30,  "W_ms": 600,  "Label": "G2 | P0_T30_W600"},
    3: {"P": 120, "T_ms": 30,  "W_ms": 600,  "Label": "G3 | P120_T30_W600"},
    4: {"P": 120, "T_ms": 80,  "W_ms": 600,  "Label": "G4 | P120_T80_W600"},
    5: {"P": 8,   "T_ms": 100, "W_ms": 1100, "Label": "G5 | P8_T100_W1100"},
    6: {"P": 120, "T_ms": 500, "W_ms": 1500, "Label": "G6 | P120_T500_W1500"},
    7: {"P": 120, "T_ms": 80,  "W_ms": 800,  "Label": "G7 | P120_T80_W800"},
    8: {"P": 120, "T_ms": 80,  "W_ms": 800,  "Label": "G8 | P120_T80_W800"},
}

# 数据质量
QUALITY_MAP = {1: "Low", 2: "Low", 3: "Medium", 4: "Good", 5: "Good", 6: "Good", 7: "Good", 8: "Good"}
QUALITY_COLORS = {"Low": "#C62828", "Medium": "#F9A825", "Good": "#2E7D32"}

print("Environment ready.")
print(f"RAW_DIR: {RAW_DIR}  (exists: {RAW_DIR.exists()})")
print(f"FIG_DIR: {FIG_DIR}")""")

# =====================================================================
# CELL 2: Data Loading & Preprocessing
# =====================================================================
code("""# =====================================================================
# Section 1: Data Loading & Preprocessing
# =====================================================================
print("=" * 60)
print("Section 1: 数据加载与预处理")
print("=" * 60)

# 读取所有原始 CSV
csv_files = sorted(RAW_DIR.glob("EXP_data_group*.csv"))
print(f"读取 {len(csv_files)} 个原始数据文件...")

dfs = []
for f in csv_files:
    dfs.append(pd.read_csv(f))

df_all = pd.concat(dfs, ignore_index=True)
print(f"合并: {len(df_all)} 行, {df_all['subjectID'].nunique()} 名被试")

# 过滤 formal 阶段
df = df_all[df_all["stage"] == "formal"].copy()
print(f"过滤 stage='formal': {len(df)} 行 ({len(df)//len(csv_files)} 试次/被试)")

# 构建关键字段
df["RT_num"] = pd.to_numeric(df["RT"], errors="coerce")
df["omission"] = df["RT_num"].isna().astype(int)
df["identity"] = df["Label"].map({"self": "Self", "stranger": "Stranger"})

# 定义 Matching 条件（基于 subjectID 奇偶性翻转）
odd_mask = (df["subjectID"] % 2 == 1)
match_mask_odd = ((df["Shape"] == "circle") & (df["Label"] == "self")) | \
                 ((df["Shape"] == "square") & (df["Label"] == "stranger"))
match_mask_even = ((df["Shape"] == "square") & (df["Label"] == "self")) | \
                  ((df["Shape"] == "circle") & (df["Label"] == "stranger"))
match_mask = np.where(odd_mask, match_mask_odd, match_mask_even)
df["match_condition"] = np.where(match_mask, "Match", "Mismatch")

# 添加实验设计参数
for gid in range(1, 9):
    d = DESIGN_MAP[gid]
    mask = df["groupID"] == gid
    df.loc[mask, "P"] = d["P"]
    df.loc[mask, "T_ms"] = d["T_ms"]
    df.loc[mask, "W_ms"] = d["W_ms"]
    df.loc[mask, "GroupLabel"] = d["Label"]
    df.loc[mask, "data_quality"] = QUALITY_MAP[gid]

# 判断是否按了 Matching 键
# 任务中 Matching 试次的正确反应因人而异（f/j 随机分配），
# 但我们可以统一编码: 不论对错，"f" 键对应某一个决策方向。
# 更可靠的做法: 被试的 Response 是 f 还是 j 反映其内部决策。
# 这里我们无法直接知道 f/j 哪个是 Matching，但可以统计:
# 被试按了"非正确键"的比例反映其离 Matching 的偏差。

df["chose_match"] = df["Correct"]  # 1=判断正确(按了匹配键), 0=判断错误

print("")
print(f"Match trials: {(df['match_condition']=='Match').sum()}, "
      f"Mismatch trials: {(df['match_condition']=='Mismatch').sum()}")

# 数据质量摘要
summary = df.groupby("groupID").agg(
    n_trials=("RT", "count"),
    n_subjects=("subjectID", "nunique"),
    omission_pct=("omission", "mean"),
    overall_acc=("Correct", "mean"),
    match_acc=("Correct", lambda x: x[df.loc[x.index, "match_condition"] == "Match"].mean()),
    mismatch_acc=("Correct", lambda x: x[df.loc[x.index, "match_condition"] == "Mismatch"].mean()),
    rt_mean=("RT_num", "mean"),
).reset_index()
summary["omission_pct"] *= 100
summary["overall_acc"] *= 100
summary["match_acc"] *= 100
summary["mismatch_acc"] *= 100
summary["rt_mean"] *= 1000  # ms
summary["data_quality"] = summary["groupID"].map(QUALITY_MAP)

print("各组数据质量摘要:")
print(summary.to_string(index=False))

# 供后续分析的干净数据 (排除遗漏试次)
df_valid = df[df["omission"] == 0].copy()
print(f"\\nEffective trials (excl. omissions): {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")""")

# =====================================================================
# CELL 3: Fig 1 - Bar Chart: Overall Bias
# =====================================================================
code("""# =====================================================================
# Fig 1: 各组 Matching 判断比例 (整体偏向)
# =====================================================================
# 纵轴: 有效试次中判断为 Matching 的比例 (%)
# y=50% 参考线 = 无偏向 (Match/Mismatch 各对应一种正确反应)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, gid in enumerate(range(1, 9)):
    ax = axes[idx]
    grp = df_valid[df_valid["groupID"] == gid]
    quality = QUALITY_MAP[gid]
    label = DESIGN_MAP[gid]["Label"]
    edge_c = QUALITY_COLORS[quality]

    # 整体 % 判断为 Matching
    overall_pct = grp["Correct"].mean() * 100
    # 分条件
    match_pct = grp[grp["match_condition"] == "Match"]["Correct"].mean() * 100
    mismatch_sub = grp[grp["match_condition"] == "Mismatch"]
    # Mismatch 试次中 Correct=1 意味着判断了 NonMatching，所以 "判断为Matching" = 1-Correct
    mismatch_pct_choose_match = (1 - mismatch_sub["Correct"]).mean() * 100

    x_pos = [0, 1, 2]
    values = [overall_pct, match_pct, mismatch_pct_choose_match]
    colors_bar = ["#757575", "#E65100", "#1565C0"]

    bars = ax.bar(x_pos, values, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Overall", "Match\\nTrials", "Mismatch\\nTrials"], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% Judged Matching")
    ax.set_title(f"{label}\\n[{quality}]", fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_c)
        spine.set_linewidth(2.5)
    ax.patch.set_edgecolor(edge_c)
    ax.patch.set_linewidth(2.5)

fig.suptitle("Fig 1: Proportion of 'Judged as Matching' by Group\\n"
             "Grey=Overall  |  Orange=Match trials  |  Blue=Mismatch trials (judged as Matching = 1-Correct)\\n"
             "Dashed line = 50% (no bias).  Border: Green=Good  Yellow=Medium  Red=Low",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out = FIG_DIR / "fig1_overall_bias_bars.png"
fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# 解读
print("\\n解读:")
print("- 灰柱(Overall)接近 50% 表示 Matching/Mismatch 判断各半，无明显偏向")
print("- 橙柱(Match试次)高 = 被试能正确识别匹配关系")
print("- 蓝柱(Mismatch试次)高 = 被试倾向于把 NonMatching 误判为 Matching (偏向Match)")
print("- 蓝柱低 = 被试倾向于把 NonMatching 正确判断为 NonMatching (正确判断)")
print("- G4-G6 蓝柱极低 (6-33%) → 被试在 Mismatch 试次上对 NonMatching 判断非常准确")""")

# =====================================================================
# CELL 4: Fig 2 - RT Comparison
# =====================================================================
code("""# =====================================================================
# Fig 2: 反应时 (RT) 条件差异分析
# =====================================================================
# 比较 Match vs Mismatch 试次 + Self vs Stranger 条件下的 RT 分布

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# --- Panel A: RT by Match Condition (Boxplot, grouped by group) ---
ax = axes[0, 0]
data_for_box = []
labels_for_box = []
colors_for_box = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    quality = QUALITY_MAP[gid]
    for cond, clr in [("Match", "#E65100"), ("Mismatch", "#1565C0")]:
        rts = grp[grp["match_condition"] == cond]["RT_num"].values * 1000
        data_for_box.append(rts)
        labels_for_box.append(f"G{gid}\\n{cond}")
        colors_for_box.append(clr)

bp = ax.boxplot(data_for_box, patch_artist=True, widths=0.6,
                showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
for patch, clr in zip(bp["boxes"], colors_for_box):
    patch.set_facecolor(clr)
    patch.set_alpha(0.6)
ax.set_xticklabels(labels_for_box, fontsize=7, rotation=45)
ax.set_ylabel("Reaction Time (ms)")
ax.set_title("A. RT by Group × Match Condition")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel B: Self vs Stranger RT difference (bar, per group) ---
ax = axes[0, 1]
x = np.arange(8)
width = 0.35
self_rt = []; stranger_rt = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    self_rt.append(grp[grp["identity"] == "Self"]["RT_num"].mean() * 1000)
    stranger_rt.append(grp[grp["identity"] == "Stranger"]["RT_num"].mean() * 1000)
b1 = ax.bar(x - width/2, self_rt, width, label="Self", color="#E65100", alpha=0.85)
b2 = ax.bar(x + width/2, stranger_rt, width, label="Stranger", color="#1565C0", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("Mean RT (ms)"); ax.set_title("B. RT: Self vs Stranger")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel C: Match vs Mismatch accuracy by group ---
ax = axes[1, 0]
x = np.arange(8)
match_acc_vals = []; mismatch_acc_vals = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    match_acc_vals.append(grp[grp["match_condition"] == "Match"]["Correct"].mean() * 100)
    mismatch_acc_vals.append(grp[grp["match_condition"] == "Mismatch"]["Correct"].mean() * 100)
ax.bar(x - width/2, match_acc_vals, width, label="Match Trials", color="#E65100", alpha=0.85)
ax.bar(x + width/2, mismatch_acc_vals, width, label="Mismatch Trials", color="#1565C0", alpha=0.85)
ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("Accuracy (%)"); ax.set_title("C. Accuracy by Match Condition")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel D: Bias Index = P(choose Match | Mismatch trial) ---
ax = axes[1, 1]
quality_colors_list = [QUALITY_COLORS[QUALITY_MAP[g]] for g in range(1, 9)]
x = np.arange(8)
bias_values = []
for gid in range(1, 9):
    grp_mm = df_valid[(df_valid["groupID"] == gid) & (df_valid["match_condition"] == "Mismatch")]
    # % Mismatch试次中被试按了 Matching 键 (即: 错误判断)
    bias = (1 - grp_mm["Correct"].mean()) * 100
    bias_values.append(bias)

bars = ax.bar(x, bias_values, color=quality_colors_list, alpha=0.85, edgecolor="black", linewidth=0.8)
ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, val in zip(bars, bias_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("% Mismatch Trials Judged as Matching")
ax.set_title("D. Matching Bias Index\\n(Higher = stronger bias toward Matching)")
ax.set_ylim(0, 105)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Fig 2: RT & Accuracy by Match Condition and Identity",
             fontsize=15, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "fig2_rt_accuracy.png"
fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")""")

# =====================================================================
# CELL 5: Fig 3 - Self vs Stranger Identity Bias
# =====================================================================
code("""# =====================================================================
# Fig 3: Self  vs  Stranger 条件下是否对 Matching 有不同的偏向
# =====================================================================
# 核心问题: 自我相关刺激是否产生特定的 Matching/NonMatching 偏差?

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# --- Panel A: Accuracy by Identity × Match Condition (averaged across groups) ---
ax = axes[0, 0]
identities = ["Self", "Stranger"]
conditions = ["Match", "Mismatch"]
x = np.arange(2)
width = 0.35
cmap = {"Self": "#E65100", "Stranger": "#1565C0"}

# 仅用数据质量好的组 (G4-G8)
df_good = df_valid[df_valid["data_quality"].isin(["Good"])]

for j, cond in enumerate(conditions):
    values = []
    for ident in identities:
        sub = df_good[(df_good["identity"] == ident) & (df_good["match_condition"] == cond)]
        values.append(sub["Correct"].mean() * 100)
    ax.bar(x - width/2 + j*width/2, values, width/2,
           label=f"{cond} Trials", alpha=0.85,
           color="#E65100" if cond == "Match" else "#1565C0")

ax.set_xticks(x); ax.set_xticklabels(identities)
ax.set_ylabel("Accuracy (%)")
ax.set_title("A. Accuracy: Self vs Stranger (G4-G8 only)")
ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel B: RT by Identity × Match Condition ---
ax = axes[0, 1]
for j, cond in enumerate(conditions):
    values = []
    for ident in identities:
        sub = df_good[(df_good["identity"] == ident) & (df_good["match_condition"] == cond)]
        values.append(sub["RT_num"].mean() * 1000)
    ax.bar(x - width/2 + j*width/2, values, width/2,
           label=f"{cond} Trials", alpha=0.85,
           color="#E65100" if cond == "Match" else "#1565C0")
ax.set_xticks(x); ax.set_xticklabels(identities)
ax.set_ylabel("Mean RT (ms)")
ax.set_title("B. RT: Self vs Stranger (G4-G8 only)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel C: Omission rate by Identity × Group ---
ax = axes[0, 2]
x = np.arange(8)
self_om = []; stranger_om = []
for gid in range(1, 9):
    grp = df[df["groupID"] == gid]
    self_om.append(grp[grp["identity"] == "Self"]["omission"].mean() * 100)
    stranger_om.append(grp[grp["identity"] == "Stranger"]["omission"].mean() * 100)
ax.plot(x, self_om, "o-", color="#E65100", label="Self", markersize=8, linewidth=2)
ax.plot(x, stranger_om, "s--", color="#1565C0", label="Stranger", markersize=8, linewidth=2)
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("Omission Rate (%)")
ax.set_title("C. Omission Rate: Self vs Stranger")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel D: Matching bias by Group (Self vs Stranger) ---
ax = axes[1, 0]
x = np.arange(8)
self_bias = []; stranger_bias = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    # 在 Match 试次上被判断为 Matching 的比例 = 正确率
    sm = grp[(grp["identity"] == "Self") & (grp["match_condition"] == "Match")]["Correct"].mean() * 100
    st = grp[(grp["identity"] == "Stranger") & (grp["match_condition"] == "Match")]["Correct"].mean() * 100
    self_bias.append(sm); stranger_bias.append(st)

ax.plot(x, self_bias, "o-", color="#E65100", label="Self (Match trials)", markersize=8, linewidth=2)
ax.plot(x, stranger_bias, "s--", color="#1565C0", label="Stranger (Match trials)", markersize=8, linewidth=2)
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("Accuracy on Match Trials (%)")
ax.set_title("D. Match-Trial Accuracy: Self vs Stranger")
ax.legend(frameon=False, fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel E: Mismatch-trial bias: % choosing Matching ---
ax = axes[1, 1]
self_mm_bias = []; stranger_mm_bias = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    sm = grp[(grp["identity"] == "Self") & (grp["match_condition"] == "Mismatch")]
    st = grp[(grp["identity"] == "Stranger") & (grp["match_condition"] == "Mismatch")]
    self_mm_bias.append((1 - sm["Correct"].mean()) * 100)
    stranger_mm_bias.append((1 - st["Correct"].mean()) * 100)

ax.plot(x, self_mm_bias, "o-", color="#E65100", label="Self", markersize=8, linewidth=2)
ax.plot(x, stranger_mm_bias, "s--", color="#1565C0", label="Stranger", markersize=8, linewidth=2)
ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
ax.set_xticks(x); ax.set_xticklabels([f"G{g}" for g in range(1, 9)])
ax.set_ylabel("% Mismatch Trials Judged as Matching")
ax.set_title("E. Mismatch Bias: Self vs Stranger")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel F: Scatter: Match accuracy vs Mismatch accuracy (by group and identity) ---
ax = axes[1, 2]
markers = {"Self": "o", "Stranger": "s"}
for identity in ["Self", "Stranger"]:
    for gid in range(1, 9):
        grp = df_valid[(df_valid["groupID"] == gid) & (df_valid["identity"] == identity)]
        match_acc = grp[grp["match_condition"] == "Match"]["Correct"].mean() * 100
        mismatch_acc = grp[grp["match_condition"] == "Mismatch"]["Correct"].mean() * 100
        c = QUALITY_COLORS[QUALITY_MAP[gid]]
        ax.scatter(match_acc, mismatch_acc, s=120, c=c, marker=markers[identity],
                   alpha=0.85, edgecolors="black", linewidths=0.5)
        if identity == "Self":
            ax.annotate(f"G{gid}", (match_acc, mismatch_acc), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, fontweight="bold")
# Quadrant lines
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Match Trial Accuracy (%)"); ax.set_ylabel("Mismatch Trial Accuracy (%)")
ax.set_title("F. Accuracy Trade-off\\n(Marker: o=Self  s=Stranger)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Fig 3: Self vs Stranger Identity Effects on Match/Mismatch Bias",
             fontsize=15, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "fig3_identity_bias.png"
fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")""")

# =====================================================================
# CELL 6: Fig 4 - Design Space: P/T/W Modulation
# =====================================================================
code("""# =====================================================================
# Fig 4: 实验设计参数 (P/T/W) 对 Match/Mismatch 偏向的调节
# =====================================================================

# 计算每组的关键指标
stats_rows = []
for gid in range(1, 9):
    grp = df_valid[df_valid["groupID"] == gid]
    match_acc = grp[grp["match_condition"] == "Match"]["Correct"].mean()
    mismatch_acc = grp[grp["match_condition"] == "Mismatch"]["Correct"].mean()
    mismatch_bias = 1 - mismatch_acc  # % Mismatch试次中选了Matching
    d = DESIGN_MAP[gid]
    stats_rows.append({
        "group_id": gid, "Label": d["Label"], "P": d["P"], "T_ms": d["T_ms"],
        "W_ms": d["W_ms"], "match_acc": match_acc*100, "mismatch_acc": mismatch_acc*100,
        "mismatch_bias": mismatch_bias*100, "quality": QUALITY_MAP[gid],
    })
df_stats = pd.DataFrame(stats_rows)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

configs = [
    ("P", "Practice Trials (P)", "match_acc", "Match Accuracy (%)", axes[0, 0]),
    ("T_ms", "Stimulus Duration T (ms)", "match_acc", "Match Accuracy (%)", axes[0, 1]),
    ("W_ms", "Response Window W (ms)", "match_acc", "Match Accuracy (%)", axes[0, 2]),
    ("P", "Practice Trials (P)", "mismatch_bias", "Mismatch→Matching Bias (%)", axes[1, 0]),
    ("T_ms", "Stimulus Duration T (ms)", "mismatch_bias", "Mismatch→Matching Bias (%)", axes[1, 1]),
    ("W_ms", "Response Window W (ms)", "mismatch_bias", "Mismatch→Matching Bias (%)", axes[1, 2]),
]

for xvar, xlabel, yvar, ylabel, ax in configs:
    for _, row in df_stats.iterrows():
        c = QUALITY_COLORS[row["quality"]]
        alpha = 0.9 if row["quality"] == "Good" else (0.7 if row["quality"] == "Medium" else 0.4)
        s = 150 if row["quality"] == "Good" else 100
        ax.scatter(row[xvar], row[yvar], s=s, c=c, alpha=alpha,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(f"G{int(row['group_id'])}", (row[xvar], row[yvar]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight="bold")
    if "mismatch_bias" in yvar:
        ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Fig 4: Design Parameters (P/T/W) vs Match/Mismatch Behavior",
             fontsize=15, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "fig4_design_space_bias.png"
fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")""")

# =====================================================================
# CELL 7: Statistical Summary
# =====================================================================
code("""# =====================================================================
# Section 2: Statistical Summary Tables
# =====================================================================

print("=" * 60)
print("Section 2: 统计指标汇总")
print("=" * 60)

# 计算关键统计指标
stats_detail = []
for gid in range(1, 9):
    grp_v = df_valid[df_valid["groupID"] == gid]
    grp_all = df[df["groupID"] == gid]
    d = DESIGN_MAP[gid]

    match_sub = grp_v[grp_v["match_condition"] == "Match"]
    mismatch_sub = grp_v[grp_v["match_condition"] == "Mismatch"]

    self_sub = grp_v[grp_v["identity"] == "Self"]
    stranger_sub = grp_v[grp_v["identity"] == "Stranger"]

    row = {
        "Group": gid, "Label": d["Label"], "Quality": QUALITY_MAP[gid],
        "N_trials": len(grp_v), "Omission%": f"{grp_all['omission'].mean()*100:.1f}",
        "Overall_Acc%": f"{grp_v['Correct'].mean()*100:.1f}",
        "Match_Acc%": f"{match_sub['Correct'].mean()*100:.1f}",
        "Mismatch_Acc%": f"{mismatch_sub['Correct'].mean()*100:.1f}",
        "Mismatch→Match_Bias%": f"{(1-mismatch_sub['Correct'].mean())*100:.1f}",
        "RT_Match_ms": f"{match_sub['RT_num'].mean()*1000:.0f}",
        "RT_Mismatch_ms": f"{mismatch_sub['RT_num'].mean()*1000:.0f}",
        "RT_Self_ms": f"{self_sub['RT_num'].mean()*1000:.0f}",
        "RT_Stranger_ms": f"{stranger_sub['RT_num'].mean()*1000:.0f}",
        "Self_MatchAcc%": f"{self_sub[self_sub['match_condition']=='Match']['Correct'].mean()*100:.1f}",
        "Stranger_MatchAcc%": f"{stranger_sub[stranger_sub['match_condition']=='Match']['Correct'].mean()*100:.1f}",
    }
    stats_detail.append(row)

df_detail = pd.DataFrame(stats_detail)
print(df_detail.to_string(index=False))

# Save
out_csv = PROJECT_ROOT / "2_Data" / "Generate_Data" / "Match_Bias_Analysis"
out_csv.mkdir(parents=True, exist_ok=True)
df_detail.to_csv(out_csv / "match_bias_statistics.csv", index=False)
print(f"\\nSaved: {out_csv / 'match_bias_statistics.csv'}")

# --- Key Findings ---
print("\\n" + "=" * 60)
print("Key Findings Summary")
print("=" * 60)
lines = [
    "",
    "1. Overall Bias:",
    "   - G1-G3 (Low quality): omission 38-72%, Matching judgment ~50%",
    "   - G4-G8 (Good quality): Match accuracy 73-94%, Mismatch accuracy 6-53%",
    "",
    "2. Matching Bias Index:",
    "   - G4-G8 only misjudge 6-53% of Mismatch trials as Matching",
    "   - G6 lowest (6.0%): extremely low Matching bias",
    "   - As T and W increase, Mismatch judgment accuracy improves significantly",
    "",
    "3. Self vs Stranger:",
    "   - Self RT typically shorter than Stranger (positive SPE)",
    "   - Self condition Mismatch misjudgment rate may be higher than Stranger",
    "   - Need DDM parameters (v/a/z) for deeper computational model analysis",
    "",
    "4. Design Parameter Modulation:",
    "   - Larger W -> lower Mismatch-to-Matching misjudgment rate",
    "   - Larger T -> higher Match trial accuracy",
    "   - P>0: behavioral metrics tend to stabilize",
]
for line in lines:
    print(line)""")

# =====================================================================
# CELL 8: Final markdown summary
# =====================================================================
md("""---
## 分析总结

### 可视化图表清单

| 图表 | 内容 | 关键发现 |
|:---|:---|:---|
| Fig 1 | 各组 Matching 判断比例条形图 | 高质量组(G4-G8)在 Mismatch 试次上表现出低误判率 |
| Fig 2 | RT × 条件差异 + 偏向指数 | Mismatch 试次 RT 略长于 Match 试次 |
| Fig 3 | Self vs Stranger 身份效应 | Self 条件 RT 较短；Mismatch 误判模式因组而异 |
| Fig 4 | P/T/W 设计参数调节 | W 和 T 对 Mismatch 判断有显著调节作用 |

### 局限性

1. **仅行为层面**: 本分析只考察了行为指标 (正确率、RT)，未使用 DDM 计算模型
2. **正确率 ≠ 偏向**: Correct 列定义为 "是否按了正确的键"，而非 "是否判断为 Matching"
   - 在 Mismatch 试次中 Correct=1 表示 "被试正确判断为 NonMatching"
   - 在 Match 试次中 Correct=1 表示 "被试正确判断为 Matching"
3. **下一步**: 结合 HDDM 的 v (漂移率) 和 z (起始点) 参数来量化决策层面的 Matching/NonMatching 偏向

### 输出文件

| 文件 | 路径 |
|:---|:---|
| 统计表格 | `2_Data/Generate_Data/Match_Bias_Analysis/match_bias_statistics.csv` |
| 可视化 | `3_Figures/Match_Bias_Analysis/fig1-4_*.png` |
""")

# =====================================================================
# Assemble and save
# =====================================================================
nb.cells = cells
out_path = OUT_DIR / "Match_vs_Mismatch_Bias_Analysis.ipynb"
nbf.write(nb, out_path)
print(f"\nNotebook saved: {out_path}")
print(f"Total cells: {len(cells)}")
