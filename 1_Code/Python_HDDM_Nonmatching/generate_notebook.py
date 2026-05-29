r"""Generate Docker_Run_Nonmatching.ipynb from cell snippets for Python_HDDM_Nonmatching."""
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_HDDM_Nonmatching")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.13"}
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ============================================================
# CELL 0: Header
# ============================================================
md("""# Docker HDDM 参数拟合工作流 — NonMatching 版本

**项目**: GP-SPE 实验设计优化 — Self-Matching Task DDM 参数提取（含 NonMatching 试次）

**与原版关键差异**:
- 纳入**全部试次**（Matching + NonMatching），各占 50%
- response 统一编码为 **"判断为 Matching(1) vs 判断为 NonMatching(0)"**
- 上边界 = 被试判断为 Matching；下边界 = 被试判断为 NonMatching
- y=0.5 参考线具有直接的行为解释力（Matching/NonMatching 判断无偏好）

---

## Docker 启动方式

在 CMD/PowerShell 中执行以下命令启动 Docker Jupyter：

```bash
docker pull hcp4715/hddm

docker run -it --rm --cpus=4 ^
  -v /d/GitHub_programe/GitHub/Guassion-Process-Experiment-Design:/home/jovyan/work ^
  -p 8888:8888 ^
  hcp4715/hddm ^
  jupyter notebook
```

> **注意**: 挂载路径是**项目根目录**，这样数据直接写入正确的 `2_Data/` 和 `3_Figures/` 位置。

---

## 工作流步骤

| 步骤 | 环境 | 内容 |
|------|------|------|
| **Step 1** | **Docker（本文件）** | 数据预处理 → HDDM 就绪 CSV（含 Matching+NonMatching） |
| **Step 2** | **Docker（本文件）** | HDDM 层级模型 MCMC 拟合 |
| Step 3 | Docker（本文件） | 迹线提取 + 参数可视化 |
""")

# ============================================================
# CELL 1: Environment Check
# ============================================================
md("""---
## 环境检查""")

code("""import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/jovyan/work")
DATA_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready_Nonmatching"
OUT_DIR  = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces_Nonmatching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR:  {BASE_DIR}  存在: {BASE_DIR.exists()}")
print(f"DATA_DIR:  {DATA_DIR}  存在: {DATA_DIR.exists()}")
print(f"OUT_DIR:   {OUT_DIR}")

try:
    import hddm
    print(f"HDDM 版本: {hddm.__version__}")
except ImportError:
    print("HDDM 未安装！请在 Docker 容器 (hcp4715/hddm) 中运行此笔记本。")
    sys.exit(1)

# 检查 HDDM Ready 数据（如果已运行过 step1，直接显示）
csv_files = sorted(DATA_DIR.glob("hddm_data_group*.csv"))
if csv_files:
    print(f"发现 {len(csv_files)} 个 HDDM 就绪数据文件（跳过 Step 1 可直接进入 Step 2）")
    for f in csv_files:
        df = pd.read_csv(f)
        n_subj = df["subj_idx"].nunique()
        n_omis = df["omission"].sum()
        n_match = (df["condition"] == 1).sum() if "condition" in df.columns else "?"
        n_nonmatch = (df["condition"] == 0).sum() if "condition" in df.columns else "?"
        print(f"   {f.name}: {n_subj}被试, {len(df)}试次, M={n_match}, NM={n_nonmatch}, 遗漏{n_omis}")
else:
    print("未找到 HDDM 就绪数据！请先运行 Step 1。")
""")

# ============================================================
# CELL 2: Step 1 Markdown
# ============================================================
md("""---
## Step 1: 数据预处理（含 Matching + NonMatching）

从 88 个原始 CSV 中提取 **全部 formal 试次**（Matching + NonMatching），
统一 response 编码为 "判断为 Matching(1) vs 判断为 NonMatching(0)"。

> 如果 `2_Data/Real_Data/HDDM_Ready_Nonmatching/` 下已有数据，可跳过此步骤。""")

# ============================================================
# CELL 3: Step 1 Code
# ============================================================
code(r"""# Step 1: 数据预处理 - 准备 HDDM 输入数据（含 NonMatching）
# 提取 Matching + NonMatching 试次，统一 response 编码

RAW_DIR = BASE_DIR / "2_Data" / "Real_Data" / "UnExtact" / "raw"
READY_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready_Nonmatching"
READY_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 1: 数据预处理（Matching + NonMatching）")
print("=" * 60)

# 调试: 列出 RAW_DIR 的内容，排查路径问题
print(f"\nRAW_DIR:    {RAW_DIR}")
print(f"RAW_DIR 存在: {RAW_DIR.exists()}")
if RAW_DIR.exists():
    all_items = list(RAW_DIR.iterdir())
    print(f"RAW_DIR 内容 ({len(all_items)} 项):")
    for item in sorted(all_items)[:10]:
        print(f"  {item.name}")
    if len(all_items) > 10:
        print(f"  ... 还有 {len(all_items)-10} 个文件")

csv_files = sorted(RAW_DIR.glob("EXP_data_group*.csv"))
print(f"\n匹配到的 CSV 文件: {len(csv_files)} 个")

if len(csv_files) == 0:
    print("错误: 未找到原始数据文件！")
    print("可能原因:")
    print("  1. Docker 挂载路径不正确（-v 参数中的路径）")
    print("  2. 原始数据尚未解压（检查 UnExtact/data.zip）")
    print("  3. 文件命名不符合 EXP_data_group*.csv 模式")
    print("\n解决方案: 在 Docker 外部本机运行 step1_prepare_data.py 后跳过此步骤")
    csv_files = []  # 确保空列表

if csv_files:
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"合并总行数: {len(df_all)}")

    # 过滤 formal 阶段
    df_f = df_all[df_all["stage"] == "formal"].copy()
    print(f"过滤 stage='formal' 后: {len(df_f)} 行")

    # 标记 Matching 条件
    # 标记 Matching 条件（基于 subjectID 奇偶性翻转）
odd_mask = (df_f["subjectID"] % 2 == 1)
match_mask_odd = ((df_f["Shape"] == "circle") & (df_f["Label"] == "self")) | \
                ((df_f["Shape"] == "square") & (df_f["Label"] == "stranger"))
match_mask_even = ((df_f["Shape"] == "square") & (df_f["Label"] == "self")) | \
                  ((df_f["Shape"] == "circle") & (df_f["Label"] == "stranger"))
matching_mask = np.where(odd_mask, match_mask_odd, match_mask_even)
df_f["condition"] = np.where(matching_mask, 1, 0)
    n_match = df_f["condition"].sum()
    n_nonmatch = len(df_f) - n_match
    print(f"Matching: {n_match} ({n_match/len(df_f)*100:.1f}%)")
    print(f"NonMatching: {n_nonmatch} ({n_nonmatch/len(df_f)*100:.1f}%)")

    # 构建 HDDM 所需列
    df_f["identity"] = df_f["Label"].map({"self": 1, "stranger": 0})
    df_f["RT_num"] = pd.to_numeric(df_f["RT"], errors="coerce")
    df_f["omission"] = df_f["RT_num"].isna().astype(int)
    df_f["T_s"] = pd.to_numeric(df_f["T"], errors="coerce")
    df_f["W_s"] = pd.to_numeric(df_f["W"], errors="coerce")
    df_f["deadline"] = df_f["T_s"] + df_f["W_s"]

    df_f["rt"] = np.where(
        df_f["omission"] == 1,
        df_f["deadline"],
        df_f["RT_num"],
    )

    # 核心: 统一 response 编码
    # response=1 表示 "被试判断为 Matching"（上边界）
    # response=0 表示 "被试判断为 NonMatching"（下边界）
    df_f["response"] = np.where(
        df_f["condition"] == 1,
        df_f["Correct"],       # Matching试次: 正确=判为Matching
        1 - df_f["Correct"],   # NonMatching试次: 错误=判为Matching
    )

    print("\n各组 HDDM 就绪数据（含 Matching + NonMatching）:")
    for gid in sorted(df_f["groupID"].unique()):
        gdf = df_f[df_f["groupID"] == gid].copy()
        subj_map = {s: i for i, s in enumerate(sorted(gdf["subjectID"].unique()))}
        gdf["subj_idx"] = gdf["subjectID"].map(subj_map)

        hddm_cols = ["subj_idx", "rt", "response", "identity", "condition", "omission"]
        hddm_df = gdf[hddm_cols].copy()

        P_val = int(gdf["P"].iloc[0])
        T_val = int(gdf["T_s"].iloc[0] * 1000)
        W_val = int(gdf["W_s"].iloc[0] * 1000)

        fn = f"hddm_data_group{gid}_P{P_val}_T{T_val}_W{W_val}.csv"
        out_path = READY_DIR / fn
        hddm_df.to_csv(out_path, index=False)

        n_subj = gdf["subjectID"].nunique()
        n_trials = len(hddm_df)
        n_omissions = hddm_df["omission"].sum()
        omission_rate = n_omissions / n_trials * 100
        n_match_trials = (hddm_df["condition"] == 1).sum()
        n_nonmatch_trials = (hddm_df["condition"] == 0).sum()
        n_valid = (hddm_df["omission"] == 0).sum()
        resp_match_pct = hddm_df.loc[hddm_df["omission"] == 0, "response"].mean() * 100 if n_valid > 0 else np.nan

        print(f"  Group {gid} (P={P_val}, T={T_val}ms, W={W_val}ms) -> {fn}")
        print(f"    被试: {n_subj} | 试次: {n_trials}  (M:{n_match_trials}, NM:{n_nonmatch_trials})")
        print(f"    有效: {n_valid} | 遗漏: {n_omissions} ({omission_rate:.1f}%)")
        print(f"    判断为Matching: {resp_match_pct:.1f}%")

    print(f"\nHDDM 就绪数据已保存到: {READY_DIR}")

    # 更新 csv_files 供后续 Step 2 使用
    csv_files = sorted(READY_DIR.glob("hddm_data_group*.csv"))
    print(f"\n准备拟合 {len(csv_files)} 个文件")
else:
    print("\n跳过数据预处理（未找到原始文件）。")
    print("如果 HDDM_Ready_Nonmatching 下已有数据，可直接进入 Step 2。")
    csv_files = sorted(READY_DIR.glob("hddm_data_group*.csv"))
    if csv_files:
        print(f"发现 {len(csv_files)} 个已有 HDDM 就绪数据文件，可直接进入 Step 2。")
""")

# ============================================================
# CELL 4: Step 2 Markdown
# ============================================================
md("""---
## Step 2: HDDM 层级模型拟合

对每组实验条件独立拟合层级 DDM，使用 `depends_on={"v": "identity"}` 区分 Self/Stranger 漂移率。

**模型设定**:
```python
model = hddm.HDDM(
    df,
    depends_on={"v": "identity"},
    include=["v", "a", "t", "z"],
    bias=False,
    p_outlier=0.05,
)
```

- v 依 identity 变化 → 得到 v(0)=v_stranger, v(1)=v_self
- z 不依 identity 变化 → Self/Stranger 共享起始点（与原版一致，保持可比性）

**预估时间**: 每组约 3-6 分钟（试次数比原版翻倍），全部 8 组约 25-60 分钟。""")

# ============================================================
# CELL 5: Step 2 Code
# ============================================================
code("""# Step 2: HDDM 层级模型 MCMC 拟合
# 此单元格在 Docker 容器中运行

import pickle

for csv_path in csv_files:
    fname = csv_path.stem
    print(f"\\n{'=' * 50}")
    print(f"拟合: {fname}")
    print(f"{'=' * 50}")

    df = pd.read_csv(csv_path)
    n_subj = df["subj_idx"].nunique()
    n_self = (df["identity"] == 1).sum()
    n_stranger = (df["identity"] == 0).sum()
    n_omission = df["omission"].sum()

    print(f"  被试: {n_subj}, 试次: {len(df)}")
    print(f"  Self: {n_self}, Stranger: {n_stranger}")
    print(f"  遗漏: {n_omission} ({n_omission/len(df)*100:.1f}%)")

    if "condition" in df.columns:
        n_match = (df["condition"] == 1).sum()
        n_nonmatch = (df["condition"] == 0).sum()
        print(f"  Matching: {n_match}, NonMatching: {n_nonmatch}")

    model = hddm.HDDM(
        df,
        depends_on={"v": "identity"},
        include=["v", "a", "t", "z"],
        bias=False,
        p_outlier=0.05,
    )

    print("  开始 MCMC 采样 (3000 draws, 500 burn)...")
    db_name = f"traces_{fname}.db"
    model.sample(3000, burn=500, dbname=db_name, db="pickle")
    print("  采样完成")

    stats = model.gen_stats()
    stats_path = OUT_DIR / f"{fname}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  统计 -> {stats_path}")

    try:
        traces_raw = model.get_traces()
        traces_simple = {}
        for key, val in traces_raw.items():
            try:
                arr = np.asarray(val, dtype=float).flatten()
                if len(arr) > 0:
                    traces_simple[key] = arr
            except Exception:
                continue

        if traces_simple:
            trace_path = OUT_DIR / f"{fname}_traces.pkl"
            with open(trace_path, "wb") as f:
                pickle.dump(traces_simple, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  迹线 (pickle) -> {trace_path}")
            print(f"  参数键名: {list(traces_simple.keys())}")

            npz_path = OUT_DIR / f"{fname}_traces.npz"
            np.savez_compressed(npz_path, **traces_simple)
            print(f"  迹线 (npz) -> {npz_path}")
        else:
            print("  get_traces() 返回空，仅保存了统计文件")
    except Exception as e:
        print(f"  迹线保存失败: {e}")
        print(f"  统计文件仍然可用，Step 3 可以从 stats.csv 提取参数")

    n_valid = (df["omission"] == 0).sum()
    resp_match_pct = df.loc[df["omission"] == 0, "response"].mean() if n_valid > 0 else np.nan
    self_rt = df[(df["identity"]==1) & (df["omission"]==0)]["rt"].mean()
    stranger_rt = df[(df["identity"]==0) & (df["omission"]==0)]["rt"].mean()
    print(f"  有效试次: {n_valid}, 判断Matching比例: {resp_match_pct:.3f}")
    print(f"  Self RT: {self_rt:.4f}s, Stranger RT: {stranger_rt:.4f}s")

    import os
    try:
        os.remove(db_name)
    except Exception:
        pass

print(f"\\n{'=' * 50}")
print("所有拟合完成!")
print(f"结果保存在: {OUT_DIR}")
""")

# ============================================================
# CELL 6: Step 3 Markdown
# ============================================================
md("""---
## Step 3: 参数提取与可视化

拟合完成后，运行此单元格提取 DDM 参数并绘制图表。
此步骤**也可以在 Docker 外部的本机运行** `step3_extract_params.py`。""")

# ============================================================
# CELL 7: Step 3 Code
# ============================================================
code(r"""# Step 3: 参数提取与可视化
import matplotlib.pyplot as plt
import re

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = BASE_DIR / "3_Figures" / "HDDM_Results_Nonmatching"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 3: 提取 HDDM 参数后验分布 (NonMatching 版本)")
print("=" * 60)

stats_files = sorted(OUT_DIR.glob("*_stats.csv"))
if not stats_files:
    print("未找到 stats 文件！请先运行 Step 2。")
else:
    all_params = []
    for stats_path in stats_files:
        fname = stats_path.stem.replace("_stats", "")
        print(f"\n处理: {fname}")

        match = re.search(r"group(\d+)_P(\d+)_T(\d+)_W(\d+)", fname)
        if not match:
            continue
        group_id = int(match.group(1))
        P_val = int(match.group(2))
        T_val = int(match.group(3))
        W_val = int(match.group(4))

        stats = pd.read_csv(stats_path, index_col=0)
        row = {
            "group_id": group_id, "P": P_val, "T_ms": T_val,
            "W_ms": W_val, "M_ms": T_val + W_val,
        }

        for stat_key, label in [("v(0)", "v_stranger"), ("v(1)", "v_self"),
                                 ("a", "a"), ("t", "t"), ("z", "z")]:
            if stat_key in stats.index:
                row[f"{label}_mean"] = float(stats.loc[stat_key, "mean"])
                row[f"{label}_std"] = float(stats.loc[stat_key, "std"])
                row[f"{label}_q025"] = float(stats.loc[stat_key, "2.5q"])
                row[f"{label}_q975"] = float(stats.loc[stat_key, "97.5q"])

        if "v_self_mean" in row and "v_stranger_mean" in row:
            row["SPE_v"] = row["v_self_mean"] - row["v_stranger_mean"]
            print(f"  v_self={row['v_self_mean']:.3f}, v_stranger={row['v_stranger_mean']:.3f}")
            print(f"  SPE_v={row['SPE_v']:.3f}, a={row.get('a_mean',np.nan):.3f}, z={row.get('z_mean',np.nan):.3f}")

        all_params.append(row)

    df_params = pd.DataFrame(all_params).sort_values("group_id")
    df_params.to_csv(OUT_DIR / "all_groups_ddm_params.csv", index=False)

    disp_cols = ["group_id","P","T_ms","W_ms","v_self_mean","v_stranger_mean",
                 "SPE_v","a_mean","t_mean","z_mean"]
    avail = [c for c in disp_cols if c in df_params.columns]
    print("\n" + "=" * 60)
    print("DDM 参数汇总 (All Trials — Matching + NonMatching)")
    print("=" * 60)
    print(df_params[avail].round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if "SPE_v" in df_params.columns:
        valid = df_params.dropna(subset=["SPE_v"])
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid)))
        for i, (_, r) in enumerate(valid.iterrows()):
            lo = r["v_self_mean"] - r["v_self_q025"]
            hi = r["v_self_q975"] - r["v_self_mean"]
            ax.errorbar(i, r["SPE_v"], yerr=[[lo], [hi]], fmt="o",
                        color=colors[i], capsize=5, markersize=8)
        ax.axhline(y=0, color="gray", linestyle="--")
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
        ax.set_ylabel("SPE_v"); ax.set_title("SPE in Drift Rate (All Trials)")

    ax = axes[0, 1]
    if "v_self_mean" in df_params.columns:
        valid = df_params.dropna(subset=["v_self_mean"])
        x = np.arange(len(valid)); w = 0.35
        ax.bar(x-w/2, valid["v_self_mean"], w, label="Self", alpha=0.8)
        ax.bar(x+w/2, valid["v_stranger_mean"], w, label="Stranger", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"G{r['group_id']:.0f}" for _, r in valid.iterrows()], rotation=45)
        ax.set_ylabel("Drift Rate v"); ax.set_title("Drift Rate (All Trials)"); ax.legend()

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
        ax.set_ylabel("Boundary a"); ax.set_title("Decision Boundary (All Trials)")

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
        ax.set_ylabel("t (s)"); ax.set_title("Nondecision Time (All Trials)")

    plt.tight_layout()
    fig_path = FIG_DIR / "ddm_params_by_group.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\n图表已保存到: {fig_path}")
    print(f"参数表已保存到: {OUT_DIR / 'all_groups_ddm_params.csv'}")
""")

# ============================================================
# CELL 8: Output List
# ============================================================
md("""---
## 输出文件清单

| 文件 | 位置 | 内容 |
|------|------|------|
| `hddm_data_group*.csv` | `2_Data/Real_Data/HDDM_Ready_Nonmatching/` | Step 1 预处理后的 HDDM 输入（含 Matching+NonMatching） |
| `*_stats.csv` | `2_Data/Real_Data/HDDM_Traces_Nonmatching/` | Step 2 各参数后验摘要 (mean, std, 分位数) |
| `*_traces.pkl` | `2_Data/Real_Data/HDDM_Traces_Nonmatching/` | Step 2 完整 MCMC 采样迹线 (pickle) |
| `*_traces.npz` | `2_Data/Real_Data/HDDM_Traces_Nonmatching/` | Step 2 完整 MCMC 采样迹线 (numpy) |
| `all_groups_ddm_params.csv` | `2_Data/Real_Data/HDDM_Traces_Nonmatching/` | Step 3 所有条件汇总 |
| `ddm_params_by_group.png` | `3_Figures/HDDM_Results_Nonmatching/` | Step 3 参数柱状图 |

---

## Response 编码说明

```
核心原则: response=1 统一表示"被试判断为 Matching"（上边界）
         response=0 统一表示"被试判断为 NonMatching"（下边界）

Matching 试次 (condition=1):
  Correct=1 → 判断正确 → 判断为Matching → response=1
  Correct=0 → 判断错误 → 判断为NonMatching → response=0

NonMatching 试次 (condition=0):
  Correct=1 → 判断正确 → 判断为NonMatching → response=0
  Correct=0 → 判断错误 → 判断为Matching → response=1
```

---

## 注意事项

1. **v 的正负号**: v>0 表示漂移指向上边界（被试倾向于判断为 Matching），v<0 指向下边界
2. **z 参数**: 与原版相同，z 不依赖 identity，Self/Stranger 共享起始点
3. **y=0.5 参考线**: 因 Matching/NonMatching 各占 50%，y=0.5 代表判断无偏好（vs 原版中代表随机正确率）
4. **遗漏试次处理**: RT 设为 T+W（截止时间），作为截尾数据参与拟合，p_outlier=0.05 自动识别
5. **Group 1/2 高遗漏率**: 拟合可能不收敛，检查 _stats.csv 的 R-hat
""")

# ============================================================
# Assemble and save
# ============================================================
nb.cells = cells
out_path = OUT_DIR / "Docker_Run_Nonmatching.ipynb"
nbf.write(nb, out_path)
print(f"\nNotebook saved: {out_path}")
print(f"Cells: {len(cells)} (4 markdown + 4 code)")
