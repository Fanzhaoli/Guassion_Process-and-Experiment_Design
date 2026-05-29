import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "2_Data" / "Real_Data" / "UnExtact" / "raw"
OUT_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 1: 数据预处理 - 准备 HDDM 输入数据")
print("=" * 60)

csv_files = sorted(RAW_DIR.glob("EXP_data_group*.csv"))
print(f"\n发现 {len(csv_files)} 个原始数据文件")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"合并总行数: {len(df_all)}")

print(f"\n各组被试分布:")
group_subjects = df_all.groupby("groupID")["subjectID"].nunique()
for gid in sorted(group_subjects.index):
    print(f"  Group {gid}: {group_subjects[gid]} 名被试")

df_f = df_all[df_all["stage"] == "formal"].copy()
print(f"\n过滤 stage='formal' 后: {len(df_f)} 行")

# 标记 Matching 条件（基于 subjectID 奇偶性翻转）
odd_mask = (df_f["subjectID"] % 2 == 1)
mask_odd = ((df_f["Shape"] == "circle") & (df_f["Label"] == "self")) | \
           ((df_f["Shape"] == "square") & (df_f["Label"] == "stranger"))
mask_even = ((df_f["Shape"] == "square") & (df_f["Label"] == "self")) | \
            ((df_f["Shape"] == "circle") & (df_f["Label"] == "stranger"))
matching_mask = np.where(odd_mask, mask_odd, mask_even)
df_match = df_f[matching_mask].copy()
print(f"过滤 Matching 试次后: {len(df_match)} 行")

# 再过滤 NonMatching 试次（供 Python_HDDM_Nonmatching 参考使用）
nonmatching_mask = np.where(odd_mask, mask_even, mask_odd)
df_nonmatch = df_f[nonmatching_mask].copy()
print(f"NonMatching 试次: {len(df_nonmatch)} 行")

df_match["identity"] = df_match["Label"].map({"self": 1, "stranger": 0})
df_match["RT_num"] = pd.to_numeric(df_match["RT"], errors="coerce")
df_match["omission"] = df_match["RT_num"].isna().astype(int)
df_match["T_s"] = pd.to_numeric(df_match["T"], errors="coerce")
df_match["W_s"] = pd.to_numeric(df_match["W"], errors="coerce")
df_match["deadline"] = df_match["T_s"] + df_match["W_s"]

df_match["rt"] = np.where(
    df_match["omission"] == 1,
    df_match["deadline"],
    df_match["RT_num"],
)

df_match["response"] = np.where(df_match["Correct"] == 1, 1, 0)

for gid in sorted(df_match["groupID"].unique()):
    gdf = df_match[df_match["groupID"] == gid].copy()

    subj_map = {s: i for i, s in enumerate(sorted(gdf["subjectID"].unique()))}
    gdf["subj_idx"] = gdf["subjectID"].map(subj_map)

    hddm_cols = ["subj_idx", "rt", "response", "identity", "omission"]
    hddm_df = gdf[hddm_cols].copy()

    P_val = int(gdf["P"].iloc[0])
    T_val = int(gdf["T_s"].iloc[0] * 1000)
    W_val = int(gdf["W_s"].iloc[0] * 1000)

    fn = f"hddm_data_group{gid}_P{P_val}_T{T_val}_W{W_val}.csv"
    out_path = OUT_DIR / fn
    hddm_df.to_csv(out_path, index=False)

    n_subj = gdf["subjectID"].nunique()
    n_trials = len(hddm_df)
    n_omissions = hddm_df["omission"].sum()
    omission_rate = n_omissions / n_trials * 100
    n_valid = (hddm_df["omission"] == 0).sum()
    acc = hddm_df.loc[hddm_df["omission"] == 0, "response"].mean()

    print(f"\n  Group {gid} (P={P_val}, T={T_val}ms, W={W_val}ms) -> {fn}")
    print(f"    被试: {n_subj} | 试次: {n_trials}")
    print(f"    有效试次: {n_valid} | 遗漏试次: {n_omissions} ({omission_rate:.1f}%)")
    print(f"    有效试次正确率: {acc:.3f}")

print(f"\nHDDM 就绪数据已保存到: {OUT_DIR}")
print("=" * 60)
