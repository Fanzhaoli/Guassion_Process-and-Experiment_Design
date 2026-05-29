import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "2_Data" / "Real_Data" / "UnExtact" / "raw"
OUT_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready_Nonmatching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 1: 数据预处理 - 准备 HDDM 输入数据 (含 NonMatching)")
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
match_mask_odd = ((df_f["Shape"] == "circle") & (df_f["Label"] == "self")) | \
                ((df_f["Shape"] == "square") & (df_f["Label"] == "stranger"))
match_mask_even = ((df_f["Shape"] == "square") & (df_f["Label"] == "self")) | \
                  ((df_f["Shape"] == "circle") & (df_f["Label"] == "stranger"))
matching_mask = np.where(odd_mask, match_mask_odd, match_mask_even)
df_f["condition"] = np.where(matching_mask, 1, 0)
n_match = df_f["condition"].sum()
n_nonmatch = len(df_f) - n_match
print(f"Matching 试次: {n_match} ({n_match/len(df_f)*100:.1f}%)")
print(f"NonMatching 试次: {n_nonmatch} ({n_nonmatch/len(df_f)*100:.1f}%)")

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

df_f["response"] = np.where(
    df_f["condition"] == 1,
    df_f["Correct"],
    1 - df_f["Correct"],
)

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
    out_path = OUT_DIR / fn
    hddm_df.to_csv(out_path, index=False)

    n_subj = gdf["subjectID"].nunique()
    n_trials = len(hddm_df)
    n_omissions = hddm_df["omission"].sum()
    omission_rate = n_omissions / n_trials * 100

    n_match_trials = (hddm_df["condition"] == 1).sum()
    n_nonmatch_trials = (hddm_df["condition"] == 0).sum()

    n_valid = (hddm_df["omission"] == 0).sum()
    acc_all = hddm_df.loc[hddm_df["omission"] == 0, "response"].mean() if n_valid > 0 else np.nan
    acc_match = hddm_df.loc[(hddm_df["omission"] == 0) & (hddm_df["condition"] == 1), "response"].mean()
    acc_nonmatch = hddm_df.loc[(hddm_df["omission"] == 0) & (hddm_df["condition"] == 0), "response"].mean()

    resp_matching_pct = hddm_df.loc[hddm_df["omission"] == 0, "response"].mean() * 100 if n_valid > 0 else np.nan

    print(f"\n  Group {gid} (P={P_val}, T={T_val}ms, W={W_val}ms) -> {fn}")
    print(f"    被试: {n_subj} | 试次: {n_trials}")
    print(f"    Matching: {n_match_trials}, NonMatching: {n_nonmatch_trials}")
    print(f"    有效试次: {n_valid} | 遗漏试次: {n_omissions} ({omission_rate:.1f}%)")
    print(f"    判断为Matching比例: {resp_matching_pct:.1f}%")
    print(f"    Matching试次正确率: {acc_match:.3f}  NonMatching试次正确率: {acc_nonmatch:.3f}")

print(f"\nHDDM 就绪数据已保存到: {OUT_DIR}")
print("=" * 60)
