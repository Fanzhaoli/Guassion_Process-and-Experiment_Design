import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
HDDM_READY_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready"
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_DIR = OUT_DIR / "drop_omission"
CENSOR_DIR = OUT_DIR / "censored"
DROP_DIR.mkdir(parents=True, exist_ok=True)
CENSOR_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Omission 敏感性分析 - 数据准备")
print("=" * 60)

csv_files = sorted(HDDM_READY_DIR.glob("hddm_data_group*.csv"))

summary_rows = []

for csv_path in csv_files:
    fname = csv_path.stem
    print(f"\n处理: {fname}")

    df = pd.read_csv(csv_path)

    n_total = len(df)
    n_omission = df["omission"].sum()
    n_valid = n_total - n_omission
    omission_rate = n_omission / n_total * 100

    match = re.search(r"group(\d+)_P(\d+)_T(\d+)_W(\d+)", fname)
    if match:
        group_id = int(match.group(1))
        P_val = int(match.group(2))
        T_val = int(match.group(3))
        W_val = int(match.group(4))
    else:
        print(f"  ⚠️ 无法解析文件名")
        continue

    df_drop = df[df["omission"] == 0].copy()
    n_drop = len(df_drop)
    df_drop = df_drop.drop(columns=["omission"])

    drop_path = DROP_DIR / f"{fname}_drop.csv"
    df_drop.to_csv(drop_path, index=False)

    df_censor = df.copy()
    censor_path = CENSOR_DIR / f"{fname}_censor.csv"
    df_censor.to_csv(censor_path, index=False)

    n_subj = df["subj_idx"].nunique()
    d_acc = df_drop["response"].mean() if n_drop > 0 else np.nan

    print(f"  Group {group_id}: P={P_val}, T={T_val}ms, W={W_val}ms")
    print(f"    被试: {n_subj} | 总试次: {n_total} | 遗漏: {n_omission} ({omission_rate:.1f}%)")
    print(f"    Drop 方案: {n_drop} 试次 (仅有效试次), 正确率={d_acc:.3f}")
    print(f"    Censor 方案: {n_total} 试次 (遗漏 rt=deadline, response=0)")

    summary_rows.append({
        "group_id": group_id,
        "P": P_val,
        "T_ms": T_val,
        "W_ms": W_val,
        "M_ms": T_val + W_val,
        "n_subjects": n_subj,
        "n_total_trials": n_total,
        "n_omission": int(n_omission),
        "omission_rate": omission_rate,
        "n_drop_trials": n_drop,
        "n_censor_trials": n_total,
        "drop_acc": d_acc,
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "data_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n{'=' * 60}")
print("数据准备完成!")
print(f"  Drop 数据 → {DROP_DIR}")
print(f"  Censor 数据 → {CENSOR_DIR}")
print(f"  汇总表 → {summary_path}")
print("=" * 60)
print("\n下一步: 在 Docker 容器中运行 run_drop_fit.py 进行 HDDM 拟合")