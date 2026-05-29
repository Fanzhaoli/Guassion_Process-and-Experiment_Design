import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design")
HDDM_READY_DIR = PROJECT_ROOT / "2_Data" / "Real_Data" / "HDDM_Ready"
OUTPUT_DIR = PROJECT_ROOT / "2_Data" / "Generate_Data" / "CRF_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESIGN_MAP = {
    1: {"P": 0, "T_ms": 30, "W_ms": 300, "Label": "G1 | P0_T30_W300"},
    2: {"P": 0, "T_ms": 30, "W_ms": 600, "Label": "G2 | P0_T30_W600"},
    3: {"P": 120, "T_ms": 30, "W_ms": 600, "Label": "G3 | P120_T30_W600"},
    4: {"P": 120, "T_ms": 80, "W_ms": 600, "Label": "G4 | P120_T80_W600"},
    5: {"P": 8, "T_ms": 100, "W_ms": 1100, "Label": "G5 | P8_T100_W1100"},
    6: {"P": 120, "T_ms": 500, "W_ms": 1500, "Label": "G6 | P120_T500_W1500"},
    7: {"P": 120, "T_ms": 80, "W_ms": 800, "Label": "G7 | P120_T80_W800"},
    8: {"P": 120, "T_ms": 80, "W_ms": 800, "Label": "G8 | P120_T80_W800"},
}

QUALITY_MAP = {
    1: "exclude",  # 72.3% omission
    2: "exclude",  # 52.4% omission
    3: "caution",  # 38.6% omission, v negative
    4: "good",
    5: "good",
    6: "good",
    7: "good",
    8: "good",
}

all_frames = []
for group_id in range(1, 9):
    fpath = HDDM_READY_DIR / f"hddm_data_group{group_id}_P{DESIGN_MAP[group_id]['P']}_T{DESIGN_MAP[group_id]['T_ms']}_W{DESIGN_MAP[group_id]['W_ms']}.csv"
    if not fpath.exists():
        print(f"WARNING: {fpath} not found, trying glob fallback...")
        candidates = list(HDDM_READY_DIR.glob(f"hddm_data_group{group_id}_*.csv"))
        if candidates:
            fpath = candidates[0]
            print(f"  Using fallback: {fpath}")
        else:
            print(f"  SKIP group {group_id}: no file found")
            continue

    df = pd.read_csv(fpath)
    df["group_id"] = group_id
    df["P"] = DESIGN_MAP[group_id]["P"]
    df["T_ms"] = DESIGN_MAP[group_id]["T_ms"]
    df["W_ms"] = DESIGN_MAP[group_id]["W_ms"]
    df["M_ms"] = DESIGN_MAP[group_id]["T_ms"] + DESIGN_MAP[group_id]["W_ms"]
    df["GroupLabel"] = DESIGN_MAP[group_id]["Label"]
    df["data_quality"] = QUALITY_MAP[group_id]
    all_frames.append(df)
    print(f"Group {group_id}: {len(df)} trials, {df['subj_idx'].nunique()} subjects")

combined = pd.concat(all_frames, ignore_index=True)
combined["Identity"] = combined["identity"].map({0: "Self", 1: "Stranger"})
combined["Response"] = combined["response"].map({1: "Matching", 0: "NonMatching"})

non_omission = combined[combined["omission"] == 0].copy()
print(f"\nTotal trials: {len(combined)}")
print(f"Non-omission trials: {len(non_omission)} ({len(non_omission)/len(combined)*100:.1f}%)")
print(f"Total subjects: {combined['subj_idx'].nunique()}")

summary = (
    combined.groupby("group_id")
    .agg(
        n_trials=("rt", "count"),
        n_subjects=("subj_idx", "nunique"),
        n_omission=("omission", "sum"),
        omission_rate=("omission", "mean"),
        acc_mean=("response", "mean"),
        rt_mean=("rt", "mean"),
        rt_sd=("rt", "std"),
    )
    .reset_index()
)
for _, row in summary.iterrows():
    print(f"  G{int(row['group_id'])}: {int(row['n_trials'])} trials, "
          f"omission={row['omission_rate']*100:.1f}%, "
          f"ACC={row['acc_mean']*100:.1f}%, "
          f"RT={row['rt_mean']*1000:.0f}±{row['rt_sd']*1000:.0f}ms")

out_path = OUTPUT_DIR / "trial_level_combined.csv"
combined.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

summary_out = OUTPUT_DIR / "data_quality_summary.csv"
summary.to_csv(summary_out, index=False)
print(f"Saved: {summary_out}")
