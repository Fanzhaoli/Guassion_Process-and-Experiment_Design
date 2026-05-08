"""整合 Self-Matching Task 原始 CSV，生成正式分析数据。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "2_Data" / "Real_Data" / "UnExtact" / "raw"
OUT_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined.csv"
SUMMARY_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined_summary.csv"

REQUIRED_COLUMNS = [
    "groupID",
    "subjectID",
    "gender",
    "age",
    "handedness",
    "stage",
    "trialID",
    "P",
    "T",
    "W",
    "Shape",
    "Label",
    "CorrectKey",
    "Response",
    "RT",
    "Correct",
]


def parse_group_info(path: Path) -> tuple[int | None, int | None, str]:
    """从 EXP_data_groupX_Y.csv 文件名恢复组别与组内被试编号。"""
    match = re.search(r"EXP_data_group(\d+)_(\d+)\.csv$", path.name)
    if not match:
        return None, None, path.stem
    group_id = int(match.group(1))
    subject_id = int(match.group(2))
    return group_id, subject_id, f"group{group_id}_{subject_id}"


def design_cell(row: pd.Series) -> str:
    """用 P/T/W 定义实验设计单元，避免只依赖 groupID。"""
    return f"P{int(row['P'])}_T{int(round(row['T_ms']))}_W{int(round(row['W_ms']))}"


def load_raw_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("EXP_data_group*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到原始 CSV: {raw_dir}")

    frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, na_values=["NA", "NaN", "", " "])
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{file_path.name} 缺少必要列: {missing_cols}")

        file_group, file_subject, group_info = parse_group_info(file_path)
        df["SourceFile"] = file_path.name
        df["FileGroupID"] = file_group
        df["FileSubjectID"] = file_subject
        df["GroupInfo"] = group_info
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def preprocess(df_raw: pd.DataFrame, keep_stage: str = "formal") -> pd.DataFrame:
    df = df_raw.copy()

    # 只保留正式实验试次；练习试次不进入 SPE/ANOVA 主分析。
    df["stage"] = df["stage"].astype(str).str.strip().str.lower()
    if keep_stage:
        df = df[df["stage"] == keep_stage].copy()

    numeric_cols = ["groupID", "subjectID", "gender", "age", "handedness", "trialID", "P", "T", "W", "RT", "Correct"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Shape"] = df["Shape"].astype(str).str.strip().str.lower()
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["CorrectKey"] = df["CorrectKey"].astype(str).str.strip().str.lower()
    df["Response"] = df["Response"].astype("string").str.strip().str.lower()
    df.loc[df["Response"].isin(["", "na", "nan", "<na>"]), "Response"] = pd.NA

    df["SubjectUID"] = df["GroupInfo"].astype(str)
    df["T_ms"] = df["T"] * 1000.0
    df["W_ms"] = df["W"] * 1000.0
    df["RT_sec"] = df["RT"]
    df["RT_ms"] = df["RT_sec"] * 1000.0

    # SMT 范式定义：circle+self 与 square+stranger 为 Matching。
    matching_mask = ((df["Shape"] == "circle") & (df["Label"] == "self")) | (
        (df["Shape"] == "square") & (df["Label"] == "stranger")
    )
    df["Matching"] = np.where(matching_mask, "Matching", "NonMatching")
    df["Identity"] = df["Label"].map({"self": "Self", "stranger": "Stranger"})
    df["ACC"] = df["Correct"].fillna(0).astype(int)
    df["responded"] = df["RT_sec"].notna() & (df["RT_sec"] > 0)
    df["omission"] = (~df["responded"]).astype(int)

    df["DesignCell"] = df.apply(design_cell, axis=1)
    df["GroupLabel"] = (
        "G"
        + df["groupID"].astype(int).astype(str)
        + " | "
        + df["DesignCell"].astype(str)
    )

    ordered_cols = REQUIRED_COLUMNS + [
        "SourceFile",
        "FileGroupID",
        "FileSubjectID",
        "GroupInfo",
        "SubjectUID",
        "T_ms",
        "W_ms",
        "RT_sec",
        "RT_ms",
        "Matching",
        "Identity",
        "ACC",
        "responded",
        "omission",
        "DesignCell",
        "GroupLabel",
    ]
    return df[ordered_cols].sort_values(["groupID", "subjectID", "trialID"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["groupID", "DesignCell", "P", "T_ms", "W_ms"], observed=True)
        .agg(
            n_subjects=("SubjectUID", "nunique"),
            n_trials=("trialID", "size"),
            acc=("ACC", "mean"),
            omission_rate=("omission", "mean"),
            rt_mean_ms=("RT_ms", "mean"),
            rt_sd_ms=("RT_ms", "std"),
        )
        .reset_index()
        .sort_values(["P", "T_ms", "W_ms", "groupID"])
    )
    return summary


def main() -> None:
    raw = load_raw_files()
    combined = preprocess(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(combined)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"原始文件数: {raw['SourceFile'].nunique()}")
    print(f"正式试次数: {len(combined)}")
    print(f"被试数: {combined['SubjectUID'].nunique()}")
    print(f"设计单元数: {combined['DesignCell'].nunique()}")
    print(f"已保存: {OUT_PATH}")
    print(f"摘要已保存: {SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
