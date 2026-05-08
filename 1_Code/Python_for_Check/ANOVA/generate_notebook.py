"""生成 SPE_Bayesian_Analysis.ipynb。"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("SPE_Bayesian_Analysis.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """# Self-Matching Task：SPE 的实验设计空间分析

本 notebook 完成三层分析：

1. 从 `2_Data/Real_Data/UnExtact/raw` 整合正式实验数据，输出 `EXP_data_combined.csv`
2. 用 `DesignCell` 检验不同 P/T/W 设计单元下 SPE 是否不同
3. 将 P、T、W 拆成连续设计空间参数，检验单个参数、两两交互与三阶交互对 SPE 的影响

注意：当前 P/T/W 采样是稀疏设计，并非完整正交 factorial。因此参数拆解结果应解释为探索性证据；整体设计单元效应是更稳的行为层面检验。
"""
    ),
    md("## 1. 路径与函数导入"),
    code(
        """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ANOVA_DIR = Path.cwd()
if ANOVA_DIR.name != "ANOVA":
    ANOVA_DIR = Path(r"d:\\GitHub_programe\\GitHub\\Guassion-Process-Experiment-Design\\1_Code\\Python_for_Check\\ANOVA")

BASE_DIR = ANOVA_DIR.parents[2]
sys.path.insert(0, str(ANOVA_DIR))

from preprocess_real_data import RAW_DIR, OUT_PATH, SUMMARY_PATH, load_raw_files, preprocess, build_summary
from run_spe_anova import (
    OUT_DIR,
    DESIGN_OUT_DIR,
    DESIGN_TABLE_DIR,
    TABLE_DIR,
    aggregate_cells,
    compute_spe,
    design_sort,
    load_data,
    plot_design_main_effects,
    plot_design_observed_summary,
    plot_design_pairwise_interactions,
    plot_design_three_way_heatmaps,
    plot_interaction,
    plot_omission,
    plot_spe,
    run_anova_tables,
    run_design_space_analysis,
    save_tables,
    setup_style,
)

print(f"项目目录: {BASE_DIR}")
print(f"原始数据目录: {RAW_DIR}")
print(f"输出目录: {OUT_DIR}")"""
    ),
    md("## 2. 数据预处理与整合"),
    code(
        """raw = load_raw_files(RAW_DIR)
combined = preprocess(raw, keep_stage="formal")
summary = build_summary(combined)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

print(f"原始文件数: {raw['SourceFile'].nunique()}")
print(f"正式试次数: {len(combined)}")
print(f"被试数: {combined['SubjectUID'].nunique()}")
print(f"设计单元数: {combined['DesignCell'].nunique()}")
display(summary)"""
    ),
    md("## 3. 计算被试层级 SPE"),
    code(
        """df = load_data()
order = design_sort(df)
rt_cell, acc_cell = aggregate_cells(df)
spe_rt, spe_acc = compute_spe(rt_cell, acc_cell)

print(f"RT 单元均值行数: {len(rt_cell)}")
print(f"ACC 单元均值行数: {len(acc_cell)}")
print(f"SPE_RT 被试行数: {len(spe_rt)}")
print(f"SPE_ACC 被试行数: {len(spe_acc)}")

display(spe_rt.head())
display(spe_acc.head())"""
    ),
    md(
        """## 4. DesignCell 整体效应

这里检验：不同 P/T/W 组合是否导致 SPE 不同。负的 `SPE_RT_ms = Self - Stranger` 表示 self 更快。"""
    ),
    code(
        """tables = run_anova_tables(rt_cell, acc_cell, spe_rt, spe_acc)
save_tables(tables, rt_cell, acc_cell, spe_rt, spe_acc)

print("SPE_RT DesignCell ANOVA")
display(tables["anova_spe_rt"])

print("SPE_ACC DesignCell ANOVA")
display(tables["anova_spe_acc"])"""
    ),
    md(
        """## 5. P/T/W 参数拆解：主效应与交互

模型形式：

`SPE ~ P_z * T_z * W_z`

其中 P、T、W 做 z 标准化。输出包括：

- Type-III ANOVA：每个单项、两两交互、三阶交互的 F、p、eta squared、partial eta squared
- 分层模型比较：主效应整体、两两交互整体、三阶交互整体的 delta R² 与 Cohen f²
- 标准化回归系数：方向与置信区间
"""
    ),
    code(
        """setup_style()
design_rt = run_design_space_analysis(spe_rt, "SPE_RT_ms", "Self - Stranger RT (ms)", "spe_rt")
design_acc = run_design_space_analysis(spe_acc, "SPE_ACC", "Self - Stranger ACC", "spe_acc")

print("SPE_RT: P/T/W Type-III ANOVA")
display(design_rt["anova"])

print("SPE_RT: 分层模型比较")
display(design_rt["block_tests"])

print("SPE_ACC: P/T/W Type-III ANOVA")
display(design_acc["anova"])

print("SPE_ACC: 分层模型比较")
display(design_acc["block_tests"])"""
    ),
    md("## 6. 可视化输出"),
    code(
        """plot_spe(spe_rt, "SPE_RT_ms", "Self - Stranger RT (ms)", "SPE_RT in Matching Trials by P/T/W Design", "spe_rt_by_design.png", order)
plot_spe(spe_acc, "SPE_ACC", "Self - Stranger ACC", "SPE_ACC in Matching Trials by P/T/W Design", "spe_acc_by_design.png", order)
plot_interaction(rt_cell, "RT_ms", "RT (ms)", "RT Interaction: Design x Matching x Identity", "rt_design_matching_identity.png", order)
plot_interaction(acc_cell, "ACC", "Accuracy", "ACC Interaction: Design x Matching x Identity", "acc_design_matching_identity.png", order)
plot_omission(df, order)

print("ANOVA 图表:")
for path in sorted(OUT_DIR.glob("*.png")):
    print(path.name)

print("\\n设计空间图表:")
for path in sorted(DESIGN_OUT_DIR.glob("*.png")):
    print(path.name)"""
    ),
    md("### 6.1 设计空间图预览"),
    code(
        """fig_paths = [
    DESIGN_OUT_DIR / "spe_rt_observed_design_summary.png",
    DESIGN_OUT_DIR / "spe_rt_main_effects.png",
    DESIGN_OUT_DIR / "spe_rt_pairwise_interactions.png",
    DESIGN_OUT_DIR / "spe_rt_three_way_heatmaps.png",
    DESIGN_OUT_DIR / "spe_acc_observed_design_summary.png",
    DESIGN_OUT_DIR / "spe_acc_main_effects.png",
]

for fig_path in fig_paths:
    img = plt.imread(fig_path)
    plt.figure(figsize=(12, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(fig_path.name)
    plt.show()"""
    ),
    md("## 7. 描述统计"),
    code(
        """spe_rt_summary = (
    spe_rt.groupby(["DesignCell", "P", "T_ms", "W_ms"], observed=True)["SPE_RT_ms"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values(["P", "T_ms", "W_ms"])
)
spe_acc_summary = (
    spe_acc.groupby(["DesignCell", "P", "T_ms", "W_ms"], observed=True)["SPE_ACC"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values(["P", "T_ms", "W_ms"])
)

display(spe_rt_summary)
display(spe_acc_summary)"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已生成: {NOTEBOOK_PATH}")
