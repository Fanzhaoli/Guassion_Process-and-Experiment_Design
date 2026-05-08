"""Self-Matching Task 的 SPE ANOVA 与设计空间参数分析。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "2_Data" / "Real_Data" / "EXP_data_combined.csv"
OUT_DIR = BASE_DIR / "3_Figures" / "ANOVA"
TABLE_DIR = OUT_DIR / "tables"
DESIGN_OUT_DIR = OUT_DIR / "DesignSpace"
DESIGN_TABLE_DIR = DESIGN_OUT_DIR / "tables"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def design_key(cell: str) -> tuple[int, int, int]:
    clean = cell.replace("P", "").replace("T", "").replace("W", "")
    return tuple(map(int, clean.split("_")))


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到整合数据，请先运行 preprocess_real_data.py: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["stage"].astype(str).str.lower().eq("formal")].copy()
    df["DesignCell"] = pd.Categorical(
        df["DesignCell"],
        categories=sorted(df["DesignCell"].unique(), key=design_key),
        ordered=True,
    )
    df["Matching"] = pd.Categorical(df["Matching"], categories=["NonMatching", "Matching"])
    df["Identity"] = pd.Categorical(df["Identity"], categories=["Stranger", "Self"])
    df["ACC"] = df["ACC"].astype(int)
    return df


def design_sort(df: pd.DataFrame) -> list[str]:
    design = (
        df[["DesignCell", "P", "T_ms", "W_ms"]]
        .drop_duplicates()
        .sort_values(["P", "T_ms", "W_ms", "DesignCell"])
    )
    return design["DesignCell"].tolist()


def ci95(series: pd.Series) -> float:
    values = series.dropna().to_numpy()
    if len(values) < 2:
        return np.nan
    return stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)


def eta_sq(anova_table: pd.DataFrame) -> pd.DataFrame:
    table = anova_table.copy()
    if "sum_sq" in table.columns:
        total = table["sum_sq"].sum()
        table["eta_sq"] = table["sum_sq"] / total if total > 0 else np.nan
    return table


def add_effect_sizes(anova_table: pd.DataFrame) -> pd.DataFrame:
    """添加 eta squared 与 partial eta squared。"""
    table = eta_sq(anova_table)
    if "sum_sq" in table.columns and "Residual" in table.index:
        residual_ss = table.loc["Residual", "sum_sq"]
        table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
        table.loc["Residual", "partial_eta_sq"] = np.nan
    return table


def aggregate_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # RT 分析只使用正确且有反应时的试次。
    rt_trials = df[(df["ACC"] == 1) & df["RT_ms"].notna() & (df["RT_ms"] > 0)].copy()
    rt_cell = (
        rt_trials.groupby(["SubjectUID", "DesignCell", "P", "T_ms", "W_ms", "Matching", "Identity"], observed=True)
        .agg(RT_ms=("RT_ms", "mean"), n_rt=("RT_ms", "size"))
        .reset_index()
    )

    # ACC 分析保留所有正式试次，遗漏也作为错误进入正确率。
    acc_cell = (
        df.groupby(["SubjectUID", "DesignCell", "P", "T_ms", "W_ms", "Matching", "Identity"], observed=True)
        .agg(ACC=("ACC", "mean"), n_trials=("ACC", "size"), omission_rate=("omission", "mean"))
        .reset_index()
    )
    return rt_cell, acc_cell


def compute_spe(rt_cell: pd.DataFrame, acc_cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rt_match = rt_cell[rt_cell["Matching"] == "Matching"].copy()
    rt_pivot = rt_match.pivot_table(
        index=["SubjectUID", "DesignCell", "P", "T_ms", "W_ms"],
        columns="Identity",
        values="RT_ms",
        observed=True,
    ).reset_index()
    rt_pivot["SPE_RT_ms"] = rt_pivot["Self"] - rt_pivot["Stranger"]

    acc_match = acc_cell[acc_cell["Matching"] == "Matching"].copy()
    acc_pivot = acc_match.pivot_table(
        index=["SubjectUID", "DesignCell", "P", "T_ms", "W_ms"],
        columns="Identity",
        values="ACC",
        observed=True,
    ).reset_index()
    acc_pivot["SPE_ACC"] = acc_pivot["Self"] - acc_pivot["Stranger"]
    return rt_pivot.dropna(subset=["SPE_RT_ms"]), acc_pivot.dropna(subset=["SPE_ACC"])


def run_anova_tables(rt_cell: pd.DataFrame, acc_cell: pd.DataFrame, spe_rt: pd.DataFrame, spe_acc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # 单元均值 ANOVA：用于展示 DesignCell x Matching x Identity 的整体模式。
    rt_model = smf.ols("RT_ms ~ C(DesignCell) * C(Matching) * C(Identity)", data=rt_cell).fit()
    tables["anova_rt_cell_means"] = add_effect_sizes(sm.stats.anova_lm(rt_model, typ=2))

    acc_model = smf.ols("ACC ~ C(DesignCell) * C(Matching) * C(Identity)", data=acc_cell).fit()
    tables["anova_acc_cell_means"] = add_effect_sizes(sm.stats.anova_lm(acc_model, typ=2))

    # SPE 主检验：在 Matching 条件下，Self - Stranger 是否随 P/T/W 设计单元变化。
    spe_rt_model = smf.ols("SPE_RT_ms ~ C(DesignCell)", data=spe_rt).fit()
    tables["anova_spe_rt"] = add_effect_sizes(sm.stats.anova_lm(spe_rt_model, typ=2))

    spe_acc_model = smf.ols("SPE_ACC ~ C(DesignCell)", data=spe_acc).fit()
    tables["anova_spe_acc"] = add_effect_sizes(sm.stats.anova_lm(spe_acc_model, typ=2))
    return tables


def prepare_design_space_data(spe_df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    model_df = spe_df[["SubjectUID", "DesignCell", "P", "T_ms", "W_ms", value_col]].dropna().copy()
    scale: dict[str, tuple[float, float]] = {}
    for raw_col, z_col in [("P", "P_z"), ("T_ms", "T_z"), ("W_ms", "W_z")]:
        mean = model_df[raw_col].mean()
        sd = model_df[raw_col].std(ddof=0)
        if sd == 0:
            raise ValueError(f"{raw_col} 没有变异，无法进行设计空间回归。")
        model_df[z_col] = (model_df[raw_col] - mean) / sd
        scale[raw_col] = (mean, sd)
    return model_df, scale


def fit_design_space_model(
    spe_df: pd.DataFrame,
    value_col: str,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    model_df, scale = prepare_design_space_data(spe_df, value_col)
    model = smf.ols(f"{value_col} ~ P_z * T_z * W_z", data=model_df).fit()
    anova = add_effect_sizes(sm.stats.anova_lm(model, typ=3))
    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )
    fit = pd.DataFrame(
        [
            {
                "outcome": value_col,
                "n": int(model.nobs),
                "r_sq": model.rsquared,
                "adj_r_sq": model.rsquared_adj,
                "f_value": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "aic": model.aic,
                "bic": model.bic,
                "condition_number": model.condition_number,
            }
        ]
    )
    return model, model_df, anova, coef, fit, scale


def fit_hierarchical_design_models(model_df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按层级比较主效应、两两交互和三阶交互的增量解释量。"""
    formulas = {
        "null": f"{value_col} ~ 1",
        "main_effects": f"{value_col} ~ P_z + T_z + W_z",
        "two_way": f"{value_col} ~ (P_z + T_z + W_z) ** 2",
        "three_way": f"{value_col} ~ P_z * T_z * W_z",
    }
    models = {name: smf.ols(formula, data=model_df).fit() for name, formula in formulas.items()}

    rows = []
    comparisons = [("main_effects", "null"), ("two_way", "main_effects"), ("three_way", "two_way")]
    for full_name, reduced_name in comparisons:
        full = models[full_name]
        reduced = models[reduced_name]
        df_diff = full.df_model - reduced.df_model
        if df_diff > 0:
            f_value, p_value, df_diff_test = full.compare_f_test(reduced)
        else:
            f_value, p_value, df_diff_test = np.nan, np.nan, df_diff
        delta_r2 = full.rsquared - reduced.rsquared
        denom = 1.0 - full.rsquared
        cohen_f2 = delta_r2 / denom if denom > 0 else np.nan
        rows.append(
            {
                "block": full_name,
                "reduced_model": reduced_name,
                "full_model": full_name,
                "df_diff": df_diff_test,
                "F": f_value,
                "p": p_value,
                "r_sq_reduced": reduced.rsquared,
                "r_sq_full": full.rsquared,
                "delta_r_sq": delta_r2,
                "cohen_f2": cohen_f2,
                "note": "" if df_diff > 0 else "当前 P/T/W 采样点下三阶项未增加可估计自由度",
            }
        )
    block_tests = pd.DataFrame(rows)

    model_fit = pd.DataFrame(
        [
            {
                "model": name,
                "formula": formulas[name],
                "df_model": model.df_model,
                "df_resid": model.df_resid,
                "r_sq": model.rsquared,
                "adj_r_sq": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "condition_number": model.condition_number,
            }
            for name, model in models.items()
        ]
    )
    return block_tests, model_fit


def predict_design_space(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    scale: dict[str, tuple[float, float]],
    grid: pd.DataFrame,
) -> pd.DataFrame:
    pred = grid.copy()
    for raw_col, z_col in [("P", "P_z"), ("T_ms", "T_z"), ("W_ms", "W_z")]:
        mean, sd = scale[raw_col]
        pred[z_col] = (pred[raw_col] - mean) / sd
    pred["pred"] = model.predict(pred)
    return pred


def plot_spe(spe_df: pd.DataFrame, value_col: str, ylabel: str, title: str, file_name: str, order: list[str]) -> None:
    summary = (
        spe_df.groupby("DesignCell", observed=True)[value_col]
        .agg(mean="mean", ci95=ci95, n="count")
        .reindex(order)
        .reset_index()
    )
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.errorbar(x, summary["mean"], yerr=summary["ci95"], marker="o", linewidth=2, capsize=4, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["DesignCell"], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for idx, row in summary.iterrows():
        ax.text(idx, row["mean"], f"n={int(row['n'])}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / file_name, bbox_inches="tight")
    plt.close(fig)


def plot_interaction(cell_df: pd.DataFrame, value_col: str, ylabel: str, title: str, file_name: str, order: list[str]) -> None:
    summary = (
        cell_df.groupby(["DesignCell", "Matching", "Identity"], observed=True)[value_col]
        .agg(mean="mean", ci95=ci95)
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {"Stranger": "#4C78A8", "Self": "#E45756"}
    for ax, matching in zip(axes, ["NonMatching", "Matching"]):
        sub = summary[summary["Matching"] == matching]
        x = np.arange(len(order))
        for identity in ["Stranger", "Self"]:
            line = sub[sub["Identity"] == identity].set_index("DesignCell").reindex(order)
            ax.errorbar(x, line["mean"], yerr=line["ci95"], marker="o", linewidth=2, capsize=3, label=identity, color=colors[identity])
        ax.set_title(matching)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_xlabel("Design cell")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / file_name, bbox_inches="tight")
    plt.close(fig)


def plot_omission(df: pd.DataFrame, order: list[str]) -> None:
    summary = (
        df.groupby(["DesignCell", "Identity"], observed=True)["omission"]
        .agg(mean="mean", ci95=ci95)
        .reset_index()
    )
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, identity, color in [(-width / 2, "Stranger", "#4C78A8"), (width / 2, "Self", "#E45756")]:
        sub = summary[summary["Identity"] == identity].set_index("DesignCell").reindex(order)
        ax.bar(x + offset, sub["mean"], width=width, yerr=sub["ci95"], capsize=3, label=identity, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Omission rate")
    ax.set_title("Omission Rate by Design Cell and Identity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "omission_by_design_identity.png", bbox_inches="tight")
    plt.close(fig)


def plot_design_observed_summary(spe_df: pd.DataFrame, value_col: str, ylabel: str, prefix: str) -> None:
    summary = (
        spe_df.groupby(["DesignCell", "P", "T_ms", "W_ms"], observed=True)[value_col]
        .agg(mean="mean", ci95=ci95, n="count")
        .reset_index()
        .sort_values(["P", "T_ms", "W_ms"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    specs = [("P", "Practice count P"), ("T_ms", "Stimulus time T (ms)"), ("W_ms", "Response window W (ms)")]
    sizes = 35 + summary["n"] * 5
    for ax, (x_col, xlabel) in zip(axes, specs):
        sc = ax.scatter(summary[x_col], summary["mean"], s=sizes, c=summary["W_ms"], cmap="viridis", alpha=0.86, edgecolor="white", linewidth=0.8)
        ax.errorbar(summary[x_col], summary["mean"], yerr=summary["ci95"], fmt="none", color="#555555", alpha=0.65, capsize=3)
        for _, row in summary.iterrows():
            ax.annotate(str(int(row["n"])), (row[x_col], row["mean"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("W (ms)")
    fig.suptitle(f"Observed Design Cells: {ylabel}", y=1.02)
    fig.tight_layout()
    fig.savefig(DESIGN_OUT_DIR / f"{prefix}_observed_design_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_design_main_effects(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_df: pd.DataFrame,
    scale: dict[str, tuple[float, float]],
    value_col: str,
    ylabel: str,
    prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    specs = [("P", "Practice count P"), ("T_ms", "Stimulus time T (ms)"), ("W_ms", "Response window W (ms)")]
    medians = {col: model_df[col].median() for col in ["P", "T_ms", "W_ms"]}
    for ax, (x_col, xlabel) in zip(axes, specs):
        xs = np.linspace(model_df[x_col].min(), model_df[x_col].max(), 120)
        grid = pd.DataFrame({col: medians[col] for col in ["P", "T_ms", "W_ms"]}, index=np.arange(len(xs)))
        grid[x_col] = xs
        pred = predict_design_space(model, scale, grid)
        ax.plot(xs, pred["pred"], color="#1f77b4", linewidth=2.4)
        ax.scatter(model_df[x_col], model_df[value_col], color="#222222", alpha=0.45, s=22)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Partial Main Effects Holding Other Parameters at Median: {ylabel}", y=1.02)
    fig.tight_layout()
    fig.savefig(DESIGN_OUT_DIR / f"{prefix}_main_effects.png", bbox_inches="tight")
    plt.close(fig)


def plot_design_pairwise_interactions(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_df: pd.DataFrame,
    scale: dict[str, tuple[float, float]],
    ylabel: str,
    prefix: str,
) -> None:
    pairs = [
        ("P", "T_ms", "W_ms", "P", "T (ms)", "W"),
        ("P", "W_ms", "T_ms", "P", "W (ms)", "T"),
        ("T_ms", "W_ms", "P", "T (ms)", "W (ms)", "P"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    medians = {col: model_df[col].median() for col in ["P", "T_ms", "W_ms"]}
    for ax, (x_col, line_col, fixed_col, xlabel, line_label, fixed_label) in zip(axes, pairs):
        xs = np.linspace(model_df[x_col].min(), model_df[x_col].max(), 100)
        line_values = np.quantile(model_df[line_col], [0.15, 0.5, 0.85])
        for line_value, color in zip(line_values, ["#4C78A8", "#59A14F", "#E45756"]):
            grid = pd.DataFrame({col: medians[col] for col in ["P", "T_ms", "W_ms"]}, index=np.arange(len(xs)))
            grid[x_col] = xs
            grid[line_col] = line_value
            pred = predict_design_space(model, scale, grid)
            ax.plot(xs, pred["pred"], color=color, linewidth=2.2, label=f"{line_label}={line_value:.0f}")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{xlabel} x {line_label}\n{fixed_label} fixed at median")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Pairwise Interaction Probes: {ylabel}", y=1.05)
    fig.tight_layout()
    fig.savefig(DESIGN_OUT_DIR / f"{prefix}_pairwise_interactions.png", bbox_inches="tight")
    plt.close(fig)


def plot_design_three_way_heatmaps(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_df: pd.DataFrame,
    scale: dict[str, tuple[float, float]],
    ylabel: str,
    prefix: str,
) -> None:
    p_grid = np.linspace(model_df["P"].min(), model_df["P"].max(), 80)
    t_grid = np.linspace(model_df["T_ms"].min(), model_df["T_ms"].max(), 80)
    w_levels = np.quantile(model_df["W_ms"], [0.15, 0.5, 0.85])
    predictions = []
    vmin = None
    vmax = None
    for w_value in w_levels:
        pp, tt = np.meshgrid(p_grid, t_grid)
        grid = pd.DataFrame({"P": pp.ravel(), "T_ms": tt.ravel(), "W_ms": w_value})
        pred = predict_design_space(model, scale, grid)
        z = pred["pred"].to_numpy().reshape(len(t_grid), len(p_grid))
        predictions.append((w_value, z))
        vmin = np.nanmin(z) if vmin is None else min(vmin, np.nanmin(z))
        vmax = np.nanmax(z) if vmax is None else max(vmax, np.nanmax(z))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (w_value, z) in zip(axes, predictions):
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[p_grid.min(), p_grid.max(), t_grid.min(), t_grid.max()],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(model_df["P"], model_df["T_ms"], s=18, c="black", alpha=0.35)
        ax.set_xlabel("P")
        ax.set_title(f"W = {w_value:.0f} ms")
    axes[0].set_ylabel("T (ms)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"Predicted {ylabel}")
    fig.suptitle("Three-Way Surface Probe: P x T across W Levels", y=1.02)
    fig.tight_layout()
    fig.savefig(DESIGN_OUT_DIR / f"{prefix}_three_way_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def save_tables(tables: dict[str, pd.DataFrame], rt_cell: pd.DataFrame, acc_cell: pd.DataFrame, spe_rt: pd.DataFrame, spe_acc: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rt_cell.to_csv(TABLE_DIR / "rt_subject_cell_means.csv", index=False, encoding="utf-8-sig")
    acc_cell.to_csv(TABLE_DIR / "acc_subject_cell_means.csv", index=False, encoding="utf-8-sig")
    spe_rt.to_csv(TABLE_DIR / "spe_rt_by_subject.csv", index=False, encoding="utf-8-sig")
    spe_acc.to_csv(TABLE_DIR / "spe_acc_by_subject.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", encoding="utf-8-sig")


def save_design_space_results(
    prefix: str,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_df: pd.DataFrame,
    anova: pd.DataFrame,
    coef: pd.DataFrame,
    fit: pd.DataFrame,
    block_tests: pd.DataFrame,
    hierarchy_fit: pd.DataFrame,
) -> None:
    DESIGN_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(DESIGN_TABLE_DIR / f"{prefix}_design_space_model_data.csv", index=False, encoding="utf-8-sig")
    anova.to_csv(DESIGN_TABLE_DIR / f"{prefix}_design_space_anova.csv", encoding="utf-8-sig")
    coef.to_csv(DESIGN_TABLE_DIR / f"{prefix}_design_space_coefficients.csv", index=False, encoding="utf-8-sig")
    fit.to_csv(DESIGN_TABLE_DIR / f"{prefix}_design_space_fit.csv", index=False, encoding="utf-8-sig")
    block_tests.to_csv(DESIGN_TABLE_DIR / f"{prefix}_hierarchical_block_tests.csv", index=False, encoding="utf-8-sig")
    hierarchy_fit.to_csv(DESIGN_TABLE_DIR / f"{prefix}_hierarchical_model_fit.csv", index=False, encoding="utf-8-sig")
    (DESIGN_TABLE_DIR / f"{prefix}_design_space_model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")


def run_design_space_analysis(spe_df: pd.DataFrame, value_col: str, ylabel: str, prefix: str) -> dict[str, pd.DataFrame]:
    DESIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model, model_df, anova, coef, fit, scale = fit_design_space_model(spe_df, value_col)
    block_tests, hierarchy_fit = fit_hierarchical_design_models(model_df, value_col)
    save_design_space_results(prefix, model, model_df, anova, coef, fit, block_tests, hierarchy_fit)
    plot_design_observed_summary(spe_df, value_col, ylabel, prefix)
    plot_design_main_effects(model, model_df, scale, value_col, ylabel, prefix)
    plot_design_pairwise_interactions(model, model_df, scale, ylabel, prefix)
    plot_design_three_way_heatmaps(model, model_df, scale, ylabel, prefix)
    return {"anova": anova, "coef": coef, "fit": fit, "block_tests": block_tests, "hierarchy_fit": hierarchy_fit}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = load_data()
    order = design_sort(df)
    rt_cell, acc_cell = aggregate_cells(df)
    spe_rt, spe_acc = compute_spe(rt_cell, acc_cell)
    tables = run_anova_tables(rt_cell, acc_cell, spe_rt, spe_acc)
    save_tables(tables, rt_cell, acc_cell, spe_rt, spe_acc)

    plot_spe(spe_rt, "SPE_RT_ms", "Self - Stranger RT (ms)", "SPE_RT in Matching Trials by P/T/W Design", "spe_rt_by_design.png", order)
    plot_spe(spe_acc, "SPE_ACC", "Self - Stranger ACC", "SPE_ACC in Matching Trials by P/T/W Design", "spe_acc_by_design.png", order)
    plot_interaction(rt_cell, "RT_ms", "RT (ms)", "RT Interaction: Design x Matching x Identity", "rt_design_matching_identity.png", order)
    plot_interaction(acc_cell, "ACC", "Accuracy", "ACC Interaction: Design x Matching x Identity", "acc_design_matching_identity.png", order)
    plot_omission(df, order)

    design_rt = run_design_space_analysis(spe_rt, "SPE_RT_ms", "Self - Stranger RT (ms)", "spe_rt")
    design_acc = run_design_space_analysis(spe_acc, "SPE_ACC", "Self - Stranger ACC", "spe_acc")

    print(f"正式试次数: {len(df)}")
    print(f"被试数: {df['SubjectUID'].nunique()}")
    print(f"设计单元: {', '.join(order)}")
    print("\n=== SPE_RT DesignCell ANOVA ===")
    print(tables["anova_spe_rt"].to_string())
    print("\n=== SPE_ACC DesignCell ANOVA ===")
    print(tables["anova_spe_acc"].to_string())
    print("\n=== SPE_RT Design Space: P * T * W ===")
    print(design_rt["anova"].to_string())
    print("\n=== SPE_RT Hierarchical Blocks ===")
    print(design_rt["block_tests"].to_string(index=False))
    print("\n=== SPE_ACC Design Space: P * T * W ===")
    print(design_acc["anova"].to_string())
    print("\n=== SPE_ACC Hierarchical Blocks ===")
    print(design_acc["block_tests"].to_string(index=False))
    print(f"\n图表与表格已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
