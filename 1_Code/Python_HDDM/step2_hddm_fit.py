import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/data")
DATA_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Ready"
OUT_DIR = BASE_DIR / "2_Data" / "Real_Data" / "HDDM_Traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("hddm_data_group*.csv"))
print(f"发现 {len(csv_files)} 个待拟合数据文件")

for csv_path in csv_files:
    fname = csv_path.stem
    print(f"\n{'=' * 50}")
    print(f"拟合: {fname}")
    print(f"{'=' * 50}")

    df = pd.read_csv(csv_path)
    n_subj = df["subj_idx"].nunique()
    n_trials = len(df)
    print(f"  被试数: {n_subj}, 试次数: {n_trials}")

    n_self = (df["identity"] == 1).sum()
    n_stranger = (df["identity"] == 0).sum()
    print(f"  Self试次: {n_self}, Stranger试次: {n_stranger}")

    n_omission = df["omission"].sum()
    print(f"  遗漏试次: {n_omission}")
    df_fit = df.copy()
    df_fit = df_fit.dropna(subset=["rt"])

    if len(df_fit) < 50:
        print(f"  ⚠️ 有效试次过少 ({len(df_fit)})，跳过此组")
        continue

    try:
        import hddm
    except ImportError:
        print("  ❌ hddm 未安装，请确认在 Docker 容器中运行")
        sys.exit(1)

    model = hddm.HDDM(
        df_fit,
        depends_on={"v": "identity"},
        include=["v", "a", "t", "z"],
        bias=False,
        p_outlier=0.05,
    )

    print("  开始 MCMC 采样...")
    model.sample(3000, burn=500, dbname="traces.db", db="pickle")
    print("  采样完成")

    traces = model.get_traces()
    npz_path = OUT_DIR / f"{fname}_traces.npz"
    trace_dict = {}
    for key, val in traces.items():
        if isinstance(val, np.ndarray):
            trace_dict[key] = val
        elif isinstance(val, pd.DataFrame):
            trace_dict[key] = val.values
    np.savez_compressed(npz_path, **trace_dict)
    print(f"  迹线已保存到: {npz_path}")

    stats = model.gen_stats()
    stats_path = OUT_DIR / f"{fname}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  统计已保存到: {stats_path}")

    summary_rows = []
    for gid, gdf in df_fit.groupby("identity"):
        label = "self" if gid == 1 else "stranger"
        valid = gdf[gdf["omission"] == 0]
        mean_rt = valid["rt"].mean()
        acc = valid["response"].mean()
        n_total = len(gdf)
        n_omis = gdf["omission"].sum()
        summary_rows.append(
            {
                "group_file": fname,
                "condition": label,
                "n_trials": n_total,
                "n_valid": len(valid),
                "n_omission": n_omis,
                "mean_rt_s": mean_rt,
                "accuracy": acc,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / f"{fname}_behavior_summary.csv", index=False
    )

print(f"\n{'=' * 50}")
print("所有拟合完成!")
print(f"结果保存于: {OUT_DIR}")

