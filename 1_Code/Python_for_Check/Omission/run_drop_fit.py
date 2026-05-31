import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import re
import os
import sys
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity" / "drop_omission"
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data" / "Omission_Sensitivity" / "drop_traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("hddm_data_group*_drop.csv"))
if not csv_files:
    print(f"未找到 drop 数据文件！路径: {DATA_DIR}")
    print("请先运行 prepare_data.py")
    sys.exit(1)

print(f"发现 {len(csv_files)} 个 drop 数据文件")

try:
    import hddm
except ImportError:
    print("hddm 未安装，请在 Docker 容器 (hcp4715/hddm) 中运行")
    print("Docker 启动命令:")
    print("docker run -it --rm --cpus=4 -v ")
    print("  /d/GitHub_programe/GitHub/Guassion-Process-Experiment-Design:/home/jovyan/work ")
    print("  -p 8888:8888 hcp4715/hddm jupyter notebook")
    sys.exit(1)

for csv_path in csv_files:
    fname = csv_path.stem
    print(f"\n{'=' * 50}")
    print(f"拟合 [DROP]: {fname}")
    print(f"{'=' * 50}")

    df = pd.read_csv(csv_path)
    n_subj = df["subj_idx"].nunique()
    n_trials = len(df)
    n_self = (df["identity"] == 1).sum()
    n_stranger = (df["identity"] == 0).sum()
    acc = df["response"].mean()

    print(f"  被试数: {n_subj}, 试次数: {n_trials}")
    print(f"  Self: {n_self}, Stranger: {n_stranger}")
    print(f"  正确率: {acc:.3f}")

    model = hddm.HDDM(
        df,
        depends_on={"v": "identity"},
        include=["v", "a", "t", "z"],
        bias=False,
        p_outlier=0.05,
    )

    print("  开始 MCMC 采样 (3000 draws, 500 burn)...")
    db_name = f"traces_drop_{fname}.db"
    model.sample(3000, burn=500, dbname=db_name, db="pickle")
    print("  采样完成")

    stats = model.gen_stats()
    stats_path = OUT_DIR / f"{fname}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  统计 → {stats_path}")

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
            print(f"  迹线 (pickle) → {trace_path}")

            npz_path = OUT_DIR / f"{fname}_traces.npz"
            np.savez_compressed(npz_path, **traces_simple)
            print(f"  迹线 (npz) → {npz_path}")
        else:
            print("  ⚠️ get_traces() 返回空，迹线未保存")
    except Exception as e:
        print(f"  ⚠️ 迹线保存失败: {e}")

    try:
        os.remove(db_name)
    except Exception:
        pass

print(f"\n{'=' * 50}")
print("所有 Drop 数据拟合完成!")
print(f"结果保存于: {OUT_DIR}")
print(f"\n下一步: 在本机运行 analyze_sensitivity.py 进行比较分析")