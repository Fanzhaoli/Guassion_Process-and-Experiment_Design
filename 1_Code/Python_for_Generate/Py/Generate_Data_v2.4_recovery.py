import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import importlib.util, sys
runner_path = Path(__file__).resolve().parents[1] / "Python_for_Generate" / "Generate_Data_v2.4_runner.py"
spec = importlib.util.spec_from_file_location("gdrv2_runner", str(runner_path))
gdrv2 = importlib.util.module_from_spec(spec)
sys.modules["gdrv2_runner"] = gdrv2
spec.loader.exec_module(gdrv2)
generate_dataset_s2 = gdrv2.generate_dataset_s2
normalize_PTW_to_unit = gdrv2.normalize_PTW_to_unit
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data"
FIG_DIR = BASE_DIR / "3_Figures" / "Generate_Data_v2.4_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def build_X(df):
    Pn, Tn, Wn = zip(*[normalize_PTW_to_unit(r.P, r.T, r.W) for r in df.itertuples()])
    label = (df['label'] == 'self').astype(float).values
    X = np.column_stack([Pn, Tn, Wn, label])
    return X

def fit_and_score_gp(X_train, y_train, X_test, y_test):
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_train, y_train)
    y_pred, std = gp.predict(X_test, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    try:
        r, _ = pearsonr(y_test, y_pred)
    except Exception:
        r = np.nan
    return rmse, r

def run_recovery(sizes=(20,50,100), v_noises=(0.2,0.5,1.0,2.0), repeats=5):
    rows = []
    for n_subj in sizes:
        for v_noise in v_noises:
            for rep in range(repeats):
                df, gen = generate_dataset_s2(n_subjects=n_subj, trials_per_sub=60, w_gp=0.5, v_noise=v_noise, a_noise=0.5, seed=1000+rep)
                # use trial-level true v and a as targets
                X = build_X(df)
                y_v = df['v'].values
                y_a = df['a'].values
                X_train, X_test, yv_train, yv_test = train_test_split(X, y_v, test_size=0.2, random_state=42)
                _, _, ya_train, ya_test = train_test_split(X, y_a, test_size=0.2, random_state=42)
                rmse_v, r_v = fit_and_score_gp(X_train, yv_train, X_test, yv_test)
                rmse_a, r_a = fit_and_score_gp(X_train, ya_train, X_test, ya_test)
                rows.append({'n_subjects': n_subj, 'v_noise': v_noise, 'rep': rep, 'rmse_v': rmse_v, 'r_v': r_v, 'rmse_a': rmse_a, 'r_a': r_a})
                print(f"n={n_subj} v_noise={v_noise} rep={rep} -> rmse_v={rmse_v:.3f} r_v={r_v:.3f} rmse_a={rmse_a:.3f} r_a={r_a:.3f}")
    res = pd.DataFrame(rows)
    res_path = OUT_DIR / 'recovery_results_v2.4.csv'
    res.to_csv(res_path, index=False)

    # summary plots
    for metric in ['rmse_v','r_v','rmse_a','r_a']:
        plt.figure(figsize=(8,5))
        for n_subj in sizes:
            subset = res[res['n_subjects']==n_subj]
            means = subset.groupby('v_noise')[metric].mean()
            plt.plot(means.index, means.values, marker='o', label=f'n={n_subj}')
        plt.xlabel('v_noise')
        plt.ylabel(metric)
        plt.title(metric + ' by v_noise and n_subjects')
        plt.legend()
        plt.savefig(FIG_DIR / f'{metric}_by_vnoise.png', dpi=200)
        plt.close()
    print('Saved recovery results to', res_path)
    print('Saved recovery figures to', FIG_DIR)

if __name__ == '__main__':
    run_recovery()
