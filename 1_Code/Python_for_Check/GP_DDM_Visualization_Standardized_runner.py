from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
GEN_DIR = BASE_DIR / '2_Data' / 'Generate_Data'
REAL_DIR = BASE_DIR / '2_Data' / 'Real_Data'
FIG_BASE = BASE_DIR / '3_Figures' / 'GP_DDM_Visualization_Standardized'

FIG_VERSION = FIG_BASE / '01_Version_Compare'
FIG_SPE = FIG_BASE / '02_GP_SPE_Surface'
FIG_GPMAP = FIG_BASE / '03_GP_DDM_Mapping'
FIG_SIG_GGP = FIG_BASE / '04_Sigmoid_vs_GP'
for d in [FIG_VERSION, FIG_SPE, FIG_GPMAP, FIG_SIG_GGP]:
    d.mkdir(parents=True, exist_ok=True)

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(paths)

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def to_ms(rt):
    x = to_num(rt)
    mx = np.nanmax(x.to_numpy(dtype=float))
    return x * 1000.0 if mx <= 10 else x

def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def standardize_generated(df, version_name):
    df = df.copy()
    colmap = {c.lower(): c for c in df.columns}
    def get(name):
        return df[colmap[name]].copy() if name in colmap else pd.Series([np.nan] * len(df))
    return pd.DataFrame({
        'version': version_name,
        'source_type': 'generated',
        'subject': to_num(get('subject')) if 'subject' in colmap else to_num(get('subjectid')),
        'trial': to_num(get('trial')) if 'trial' in colmap else to_num(get('trialid')),
        'P': to_num(get('p')),
        'T': to_num(get('t')),
        'W': to_num(get('w')),
        'M': to_num(get('m')),
        'label': get('label').astype(str).str.lower(),
        'v': to_num(get('v')),
        'a': to_num(get('a')),
        't0': to_num(get('t0')),
        'z': to_num(get('z')),
        'RT_ms': to_ms(get('rt')),
        'response': to_num(get('response')),
        'correct': to_num(get('correct')) if 'correct' in colmap else to_num(get('response')),
    })

def standardize_real(df):
    return pd.DataFrame({
        'version': 'real',
        'source_type': 'real',
        'subject': to_num(df['subjectID'] if 'subjectID' in df.columns else df['subject']),
        'trial': to_num(df['trialID'] if 'trialID' in df.columns else df['trial']),
        'P': to_num(df['P']),
        'T': to_num(df['T']),
        'W': to_num(df['W']),
        'M': to_num(df['P']) + to_num(df['T']) + to_num(df['W']),
        'label': df['Label'].astype(str).str.lower(),
        'v': np.nan,
        'a': np.nan,
        't0': np.nan,
        'z': np.nan,
        'RT_ms': to_ms(df['RT']),
        'response': df['Response'].astype(str),
        'correct': to_num(df['Correct']),
    })

def build_design_spe(df):
    tmp = df.dropna(subset=['P', 'T', 'W', 'label', 'RT_ms']).copy()
    g = tmp.groupby(['P', 'T', 'W', 'label'], as_index=False)['RT_ms'].mean()
    pivot = g.pivot_table(index=['P', 'T', 'W'], columns='label', values='RT_ms')
    if 'self' not in pivot.columns or 'stranger' not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=['self', 'stranger']).reset_index()
    pivot['SPE_ms'] = pivot['self'] - pivot['stranger']
    return pivot.rename(columns={'self': 'RT_self_ms', 'stranger': 'RT_stranger_ms'})

def fit_gp(df, target, feature_cols=('P', 'T', 'W')):
    clean = df.dropna(subset=list(feature_cols) + [target]).copy()
    X = clean.loc[:, feature_cols].astype(float).to_numpy()
    y = clean[target].astype(float).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=np.ones(Xs.shape[1]), length_scale_bounds=(0.1, 50.0)) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42)
    gp.fit(Xs, y)
    return gp, scaler, clean

def make_grid(df, x_col='P', y_col='T', fixed_col='W', fixed_value=None, n=50):
    x = np.linspace(df[x_col].quantile(0.05), df[x_col].quantile(0.95), n)
    y = np.linspace(df[y_col].quantile(0.05), df[y_col].quantile(0.95), n)
    Xg, Yg = np.meshgrid(x, y)
    if fixed_value is None:
        fixed_value = df[fixed_col].median()
    grid = pd.DataFrame({x_col: Xg.ravel(), y_col: Yg.ravel(), fixed_col: fixed_value})
    return Xg, Yg, grid, fixed_value

def predict_surface(gp, scaler, grid_df, feature_cols=('P', 'T', 'W')):
    Xg = scaler.transform(grid_df.loc[:, feature_cols].astype(float).to_numpy())
    return gp.predict(Xg, return_std=True)

def sigmoid_features(X, centers=None, scales=None):
    X = np.asarray(X, dtype=float)
    if centers is None:
        centers = np.median(X, axis=0)
    if scales is None:
        scales = np.std(X, axis=0) + 1e-6
    Z = expit((X - centers) / scales)
    return np.column_stack([np.ones(len(X)), Z, X])

def fit_sigmoid_baseline(X_train, y_train, X_test):
    centers = np.median(X_train, axis=0)
    scales = np.std(X_train, axis=0) + 1e-6
    Phi_train = sigmoid_features(X_train, centers, scales)
    Phi_test = sigmoid_features(X_test, centers, scales)
    model = Ridge(alpha=1.0)
    model.fit(Phi_train, y_train)
    return model.predict(Phi_test)

def plot_heatmap_and_boundary(df, target, version_name, x_col='P', y_col='T', fixed_col='W', fixed_value=None, out_dir=FIG_SPE):
    gp, scaler, clean = fit_gp(df, target)
    Xg, Yg, grid, fixed_value = make_grid(clean, x_col=x_col, y_col=y_col, fixed_col=fixed_col, fixed_value=fixed_value)
    mean, std = predict_surface(gp, scaler, grid)
    mean = mean.reshape(Xg.shape)
    std = std.reshape(Xg.shape)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].contourf(Xg, Yg, mean, levels=20, cmap='viridis')
    axes[0].scatter(clean[x_col], clean[y_col], s=10, c='white', alpha=0.35, edgecolor='none')
    axes[0].set_title(f'{version_name} | GP mean: {target}')
    axes[0].set_xlabel(x_col); axes[0].set_ylabel(y_col)
    fig.colorbar(im0, ax=axes[0], label=target)
    im1 = axes[1].contourf(Xg, Yg, std, levels=20, cmap='magma')
    axes[1].contour(Xg, Yg, std, levels=[np.quantile(std, 0.8)], colors='cyan', linewidths=2)
    axes[1].scatter(clean[x_col], clean[y_col], s=10, c='white', alpha=0.35, edgecolor='none')
    axes[1].set_title(f'{version_name} | GP std: {target}')
    axes[1].set_xlabel(x_col); axes[1].set_ylabel(y_col)
    fig.colorbar(im1, ax=axes[1], label='std')
    plt.tight_layout()
    save_fig(fig, out_dir / version_name / f'{version_name}_{target}_{x_col}_{y_col}_{fixed_col}_gp_mean_std.png')

def plot_3d_surface(df, target, version_name, x_col='T', y_col='W', fixed_col='P', fixed_value=None, out_dir=FIG_SPE):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    gp, scaler, clean = fit_gp(df, target)
    Xg, Yg, grid, fixed_value = make_grid(clean, x_col=x_col, y_col=y_col, fixed_col=fixed_col, fixed_value=fixed_value)
    mean, _ = predict_surface(gp, scaler, grid)
    Z = mean.reshape(Xg.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Z, cmap='viridis', alpha=0.88, linewidth=0)
    sel = np.abs(clean[fixed_col] - fixed_value) <= max(1.0, clean[fixed_col].std() * 0.15)
    ax.scatter(clean.loc[sel, x_col], clean.loc[sel, y_col], clean.loc[sel, target], c='red', s=18, alpha=0.6)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_zlabel(target)
    fig.colorbar(surf, shrink=0.65)
    plt.tight_layout()
    save_fig(fig, out_dir / version_name / f'{version_name}_{target}_{x_col}_{y_col}_{fixed_col}_gp_3d.png')

def gp_vs_sigmoid_compare(df, target, version_name, x_cols=('P','T','W'), out_dir=FIG_SIG_GGP):
    clean = df.dropna(subset=list(x_cols)+[target]).copy()
    X = clean.loc[:, x_cols].astype(float).to_numpy()
    y = clean[target].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    gp, scaler, _ = fit_gp(clean, target, feature_cols=x_cols)
    gp_pred = gp.predict(scaler.transform(X_test))
    sig_pred = fit_sigmoid_baseline(X_train, y_train, X_test)
    metrics = pd.DataFrame([
        {'model': 'GP', 'rmse': np.sqrt(mean_squared_error(y_test, gp_pred)), 'r2': r2_score(y_test, gp_pred)},
        {'model': 'Sigmoid', 'rmse': np.sqrt(mean_squared_error(y_test, sig_pred)), 'r2': r2_score(y_test, sig_pred)},
    ])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred, title in [(axes[0], gp_pred, 'GP'), (axes[1], sig_pred, 'Sigmoid')]:
        ax.scatter(y_test, pred, s=18, alpha=0.75)
        mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
        ax.set_title(f'{version_name} | {target} | {title}')
        ax.set_xlabel('Observed'); ax.set_ylabel('Predicted')
        ax.grid(alpha=0.25)
    plt.tight_layout()
    save_fig(fig, out_dir / version_name / f'{version_name}_{target}_sigmoid_vs_gp.png')
    metrics.to_csv(out_dir / version_name / f'{version_name}_{target}_sigmoid_vs_gp_metrics.csv', index=False, encoding='utf-8-sig')

def main():
    v1_path = first_existing(sorted(GEN_DIR.glob('gp_ddm_simulation_v1*.csv')))
    v24_path = first_existing([
        GEN_DIR / 'Generate_Data_v2.4.5_checks' / 'gp_ddm_v2.4.5_large.csv',
        GEN_DIR / 'Generate_Data_v2.4.4_checks' / 'gp_ddm_v2.4.4_large.csv',
        GEN_DIR / 'Generate_Data_v2.4.3_checks' / 'gp_ddm_v2.4.3_large.csv',
        GEN_DIR / 'Generate_Data_v2.4_checks' / 'gp_ddm_v2.4_large.csv',
    ])
    v25_path = first_existing([GEN_DIR / 'Generate_Data_v2.5' / 'gp_ddm_v2.5_2000.csv'])
    real_path = REAL_DIR / 'EXP_data_combined.csv'

    dfs = {
        'v1': standardize_generated(pd.read_csv(v1_path), 'v1'),
        'v2.4.x': standardize_generated(pd.read_csv(v24_path), 'v2.4.x'),
        'v2.5': standardize_generated(pd.read_csv(v25_path), 'v2.5'),
        'real': standardize_real(pd.read_csv(real_path)),
    }
    design_tables = {name: build_design_spe(df) for name, df in dfs.items()}

    rows = []
    for name, df in dfs.items():
        spe_tbl = design_tables[name]
        rows.append({'version': name, 'n_rows': len(df), 'spe_ms': spe_tbl['SPE_ms'].mean() if len(spe_tbl) else np.nan})
    pd.DataFrame(rows).to_csv(FIG_VERSION / 'version_summary.csv', index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(rows).plot(kind='bar', x='version', y='spe_ms', ax=ax, color='#5470C6', legend=False)
    ax.set_title('SPE by Version'); ax.set_ylabel('SPE (ms)'); ax.grid(alpha=0.25)
    plt.tight_layout(); save_fig(fig, FIG_VERSION / 'version_compare_overview.png')

    for version_name in ['v1', 'v2.4.x', 'v2.5', 'real']:
        df = design_tables[version_name]
        if df.empty:
            continue
        plot_heatmap_and_boundary(df, 'SPE_ms', version_name, x_col='P', y_col='T', fixed_col='W', out_dir=FIG_SPE)
        plot_3d_surface(df, 'SPE_ms', version_name, x_col='T', y_col='W', fixed_col='P', out_dir=FIG_SPE)
    for version_name in ['v2.4.x', 'v2.5']:
        df = dfs[version_name]
        for target in [c for c in ['v', 'a', 't0', 'z'] if c in df.columns and df[c].notna().any()]:
            plot_heatmap_and_boundary(df, target, version_name, x_col='P', y_col='T', fixed_col='W', out_dir=FIG_GPMAP)
            plot_3d_surface(df, target, version_name, x_col='T', y_col='W', fixed_col='P', out_dir=FIG_GPMAP)
    for version_name in ['v1', 'v2.4.x', 'v2.5', 'real']:
        df = design_tables[version_name]
        if df.empty:
            continue
        gp_vs_sigmoid_compare(df, 'SPE_ms', version_name, out_dir=FIG_SIG_GGP)
    print('Done.')

if __name__ == '__main__':
    main()
