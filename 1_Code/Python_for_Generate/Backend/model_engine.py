import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.special import expit as sigmoid
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parents[3]

# =============================================================================
# S2 机制共享函数 (所有版本共用)
# =============================================================================

def k_P(P, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    return k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))

def v_P_Function(P, P1=4, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    k = k_P(P, k_min, k_max, gamma, P0)
    return 1 / (1 + np.exp(-k * (P - P1)))

def compute_v_s2(T, P, condition_key, alaph1=1.5, alaph2=-0.4, gamma=0.2):
    T_0 = 100
    k_T = 0.01
    v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=gamma, P0=32)
    v_0 = v_T * v_P * 3.0
    if condition_key == 1:
        return v_0 * (1 + alaph1)
    return v_0 * (1 + alaph2)

def compute_a_s2(M, beta1=0.2, beta2=0.0, k=0.01, M_0=600):
    a_0 = 1 / (1 + np.exp(-k * (M - M_0))) * 3.0
    if M > 600:
        return a_0 * (1 + beta1)
    return a_0 * (1 + beta2)

def normalize_PTW_to_unit(P, T, W):
    P_norm = (P - 75.0) / 75.0
    T_norm = (T - 305.0) / 295.0
    W_norm = (W - 850.0) / 650.0
    return P_norm, T_norm, W_norm

# =============================================================================
# DDM 仿真器 (不同版本使用不同实现)
# =============================================================================

def simulate_ddm_euler(v, a, z, t0, dt=0.001, max_time_s=2.0):
    x = float(z)
    time = 0.0
    max_steps = int(max_time_s / dt)
    for _ in range(max_steps):
        dx = v * dt + np.sqrt(dt) * np.random.randn()
        x += dx
        time += dt
        if x >= a:
            return t0 + time, 1
        if x <= 0:
            return t0 + time, 0
    return np.nan, np.nan

def simulate_ddm_with_deadline(v, a, z, t0, deadline_s, dt=0.001):
    decision_budget = deadline_s - t0
    if np.isnan(decision_budget) or decision_budget <= dt:
        return np.nan, 0, 1, 'deadline'
    x = float(z)
    time = 0.0
    max_steps = int(decision_budget / dt)
    for _ in range(max_steps):
        dx = v * dt + np.sqrt(dt) * np.random.randn()
        x += dx
        time += dt
        if x >= a:
            return t0 + time, 1, 0, 'upper'
        if x <= 0:
            return t0 + time, 2, 0, 'lower'
    return np.nan, 0, 1, 'deadline'

def compute_lapse_omission_prob(T_ms, W_ms, lapse_max=0.35, T_mid=80, W_mid=600, T_scale=25, W_scale=120):
    t_term = 1.0 / (1.0 + np.exp((T_ms - T_mid) / T_scale))
    w_term = 1.0 / (1.0 + np.exp((W_ms - W_mid) / W_scale))
    return float(lapse_max * t_term * w_term)

def sample_a_positive(a_mix, a_noise, a_floor=0.03, max_resample=30):
    for _ in range(max_resample):
        a_candidate = np.random.normal(a_mix, a_noise)
        if a_candidate > a_floor:
            return a_candidate
    return max(a_floor + abs(np.random.normal(0, a_noise * 0.25)), a_mix * 0.5, a_floor)

# =============================================================================
# Hybrid GP 参数生成器 (v2.4/v2.4.5/v2.5 共用)
# =============================================================================

class HybridDDMParameterGenerator:
    def __init__(self, w=0.5):
        self.w = w
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        self.gp_v = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_a = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.beta_v = np.array([0.01, 0.02, -0.01])
        self.beta_a = np.array([0.005, -0.01, 0.015])

    def sigmoid_part(self, X, beta):
        return sigmoid(X @ beta)

    def fit_gp(self, X, Y_v, Y_a):
        self.gp_v.fit(X, Y_v)
        self.gp_a.fit(X, Y_a)

    def predict_params(self, X):
        sig_v = self.sigmoid_part(X, self.beta_v)
        sig_a = self.sigmoid_part(X, self.beta_a)
        gp_v = self.gp_v.predict(X)
        gp_a = self.gp_a.predict(X)
        v = self.w * sig_v + (1 - self.w) * gp_v
        a = self.w * sig_a + (1 - self.w) * gp_a
        t0 = np.full(len(v), 0.2)
        z = a / 2.0
        return v, a, t0, z

# =============================================================================
# 模型版本 1: Sigmoid + DDM (S2 gen_data_jh.ipynb)
# =============================================================================

def generate_sigmoid_ddm(n_subjects=10, trials_per_sub=60, seed=None,
                         alaph1=1.5, alaph2=-0.4, beta1=0.2, beta2=0.0,
                         gamma=0.2, t0_val=0.2):
    if seed is not None:
        np.random.seed(seed)

    rows = []
    for subject in range(n_subjects):
        T = np.random.randint(10, 600)
        P = np.random.randint(0, 150)
        W = np.random.randint(200, 1500)
        M = T + W

        a = compute_a_s2(M, beta1=beta1, beta2=beta2)
        a_noisy = np.random.normal(a, 0.5)
        while a_noisy <= 0:
            a_noisy = np.random.normal(a, 0.5)

        trials_per_condition = trials_per_sub // 2

        for condition_key in range(2):
            label = 'self' if condition_key == 1 else 'stranger'
            trial_count = 0
            while trial_count < trials_per_condition:
                v_base = compute_v_s2(T, P, condition_key, alaph1=alaph1, alaph2=alaph2, gamma=gamma)
                v = np.random.normal(v_base, 1.0)
                evidence = a_noisy / 2.0
                delta_t = 0.001
                max_time = (W + T - t0_val) * 0.001 + 0.001
                decision_time = 0
                time = 0.0

                while evidence > 0 and evidence < a_noisy:
                    instantaneous_evidence = v * delta_t
                    noise = norm.rvs(loc=0, scale=1) * np.sqrt(delta_t)
                    evidence += instantaneous_evidence + noise
                    time += delta_t
                    if evidence >= a_noisy:
                        decision_time = time + t0_val
                        response = 1
                        break
                    elif evidence <= 0:
                        decision_time = time + t0_val
                        response = 2
                        break

                if decision_time < max_time:
                    rows.append({
                        'subjectID': subject + 1,
                        'trialID': trial_count + 1,
                        'T': T, 'P': P, 'W': W,
                        'Label': label,
                        'v': v, 'a': a_noisy,
                        'RT': decision_time,
                        'response': response,
                    })
                    trial_count += 1

    df = pd.DataFrame(rows)
    return df

# =============================================================================
# 模型版本 2: GP + DDM v2.4 (Generate_Data_v2.4_runner.py)
# =============================================================================

def generate_gp_ddm_v24(n_subjects=50, trials_per_sub=60, w_gp=0.5,
                        v_noise=1.0, a_noise=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    gen = HybridDDMParameterGenerator(w=w_gp)
    X_train = np.random.uniform(-1, 1, size=(50, 3))
    Y_v = np.sin(X_train[:, 0]) * 0.5 + 0.1 * np.random.randn(50)
    Y_a = 1.5 + 0.3 * np.cos(X_train[:, 1]) + 0.05 * np.random.randn(50)
    gen.fit_gp(X_train, Y_v, Y_a)

    rows = []
    for subj in range(1, n_subjects + 1):
        T = np.random.randint(10, 600)
        P = np.random.randint(0, 150)
        W = np.random.randint(200, 1500)
        M = T + W
        a_s2 = compute_a_s2(M)
        while a_s2 <= 0:
            T = np.random.randint(10, 600)
            W = np.random.randint(200, 1500)
            M = T + W
            a_s2 = compute_a_s2(M)

        trials_per_condition = trials_per_sub // 2
        if trials_per_sub % 2 != 0:
            trials_per_condition += 1

        for condition_key in range(2):
            label = 'self' if condition_key == 1 else 'stranger'
            trial_count = 0
            while trial_count < trials_per_condition:
                v_s2 = compute_v_s2(T, P, condition_key)
                Pn, Tn, Wn = normalize_PTW_to_unit(P, T, W)
                X = np.array([[Pn, Tn, Wn]])
                v_gp, a_gp, t0_arr, _ = gen.predict_params(X)

                v_mix = w_gp * v_gp[0] + (1 - w_gp) * v_s2
                a_mix = w_gp * a_gp[0] + (1 - w_gp) * a_s2
                v_final = np.random.normal(v_mix, v_noise)
                a_final = np.random.normal(a_mix, a_noise)
                if a_final <= 0:
                    a_final = max(0.1, a_mix)
                z_final = a_final / 2.0

                RT, resp = simulate_ddm_euler(v_final, a_final, z_final, 0.2)
                if np.isnan(RT):
                    RT = (T + W) * 0.001
                    response = 0
                else:
                    response = 1 if resp == 1 else 2

                rows.append({
                    'subject': subj, 'trial': trial_count + 1,
                    'P': P, 'T': T, 'W': W, 'M': M, 'label': label,
                    'v': v_final, 'a': a_final, 't0': 0.2, 'z': z_final,
                    'RT': RT, 'response': response,
                })
                trial_count += 1

    df = pd.DataFrame(rows)
    return df, gen

# =============================================================================
# 模型版本 3: GP + DDM v2.4.5 最新版 (Generate_Data_v2.4.5.ipynb)
# =============================================================================

def generate_gp_ddm_v245(n_subjects=50, trials_per_sub=60, w_gp=0.5,
                         v_noise=1.0, a_noise=0.5, lapse_max=0.35, seed=None):
    if seed is not None:
        np.random.seed(seed)

    gen = HybridDDMParameterGenerator(w=w_gp)
    X_train = np.random.uniform(-1, 1, size=(50, 3))
    Y_v = np.sin(X_train[:, 0]) * 0.5 + 0.1 * np.random.randn(50)
    Y_a = 1.5 + 0.3 * np.cos(X_train[:, 1]) + 0.05 * np.random.randn(50)
    gen.fit_gp(X_train, Y_v, Y_a)

    rows = []
    for subj in range(1, n_subjects + 1):
        T = np.random.randint(10, 600)
        P = np.random.randint(0, 150)
        W = np.random.randint(200, 1500)
        M = T + W
        a_s2 = compute_a_s2(M)

        trials_per_condition = trials_per_sub // 2
        if trials_per_sub % 2 != 0:
            trials_per_condition += 1

        for condition_key in range(2):
            label = 'self' if condition_key == 1 else 'stranger'
            for trial_idx in range(trials_per_condition):
                v_s2 = compute_v_s2(T, P, condition_key)
                Pn, Tn, Wn = normalize_PTW_to_unit(P, T, W)
                X = np.array([[Pn, Tn, Wn]])
                v_gp, a_gp, t0_arr, _ = gen.predict_params(X)

                v_mix = w_gp * v_gp[0] + (1 - w_gp) * v_s2
                a_mix = w_gp * a_gp[0] + (1 - w_gp) * a_s2
                t0 = float(t0_arr[0])

                v_final = np.random.normal(v_mix, v_noise)
                a_final = sample_a_positive(a_mix, a_noise, a_floor=0.03)
                z_final = a_final / 2.0
                deadline_s = W / 1000.0

                p_lapse = compute_lapse_omission_prob(T, W, lapse_max=lapse_max)
                if np.random.rand() < p_lapse:
                    RT = np.nan
                    response = 0
                    omission = 1
                    omission_source = 'lapse'
                else:
                    RT, response, omission, omission_source = simulate_ddm_with_deadline(
                        v_final, a_final, z_final, t0, deadline_s
                    )

                rows.append({
                    'subject': subj, 'trial': trial_idx + 1,
                    'P': P, 'T': T, 'W': W, 'M': M, 'label': label,
                    'v': v_final, 'a': a_final, 't0': t0, 'z': z_final,
                    'RT': RT, 'response': response,
                    'correct': 1 if response == 1 else 0,
                    'responded': int(omission == 0),
                    'omission': omission,
                    'omission_source': omission_source,
                    'deadline_s': deadline_s,
                    'p_lapse': p_lapse,
                    'v_s2': v_s2, 'a_s2': a_s2,
                    'v_gp_raw': v_gp[0], 'a_gp_raw': a_gp[0],
                    'v_mix': v_mix, 'a_mix': a_mix,
                })

    df = pd.DataFrame(rows)
    return df, gen

# =============================================================================
# 模型版本 4: GP + DDM v2.5 (Generate_Data_v2.5_runner.py)
# =============================================================================

EXPERIMENT_CONDITIONS = [
    {'P': 0, 'T': 30, 'W': 300, 'group': 1},
    {'P': 0, 'T': 30, 'W': 600, 'group': 2},
    {'P': 120, 'T': 30, 'W': 600, 'group': 3},
    {'P': 120, 'T': 80, 'W': 600, 'group': 4},
    {'P': 8, 'T': 100, 'W': 1100, 'group': 5},
    {'P': 120, 'T': 500, 'W': 1500, 'group': 6},
]

def generate_gp_ddm_v25(n_subjects_per_group=8, trials_per_sub=60,
                        base_v=1.5, alaph_self=0.6, alaph_stranger=0.9,
                        v_noise=0.3, a_noise=0.25, w_gp=0.5,
                        ptw_power=2.0, seed=42):
    np.random.seed(seed)

    gen = HybridDDMParameterGenerator(w=w_gp)
    X_train = np.random.uniform(-1, 1, size=(50, 3))
    Y_v = np.sin(X_train[:, 0]) * 0.5 + 0.1 * np.random.randn(50)
    Y_a = 1.5 + 0.3 * np.cos(X_train[:, 1]) + 0.05 * np.random.randn(50)
    gen.fit_gp(X_train, Y_v, Y_a)

    rows = []
    subject_id = 1

    for cond in EXPERIMENT_CONDITIONS:
        P, T, W, group = cond['P'], cond['T'], cond['W'], cond['group']
        ptw_factor = (P / 120.0 * T / 500.0 * W / 1500.0 + 0.1) ** ptw_power

        for subj in range(n_subjects_per_group):
            trials_per_condition = trials_per_sub // 2

            for condition_key in range(2):
                label = 'self' if condition_key == 1 else 'stranger'

                for trial in range(trials_per_condition):
                    T_0 = 100
                    k_T = 0.01
                    v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
                    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=0.2, P0=32)
                    v_base = v_T * v_P * base_v

                    if condition_key == 1:
                        v_s2 = v_base * alaph_self * ptw_factor
                    else:
                        v_s2 = v_base * alaph_stranger

                    Pn, Tn, Wn = normalize_PTW_to_unit(P, T, W)
                    X = np.array([[Pn, Tn, Wn]])
                    v_gp, a_gp, _, _ = gen.predict_params(X)

                    v_mix = w_gp * v_gp[0] + (1 - w_gp) * v_s2
                    a_mix = w_gp * a_gp[0] + (1 - w_gp) * 1.5
                    v_final = np.random.normal(v_mix, v_noise)
                    a_final = np.random.normal(a_mix, a_noise)
                    if a_final <= 0:
                        a_final = max(0.1, a_mix)
                    z_final = a_final / 2.0

                    RT, resp = simulate_ddm_euler(v_final, a_final, z_final, 0.2)
                    if np.isnan(RT):
                        RT = (T + W) * 0.001
                        response = 0
                    else:
                        response = 1 if resp == 1 else 2

                    rows.append({
                        'subject': subject_id, 'group': group,
                        'trial': trial + 1,
                        'P': P, 'T': T, 'W': W, 'M': T + W, 'label': label,
                        'v': v_final, 'a': a_final, 't0': 0.2, 'z': z_final,
                        'RT': RT, 'response': response,
                    })
            subject_id += 1

    df = pd.DataFrame(rows)
    return df, gen

# =============================================================================
# 统计摘要 & 图表生成
# =============================================================================

def compute_summary(df, model_name):
    df_valid = df[df['RT'].notna() & (df['RT'] > 0)].copy()
    df_valid['RT_ms'] = df_valid['RT'] * 1000

    # 统一列名: Label -> label, subjectID -> subject (兼容 sigmoid_ddm)
    if 'Label' in df_valid.columns and 'label' not in df_valid.columns:
        df_valid['label'] = df_valid['Label']
    if 'subjectID' in df_valid.columns and 'subject' not in df_valid.columns:
        df_valid['subject'] = df_valid['subjectID']

    total_trials = len(df)
    valid_trials = len(df_valid)
    respond_rate = valid_trials / max(total_trials, 1)

    self_data = df_valid[df_valid['label'] == 'self']
    stranger_data = df_valid[df_valid['label'] == 'stranger']
    self_rt = self_data['RT'].mean() * 1000 if len(self_data) > 0 else np.nan
    stranger_rt = stranger_data['RT'].mean() * 1000 if len(stranger_data) > 0 else np.nan

    subj_means = df_valid.groupby(['subject', 'label'])['RT_ms'].mean().reset_index()
    pivot = subj_means.pivot(index='subject', columns='label', values='RT_ms').dropna()
    if len(pivot) > 1:
        pivot['SPE_ms'] = pivot.get('self', 0) - pivot.get('stranger', 0)
        mean_spe = pivot['SPE_ms'].mean()
        sd_spe = pivot['SPE_ms'].std(ddof=1)
        cohens_d = mean_spe / sd_spe if sd_spe > 0 else np.nan
    else:
        mean_spe = np.nan
        cohens_d = np.nan

    acc = (df_valid['response'] == 1).mean() if 'response' in df_valid.columns else np.nan

    n_subjects = df_valid['subject'].nunique() if 'subject' in df_valid.columns else df['subjectID'].nunique()
    return {
        'model': model_name,
        'total_trials': total_trials,
        'valid_trials': valid_trials,
        'respond_rate': round(respond_rate, 4),
        'rt_mean_ms': round(df_valid['RT_ms'].mean(), 1),
        'rt_median_ms': round(df_valid['RT_ms'].median(), 1),
        'self_rt_ms': round(self_rt, 1),
        'stranger_rt_ms': round(stranger_rt, 1),
        'SPE_ms': round(mean_spe, 1) if not np.isnan(mean_spe) else None,
        'cohens_d': round(cohens_d, 3) if not np.isnan(cohens_d) else None,
        'accuracy': round(acc, 4) if not np.isnan(acc) else None,
        'n_subjects': n_subjects,
    }

def generate_figures(df, fig_dir, model_name):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    df_valid = df[df['RT'].notna() & (df['RT'] > 0)].copy()
    df_valid['RT_ms'] = df_valid['RT'] * 1000
    if 'Label' in df_valid.columns and 'label' not in df_valid.columns:
        df_valid['label'] = df_valid['Label']
    if 'subjectID' in df_valid.columns and 'subject' not in df_valid.columns:
        df_valid['subject'] = df_valid['subjectID']

    # 1. RT 分布
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_valid['RT_ms'], bins=60, edgecolor='black', alpha=0.7)
    ax.set_title(f'RT Distribution - {model_name}')
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Count')
    path = fig_dir / f'RT_distribution_{model_name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    figures.append(str(path))

    # 2. Self vs Stranger RT
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in ['self', 'stranger']:
        subset = df_valid[df_valid['label'] == label]['RT_ms']
        if len(subset) > 0:
            ax.hist(subset, bins=40, alpha=0.5, label=label)
    ax.legend()
    ax.set_title(f'RT by Label - {model_name}')
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Count')
    path = fig_dir / f'RT_by_label_{model_name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    figures.append(str(path))

    # 3. SPE 分布
    subj_means = df_valid.groupby(['subject', 'label'])['RT_ms'].mean().reset_index()
    pivot = subj_means.pivot(index='subject', columns='label', values='RT_ms').dropna()
    if len(pivot) > 0 and 'self' in pivot.columns and 'stranger' in pivot.columns:
        pivot['SPE_ms'] = pivot['self'] - pivot['stranger']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pivot['SPE_ms'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'SPE Distribution (Self - Stranger) - {model_name}')
        ax.set_xlabel('SPE (ms)')
        ax.set_ylabel('Subjects')
        path = fig_dir / f'SPE_dist_{model_name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(path))

    # 4. P/T/W 对 RT 的趋势
    if len(df_valid) > 100:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, var in enumerate(['P', 'T', 'W']):
            axes[i].scatter(df_valid[var], df_valid['RT_ms'], alpha=0.1, s=2)
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('RT (ms)')
            axes[i].set_title(f'{var} vs RT')
        fig.suptitle(f'Design Variables vs RT - {model_name}', fontsize=13)
        plt.tight_layout()
        path = fig_dir / f'design_vs_RT_{model_name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(path))

    return figures

# =============================================================================
# 统一运行入口
# =============================================================================

MODEL_VERSIONS = {
    'sigmoid_ddm': {
        'name': 'Sigmoid + DDM (基础版)',
        'source': 'S2 gen_data_jh.ipynb',
        'description': '纯 Sigmoid S2 机制 + Euler DDM 仿真，最基础的版本',
    },
    'gp_ddm_v24': {
        'name': 'GP + DDM v2.4 (稳定版)',
        'source': 'Generate_Data_v2.4_runner.py',
        'description': 'Sigmoid S2 + GP 混合 v/a 参数 + Euler DDM',
    },
    'gp_ddm_v245': {
        'name': 'GP + DDM v2.4.5 (最新版)',
        'source': 'Generate_Data_v2.4.5.ipynb',
        'description': 'Sigmoid S2 + GP 混合 + NA/omission机制 + Deadline DDM + 正a重采样',
    },
    'gp_ddm_v25': {
        'name': 'GP + DDM v2.5 (实验组版)',
        'source': 'Generate_Data_v2.5_runner.py',
        'description': '基于6组真实实验条件的 GP + DDM，支持 PTW 因子调控',
    },
}

def run_model(model_key, params, data_dir, fig_dir):
    if model_key not in MODEL_VERSIONS:
        raise ValueError(f'未知模型: {model_key}。可用: {list(MODEL_VERSIONS.keys())}')

    n_subjects = params.get('n_subjects', 50)
    trials_per_sub = params.get('trials_per_sub', 60)
    seed = params.get('seed', 42)

    if model_key == 'sigmoid_ddm':
        df = generate_sigmoid_ddm(
            n_subjects=n_subjects, trials_per_sub=trials_per_sub, seed=seed,
            alaph1=params.get('alaph1', 1.5),
            alaph2=params.get('alaph2', -0.4),
            beta1=params.get('beta1', 0.2),
            beta2=params.get('beta2', 0.0),
            gamma=params.get('gamma', 0.2),
            t0_val=params.get('t0_val', 0.2),
        )
        gen = None

    elif model_key == 'gp_ddm_v24':
        df, gen = generate_gp_ddm_v24(
            n_subjects=n_subjects, trials_per_sub=trials_per_sub,
            w_gp=params.get('w_gp', 0.5),
            v_noise=params.get('v_noise', 1.0),
            a_noise=params.get('a_noise', 0.5),
            seed=seed,
        )

    elif model_key == 'gp_ddm_v245':
        df, gen = generate_gp_ddm_v245(
            n_subjects=n_subjects, trials_per_sub=trials_per_sub,
            w_gp=params.get('w_gp', 0.5),
            v_noise=params.get('v_noise', 1.0),
            a_noise=params.get('a_noise', 0.5),
            lapse_max=params.get('lapse_max', 0.35),
            seed=seed,
        )

    elif model_key == 'gp_ddm_v25':
        df, gen = generate_gp_ddm_v25(
            n_subjects_per_group=params.get('n_subjects_per_group', 8),
            trials_per_sub=trials_per_sub,
            base_v=params.get('base_v', 1.5),
            alaph_self=params.get('alaph_self', 0.6),
            alaph_stranger=params.get('alaph_stranger', 0.9),
            v_noise=params.get('v_noise', 0.3),
            a_noise=params.get('a_noise', 0.25),
            w_gp=params.get('w_gp', 0.5),
            ptw_power=params.get('ptw_power', 2.0),
            seed=seed,
        )

    # 保存 CSV
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f'gp_ddm_{model_key}.csv'
    df.to_csv(csv_path, index=False)

    # 计算摘要
    summary = compute_summary(df, model_key)

    # 生成图表
    figures = generate_figures(df, fig_dir, model_key)

    return {
        'csv_path': str(csv_path),
        'summary': summary,
        'figures': figures,
        'n_rows': len(df),
    }

# =============================================================================
# 参数扫描：计算 SPE / v / a / t0 / z / ACC 沿T/P/W的曲线
# =============================================================================

def _analytical_rt(v, a, t0=0.2):
    """DDM 一阶近似：mean RT ≈ t0 + a/(2*|v|)"""
    v_abs = np.abs(v)
    v_abs = np.where(v_abs < 1e-6, 1e-6, v_abs)
    return t0 + a / (2.0 * v_abs)

def _analytical_acc(v, a):
    """DDM 解析正确率：P(upper) = 1/(1+exp(-2*v*a)) with z=a/2"""
    return 1.0 / (1.0 + np.exp(-2.0 * v * a))

def _init_gp_generator(w_gp=0.5, seed=42):
    rng = np.random.RandomState(seed)
    gen = HybridDDMParameterGenerator(w=w_gp)
    X_train = rng.uniform(-1, 1, size=(50, 3))
    Y_v = np.sin(X_train[:, 0]) * 0.5 + 0.1 * rng.randn(50)
    Y_a = 1.5 + 0.3 * np.cos(X_train[:, 1]) + 0.05 * rng.randn(50)
    gen.fit_gp(X_train, Y_v, Y_a)
    return gen

def compute_sweep(model_key, sweep_var, fixed_params, w_gp=0.5, seed=42):
    """
    计算单个模型的参数扫描曲线（确定性，不含噪声）。

    参数:
        model_key: 模型版本 key
        sweep_var: 'T', 'P', 或 'W'
        fixed_params: {P: val, T: val, W: val}，sweep_var 的值会被覆盖
        w_gp: GP 权重
        seed: 随机种子

    返回:
        dict: {
            'x_values': [...],
            'v_self': [...], 'v_stranger': [...],
            'a': [...], 't0': [...], 'z': [...],
            'rt_self': [...], 'rt_stranger': [...],
            'spe_rt': [...],
            'acc_self': [...], 'acc_stranger': [...],
            'spe_acc': [...],
            'label': str,
        }
    """
    if model_key not in MODEL_VERSIONS:
        raise ValueError(f'未知模型: {model_key}')

    P_fix = fixed_params.get('P', 64)
    T_fix = fixed_params.get('T', 200)
    W_fix = fixed_params.get('W', 600)

    ranges = {
        'P': (0, 150, 61),
        'T': (10, 600, 60),
        'W': (200, 1500, 66),
    }
    v_min, v_max, n_pts = ranges[sweep_var]
    x_values = np.linspace(v_min, v_max, n_pts)

    # 初始化 GP（仅 GP 模型需要）
    gen = _init_gp_generator(w_gp=w_gp, seed=seed) if 'gp_ddm' in model_key else None

    n = len(x_values)
    v_self = np.zeros(n); v_stranger = np.zeros(n)
    a_arr = np.zeros(n); t0_arr = np.full(n, 0.2); z_arr = np.zeros(n)

    for i, x_val in enumerate(x_values):
        if sweep_var == 'T':
            P, T, W = P_fix, x_val, W_fix
        elif sweep_var == 'P':
            P, T, W = x_val, T_fix, W_fix
        else:  # W
            P, T, W = P_fix, T_fix, x_val

        M = T + W

        if model_key == 'sigmoid_ddm':
            v_s2_self = compute_v_s2(T, P, 1)
            v_s2_str = compute_v_s2(T, P, 0)
            a_s2 = compute_a_s2(M)
            v_self[i] = v_s2_self
            v_stranger[i] = v_s2_str
            a_arr[i] = a_s2
            t0_arr[i] = 0.2
            z_arr[i] = a_s2 / 2.0

        elif model_key in ('gp_ddm_v24', 'gp_ddm_v245'):
            v_s2_self = compute_v_s2(T, P, 1)
            v_s2_str = compute_v_s2(T, P, 0)
            a_s2 = compute_a_s2(M)
            Pn, Tn, Wn = normalize_PTW_to_unit(P, T, W)
            X = np.array([[Pn, Tn, Wn]])
            v_gp, a_gp, _, _ = gen.predict_params(X)
            v_mix_self = w_gp * v_gp[0] + (1 - w_gp) * v_s2_self
            v_mix_str = w_gp * v_gp[0] + (1 - w_gp) * v_s2_str
            a_mix = w_gp * a_gp[0] + (1 - w_gp) * a_s2
            v_self[i] = v_mix_self
            v_stranger[i] = v_mix_str
            a_arr[i] = max(a_mix, 0.03)
            t0_arr[i] = 0.2
            z_arr[i] = a_arr[i] / 2.0

        elif model_key == 'gp_ddm_v25':
            T_0 = 100; k_T = 0.01
            v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
            v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=0.2, P0=32)
            v_base = v_T * v_P * 1.5
            ptw_factor = (P / 120.0 * T / 500.0 * W / 1500.0 + 0.1) ** 2.0
            v_s2_self = v_base * 0.6 * ptw_factor
            v_s2_str = v_base * 0.9
            Pn, Tn, Wn = normalize_PTW_to_unit(P, T, W)
            X = np.array([[Pn, Tn, Wn]])
            v_gp, a_gp, _, _ = gen.predict_params(X)
            v_self[i] = w_gp * v_gp[0] + (1 - w_gp) * v_s2_self
            v_stranger[i] = w_gp * v_gp[0] + (1 - w_gp) * v_s2_str
            a_mix = w_gp * a_gp[0] + (1 - w_gp) * 1.5
            a_arr[i] = max(a_mix, 0.1)
            t0_arr[i] = 0.2
            z_arr[i] = a_arr[i] / 2.0

    rt_self = _analytical_rt(v_self, a_arr, t0_arr)
    rt_stranger = _analytical_rt(v_stranger, a_arr, t0_arr)
    spe_rt = (rt_self - rt_stranger) * 1000.0
    acc_self = _analytical_acc(v_self, a_arr)
    acc_stranger = _analytical_acc(v_stranger, a_arr)
    spe_acc = acc_self - acc_stranger

    label = f'{MODEL_VERSIONS[model_key]["name"]} (P={P_fix}, T={T_fix}, W={W_fix})'

    return {
        'x_values': x_values.round(1).tolist(),
        'sweep_var': sweep_var,
        'fixed': {'P': P_fix, 'T': T_fix, 'W': W_fix},
        'model': model_key,
        'label': label,
        'v_self': v_self.round(4).tolist(),
        'v_stranger': v_stranger.round(4).tolist(),
        'a': a_arr.round(4).tolist(),
        't0': t0_arr.round(4).tolist(),
        'z': z_arr.round(4).tolist(),
        'rt_self': rt_self.round(4).tolist(),
        'rt_stranger': rt_stranger.round(4).tolist(),
        'spe_rt': spe_rt.round(2).tolist(),
        'acc_self': acc_self.round(4).tolist(),
        'acc_stranger': acc_stranger.round(4).tolist(),
        'spe_acc': spe_acc.round(4).tolist(),
    }
