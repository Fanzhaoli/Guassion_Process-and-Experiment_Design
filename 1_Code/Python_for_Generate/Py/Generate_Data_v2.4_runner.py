import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.special import expit as sigmoid

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data"
FIG_DIR = BASE_DIR / "3_Figures" / "Generate_Data_v2.4_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Hybrid GP generator
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
        z = 2 / a
        return v, a, t0, z

# S2 helpers
def k_P(P, k_min = 0.01, k_max = 0.15, gamma = 0.1, P0 = 32):
    return k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))

def v_P_Function(P, P1 = 4, k_min = 0.01, k_max = 0.15, gamma = 0.1, P0 = 32):
    k = k_P(P, k_min, k_max, gamma, P0)
    return 1 / (1 + np.exp(-k * (P - P1)))

def compute_v_s2(T, P, condition_key, alaph1=1.5, alaph2=-0.4, gamma=0.2):
    T_0 = 100
    k_T = 0.01
    v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
    v_P = v_P_Function(P = P, P1 = 4, k_min = 0.1, k_max = 0.05, gamma = gamma, P0 = 32)
    v_0 = v_T * v_P * 3
    if condition_key == 1:
        v_1 = v_0 * (1 + alaph1)
    else:
        v_1 = v_0 * (1 + alaph2)
    return v_1

def compute_a_s2(M, beta1=0.2, beta2=0, k=0.01, M_0=600):
    a_0 = 1 / (1 + np.exp(-k * (M - M_0))) * 3
    if M > 600:
        a_1 = a_0 * (1 + beta1)
    else:
        a_1 = a_0 * (1 + beta2)
    return a_1

def normalize_PTW_to_unit(P, T, W):
    P_norm = (P - 75.0) / 75.0
    T_norm = (T - 305.0) / 295.0
    W_norm = (W - 850.0) / 650.0
    return P_norm, T_norm, W_norm

# DDM Euler
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

# Generator
def generate_dataset_s2(n_subjects=50, trials_per_sub=60, w_gp=0.5, v_noise=1.0, a_noise=0.5, save_path=None, seed=None):
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
                v_gp, a_gp, t0_arr, z_arr = gen.predict_params(X)
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
                    'subject': subj,
                    'trial': trial_count+1,
                    'P': P, 'T': T, 'W': W, 'M': M, 'label': label,
                    'v': v_final, 'a': a_final, 't0': 0.2, 'z': z_final,
                    'RT': RT, 'response': response
                })
                trial_count += 1
    df = pd.DataFrame(rows)
    if save_path is None:
        save_path = OUT_DIR / 'gp_ddm_v2.4_small.csv'
    else:
        save_path = Path(save_path)
    df.to_csv(save_path, index=False)
    return df, gen

# Small run
if __name__ == '__main__':
    df, gen = generate_dataset_s2(n_subjects=50, trials_per_sub=60, w_gp=0.5, seed=42)
    out_csv = OUT_DIR / 'gp_ddm_v2.4_small.csv'
    df.to_csv(out_csv, index=False)
    print('Saved CSV to', out_csv)

    # Basic stats
    df = df.copy()
    df['RT_ms'] = df['RT'] * 1000
    print('\nOverall trials:', len(df))
    print('RT mean (s):', df['RT'].mean())
    print('RT median (s):', df['RT'].median())
    pct_0_2s = (df['RT'] <= 2.0).mean()
    print('Percent RT <= 2s:', pct_0_2s)

    # Accuracy
    acc_upper = (df['response'] == 1).mean()
    acc_lower = (df['response'] == 2).mean()
    print('Proportion upper (1):', acc_upper)
    print('Proportion lower (2):', acc_lower)

    # Condition differences
    mean_by_label = df.groupby('label')['RT'].mean()
    print('\nMean RT by label (s):')
    print(mean_by_label)

    # Subject-level SPE
    subj_means = df.groupby(['subject','label'])['RT_ms'].mean().reset_index().pivot(index='subject',columns='label',values='RT_ms').dropna()
    subj_means['SPE_ms'] = subj_means['self'] - subj_means['stranger']
    mean_spe = subj_means['SPE_ms'].mean()
    sd_spe = subj_means['SPE_ms'].std(ddof=1)
    cohens_d = mean_spe / sd_spe if sd_spe>0 else np.nan
    print('\nMean SPE (ms):', mean_spe)
    print('SD SPE (ms):', sd_spe)
    print("Cohen's d (paired):", cohens_d)

    # Save simple plots
    plt.figure()
    plt.hist(df['RT_ms'], bins=60)
    plt.title('RT distribution (ms)')
    plt.savefig(FIG_DIR / 'RT_distribution_v2.4.png', dpi=200)
    plt.close()

    plt.figure()
    for label in ['self','stranger']:
        subset = df[df['label']==label]['RT_ms']
        plt.hist(subset, bins=40, alpha=0.5, label=label)
    plt.legend()
    plt.title('RT by label (ms)')
    plt.savefig(FIG_DIR / 'RT_by_label_v2.4.png', dpi=200)
    plt.close()

    plt.figure()
    plt.hist(subj_means['SPE_ms'].dropna(), bins=30)
    plt.title('SPE distribution (ms)')
    plt.savefig(FIG_DIR / 'SPE_dist_v2.4.png', dpi=200)
    plt.close()

    # GP uncertainty diagnostic: predictive std on subject anchors
    anchors = np.array([normalize_PTW_to_unit(r['P'], r['T'], r['W']) for _,r in df.groupby('subject').first().iterrows()])
    _, std_v = gen.gp_v.predict(anchors, return_std=True)
    print('\nGP predictive std (v) - mean:', np.mean(std_v))
    print('GP predictive std (v) - std:', np.std(std_v))

    print('\nFigures saved to', FIG_DIR)
