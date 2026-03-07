import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.special import expit as sigmoid
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def v_P_Function(P, P1=4, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    k = k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))
    return 1 / (1 + np.exp(-k * (P - P1)))

def normalize_PTW_to_unit(P, T, W):
    P_norm = (P - 75.0) / 75.0
    T_norm = (T - 305.0) / 295.0
    W_norm = (W - 850.0) / 650.0
    return P_norm, T_norm, W_norm

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

EXPERIMENT_CONDITIONS = [
    {'P': 0, 'T': 30, 'W': 300, 'group': 1},
    {'P': 0, 'T': 30, 'W': 600, 'group': 2},
    {'P': 120, 'T': 30, 'W': 600, 'group': 3},
    {'P': 120, 'T': 80, 'W': 600, 'group': 4},
    {'P': 8, 'T': 100, 'W': 1100, 'group': 5},
    {'P': 120, 'T': 500, 'W': 1500, 'group': 6},
]

TARGET_SPE = {1: 2.0, 2: 3.8, 3: -0.4, 4: -1.2, 5: -34.3, 6: -90.3}

def generate_dataset_v2_5(n_subjects_per_group=8, trials_per_sub=60, 
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
                        'subject': subject_id,
                        'group': group,
                        'trial': trial + 1,
                        'P': P, 'T': T, 'W': W, 'M': T+W, 'label': label,
                        'v': v_final, 'a': a_final, 't0': 0.2, 'z': z_final,
                        'RT': RT, 'response': response
                    })
            subject_id += 1
    
    df = pd.DataFrame(rows)
    return df, gen

if __name__ == '__main__':
    print("=" * 70)
    print("V2.5 数据生成")
    print("=" * 70)
    
    df_final, _ = generate_dataset_v2_5(
        n_subjects_per_group=8,
        trials_per_sub=60,
        base_v=1.5,
        alaph_self=1.2,
        alaph_stranger=0.9,
        v_noise=0.3,
        a_noise=0.25,
        w_gp=0.5,
        ptw_power=1.2,
        seed=123
    )
    
    out_csv = OUT_DIR / 'gp_ddm_v2.5.csv'
    df_final.to_csv(out_csv, index=False)
    print(f"数据已保存到: {out_csv}")
    
    df_valid = df_final[df_final['RT'].notna() & (df_final['RT'] > 0)]
    
    print("\n" + "=" * 70)
    print("V2.5 生成数据统计")
    print("=" * 70)
    print(f"总试次数: {len(df_final)}")
    print(f"被试数: {df_final['subject'].nunique()}")
    print(f"RT均值: {df_valid['RT'].mean()*1000:.1f}ms")
    print(f"正确率: {(df_valid['response']==1).mean()*100:.1f}%")
    
    for label in ['self', 'stranger']:
        subset = df_valid[df_valid['label'] == label]
        print(f"{label}: RT={subset['RT'].mean()*1000:.1f}ms, Acc={(subset['response']==1).mean()*100:.1f}%")
    
    print("\n各组SPE对比:")
    for group in sorted(df_final['group'].unique()):
        gdata = df_valid[df_valid['group'] == group]
        g_self = gdata[gdata['label']=='self'].groupby('subject')['RT'].mean()
        g_stranger = gdata[gdata['label']=='stranger'].groupby('subject')['RT'].mean()
        g_common = g_self.index.intersection(g_stranger.index)
        if len(g_common) > 0:
            spe = (g_self[g_common] - g_stranger[g_common]).mean() * 1000
            target = TARGET_SPE[group]
            diff = spe - target
            cond = EXPERIMENT_CONDITIONS[group-1]
            print(f"  Group{group} (P={cond['P']:3}, T={cond['T']}, W={cond['W']}): SPE={spe:6.1f}ms (目标: {target:6.1f}ms, 差异: {diff:+6.1f}ms)")
