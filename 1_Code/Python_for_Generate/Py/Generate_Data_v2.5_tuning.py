import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.special import expit as sigmoid

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "2_Data" / "Generate_Data"
FIG_DIR = BASE_DIR / "3_Figures" / "Generate_Data_v2.5_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

def k_P(P, a, t, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    return k_min + (k_max - k_min) / (1 + np.exp(-gamma * (P - P0)))

def v_P_Function(P, P1=4, k_min=0.01, k_max=0.15, gamma=0.1, P0=32):
    k = k_P(P, k_min, k_max, gamma, P0)
    return 1 / (1 + np.exp(-k * (P - P1)))

def compute_v_s2(T, P, W, condition_key, alaph1=0.15, alaph2=0.02, gamma=0.2, ptw_weight=0.3):
    T_0 = 100
    k_T = 0.01
    v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=gamma, P0=32)
    v_0 = v_T * v_P * 3
    
    ptw_factor = ((P + 1) / 121.0) * (T / 500.0) * (W / 1500.0)
    ptw_factor = ptw_factor ** ptw_weight
    
    if condition_key == 1:
        v_1 = v_0 * (1 + alaph1) * ptw_factor
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

def generate_dataset_v2_5(n_subjects_per_group=8, trials_per_sub=60, 
                          alaph1=0.15, alaph2=0.02, v_noise=0.4, a_noise=0.3, 
                          w_gp=0.5, ptw_weight=0.3, seed=42):
    if seed is not None:
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
        M = T + W
        
        for subj in range(n_subjects_per_group):
            a_s2 = compute_a_s2(M)
            
            trials_per_condition = trials_per_sub // 2
            
            for condition_key in range(2):
                label = 'self' if condition_key == 1 else 'stranger'
                
                for trial in range(trials_per_condition):
                    ptw_factor = ((P + 1) / 121.0) * (T / 500.0) * (W / 1500.0)
                    ptw_factor = ptw_factor ** ptw_weight
                    
                    T_0 = 100
                    k_T = 0.01
                    v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
                    v_P = v_P_Function(P=P, P1=4, k_min=0.1, k_max=0.05, gamma=0.2, P0=32)
                    v_0 = v_T * v_P * 3
                    
                    if condition_key == 1:
                        v_s2 = v_0 * (1 + alaph1) * ptw_factor
                    else:
                        v_s2 = v_0 * (1 + alaph2)
                    
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
                        'subject': subject_id,
                        'group': group,
                        'trial': trial + 1,
                        'P': P, 'T': T, 'W': W, 'M': M, 'label': label,
                        'v': v_final, 'a': a_final, 't0': 0.2, 'z': z_final,
                        'RT': RT, 'response': response
                    })
            subject_id += 1
    
    df = pd.DataFrame(rows)
    return df, gen

def evaluate_params(alaph1, alaph2, v_noise, a_noise, ptw_weight, n_subjects=6, trials=40, seed=42):
    df, _ = generate_dataset_v2_5(
        n_subjects_per_group=n_subjects,
        trials_per_sub=trials,
        alaph1=alaph1, alaph2=alaph2,
        v_noise=v_noise, a_noise=a_noise,
        ptw_weight=ptw_weight, seed=seed
    )
    
    df_valid = df[df['RT'].notna() & (df['RT'] > 0)]
    
    overall_rt = df_valid['RT'].mean() * 1000
    overall_acc = (df_valid['response'] == 1).mean() * 100
    
    self_rt = df_valid[df_valid['label'] == 'self'].groupby('subject')['RT'].mean()
    stranger_rt = df_valid[df_valid['label'] == 'stranger'].groupby('subject')['RT'].mean()
    common = self_rt.index.intersection(stranger_rt.index)
    overall_spe = (self_rt[common] - stranger_rt[common]).mean() * 1000
    
    group_spe = {}
    for group in range(1, 7):
        gdata = df_valid[df_valid['group'] == group]
        g_self = gdata[gdata['label'] == 'self'].groupby('subject')['RT'].mean()
        g_stranger = gdata[gdata['label'] == 'stranger'].groupby('subject')['RT'].mean()
        g_common = g_self.index.intersection(g_stranger.index)
        if len(g_common) > 0:
            group_spe[group] = (g_self[g_common] - g_stranger[g_common]).mean() * 1000
    
    return {
        'overall_rt': overall_rt,
        'overall_acc': overall_acc,
        'overall_spe': overall_spe,
        'group_spe': group_spe
    }

TARGET_SPE = {
    1: 2.0, 2: 3.8, 3: -0.4, 4: -1.2, 5: -34.3, 6: -90.3
}
TARGET_OVERALL_SPE = -14.2
TARGET_RT = 563
TARGET_ACC = 67.6

def compute_error(metrics):
    spe_error = 0
    for g in range(1, 7):
        if g in metrics['group_spe']:
            spe_error += (metrics['group_spe'][g] - TARGET_SPE[g]) ** 2
        else:
            spe_error += 10000
    
    overall_error = (
        0.3 * (metrics['overall_spe'] - TARGET_OVERALL_SPE) ** 2 +
        0.001 * (metrics['overall_rt'] - TARGET_RT) ** 2 +
        0.5 * (metrics['overall_acc'] - TARGET_ACC) ** 2
    )
    
    return spe_error + overall_error

def grid_search():
    print("=" * 60)
    print("开始网格搜索最优参数...")
    print("=" * 60)
    
    results = []
    
    alaph1_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    alaph2_values = [0.0, 0.02, 0.05]
    v_noise_values = [0.2, 0.3, 0.4, 0.5]
    ptw_weight_values = [0.2, 0.3, 0.4, 0.5]
    
    best_error = float('inf')
    best_params = None
    best_metrics = None
    
    total = len(alaph1_values) * len(alaph2_values) * len(v_noise_values) * len(ptw_weight_values)
    count = 0
    
    for alaph1 in alaph1_values:
        for alaph2 in alaph2_values:
            for v_noise in v_noise_values:
                for ptw_weight in ptw_weight_values:
                    count += 1
                    
                    a_noise = v_noise * 0.75
                    
                    try:
                        metrics = evaluate_params(
                            alaph1=alaph1, alaph2=alaph2,
                            v_noise=v_noise, a_noise=a_noise,
                            ptw_weight=ptw_weight,
                            n_subjects=6, trials=40, seed=42
                        )
                        
                        error = compute_error(metrics)
                        
                        results.append({
                            'alaph1': alaph1, 'alaph2': alaph2,
                            'v_noise': v_noise, 'a_noise': a_noise,
                            'ptw_weight': ptw_weight,
                            'error': error,
                            **metrics
                        })
                        
                        if error < best_error:
                            best_error = error
                            best_params = {
                                'alaph1': alaph1, 'alaph2': alaph2,
                                'v_noise': v_noise, 'a_noise': a_noise,
                                'ptw_weight': ptw_weight
                            }
                            best_metrics = metrics
                            
                            print(f"[{count}/{total}] 新最优! alaph1={alaph1}, alaph2={alaph2}, "
                                  f"v_noise={v_noise}, ptw_weight={ptw_weight}")
                            print(f"  SPE: {metrics['overall_spe']:.1f}ms, RT: {metrics['overall_rt']:.0f}ms, "
                                  f"Acc: {metrics['overall_acc']:.1f}%")
                            print(f"  Group SPE: {metrics['group_spe']}")
                            print()
                    
                    except Exception as e:
                        print(f"Error with params: alaph1={alaph1}, alaph2={alaph2}, v_noise={v_noise}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('error')
    results_df.to_csv(FIG_DIR / 'grid_search_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("网格搜索完成！")
    print("=" * 60)
    print(f"\n最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\n最优指标:")
    print(f"  整体SPE: {best_metrics['overall_spe']:.1f}ms (目标: {TARGET_OVERALL_SPE}ms)")
    print(f"  整体RT: {best_metrics['overall_rt']:.0f}ms (目标: {TARGET_RT}ms)")
    print(f"  整体正确率: {best_metrics['overall_acc']:.1f}% (目标: {TARGET_ACC}%)")
    print(f"\n各组SPE:")
    for g in range(1, 7):
        if g in best_metrics['group_spe']:
            print(f"  Group {g}: {best_metrics['group_spe'][g]:.1f}ms (目标: {TARGET_SPE[g]}ms)")
    
    return best_params, best_metrics, results_df

if __name__ == '__main__':
    best_params, best_metrics, results = grid_search()
    
    print("\n开始生成最终数据...")
    df_final, gen_final = generate_dataset_v2_5(
        n_subjects_per_group=8,
        trials_per_sub=60,
        **best_params,
        seed=123
    )
    
    out_csv = OUT_DIR / 'gp_ddm_v2.5.csv'
    df_final.to_csv(out_csv, index=False)
    print(f"数据已保存到: {out_csv}")
    
    df_valid = df_final[df_final['RT'].notna() & (df_final['RT'] > 0)]
    
    print("\n" + "=" * 60)
    print("V2.5 最终数据统计")
    print("=" * 60)
    print(f"总试次数: {len(df_final)}")
    print(f"被试数: {df_final['subject'].nunique()}")
    print(f"RT均值: {df_valid['RT'].mean()*1000:.1f}ms")
    print(f"RT中位数: {df_valid['RT'].median()*1000:.1f}ms")
    print(f"正确率: {(df_valid['response']==1).mean()*100:.1f}%")
    
    for label in ['self', 'stranger']:
        subset = df_valid[df_valid['label'] == label]
        print(f"{label}: RT均值={subset['RT'].mean()*1000:.1f}ms, "
              f"正确率={(subset['response']==1).mean()*100:.1f}%")
    
    print("\n各组SPE:")
    for group in sorted(df_final['group'].unique()):
        gdata = df_valid[df_valid['group'] == group]
        g_self = gdata[gdata['label']=='self'].groupby('subject')['RT'].mean()
        g_stranger = gdata[gdata['label']=='stranger'].groupby('subject')['RT'].mean()
        g_common = g_self.index.intersection(g_stranger.index)
        if len(g_common) > 0:
            spe = (g_self[g_common] - g_stranger[g_common]).mean() * 1000
            cond = EXPERIMENT_CONDITIONS[group-1]
            print(f"  Group {group} (P={cond['P']}, T={cond['T']}, W={cond['W']}): "
                  f"SPE={spe:.1f}ms (目标: {TARGET_SPE[group]}ms)")
