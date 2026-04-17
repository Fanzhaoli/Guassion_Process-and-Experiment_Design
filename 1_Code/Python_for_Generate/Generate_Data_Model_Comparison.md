# 生成模型对比：Generate_Data_v1 到 Generate_Data_v3

本文档总结了 `1_Code/Python_for_Generate` 目录中从 `Generate_Data_v1.ipynb` 到 `Generate_Data_v3.ipynb` 的生成模型演进。`Generate_Data_v2.3.ipynb` 为草稿版本，未纳入本次对比。

## 1. 概览

该系列模型旨在用 DDM（漂移扩散模型）模拟自我/他人匹配任务中的反应时和选择。核心演进方向为：

- 从手写 Sigmoid 基线与 GP anchor 映射，逐步过渡到
- Sigmoid 与 GP 混合预测，再到
- GPU 先验残差场与插值生成，更接近真实实验数据模拟。

## 2. 模型对比表格

| 版本 | 核心思路 | GP 角色 | DDM 参数处理 | 真实感 / 扩展 |
|---|---|---|---|---|
| `v1` | 手写 Sigmoid 基线 + anchor GP 映射 | 作为映射学习器，基于 anchor 样本拟合 `S=[P,T,W,cond] -> DDM` | `v`、`a`、`t0`、`z` 均由 Sigmoid 基线计算；self/stranger 用 `alaph_self=1.5` / `alaph_stranger=-0.4` 调节 | 早期原型；功能完整但更偏理论演示 |
| `v2` | `HybridDDMParameterGenerator`：Sigmoid 与 GP 直接混合 | 为 `v,a,t0,z` 建立 GP，最后与 Sigmoid 按权重 `w` 混合 | `v = w*sig_v + (1-w)*gp_v`; `a = w*sig_a + ...`; `t0`、`z` 也混合预测 | 明确混合架构，增加模型可塑性 |
| `v2.4` | 小规模运行版，保留 `v,a` 混合，简化 `t0,z` | 仅为 `v` 和 `a` 训练 GP；`t0` 固定，`z` 由 `2/a` 计算 | `v`/`a` 混合预测；`t0=0.2`；`z=2/a` | 用于快速检查与演示，GP 训练示例偏 synthetic |
| `v2.5` | 可配置实验模拟；backbone + GP 残差 + subject noise | GP 建模为残差层，`gp_weight_v` / `gp_weight_a` 控制残差强度 | 基线 `compute_v_backbone` / `compute_a_backbone`；加 subject-level `v_subject_shift`、`t0_subject`噪声 | 更真实，支持 invalid/timeout、subject 差异、参数配置 |
| `v3` | GP 先验残差场 + 插值；Sigmoid baseline 保留 | 用 GP 先验采样残差场，通过 anchor grid + `LinearNDInterpolator`/`NearestNDInterpolator` 插值 | `v = baseline + gp_resid`; `a = baseline + gp_resid`; `z=2/a`; `t0=0.2` | 更偏随机场/残差生成，保留原始 DDM 仿真 |

## 3. 关键参数对比

### 公共变量范围

- `P`: 0 ~ 150
- `T`: 10 ~ 600 ms
- `W`: 200 ~ 1500 ms
- `M = T + W`

### `v1` 重要参数

- `k_P(P)`: `0.01 + (0.15-0.01)/(1+exp(-0.1*(P-32)))`
- `v_P_Function`: `1/(1+exp(-k*(P-4)))`
- `compute_v_sigmoid`: `v_T * v_P * 3.0`，self/stranger 乘性调节
- `compute_a_sigmoid`: `M` 逻辑映射，`k=0.01`, `M0=600`
- `compute_t0_sigmoid`: `base=0.35`，减去与 `T,W` 线性相关项
- `compute_z_from_a`: `a/2 + noise`

### `v2` 重要参数

- `self.w`: Sigmoid/GP 混合权重
- `beta_v`, `beta_a`, `beta_t0`, `beta_z`: Sigmoid 线性参数向量
- GP kernel: `RBF + WhiteKernel`
- `t0 = 0.2 + 0.1*...`, `z = 0.5 + 0.2*...`

### `v2.4` 重要参数

- `compute_v_s2` 保留 `T0=100`、`k_T=0.01`
- `v_P_Function` 使用 `P1=4`, `k_min=0.1`, `k_max=0.05`
- `compute_a_s2` 使用 `M0=600`, `beta1=0.2`, `beta2=0`
- `t0=0.2`, `z=2/a`

### `v2.5` 重要参数

- `alpha_self=1.5`, `alpha_stranger=-0.4`
- `beta_high_M=0.2`, `beta_low_M=0.0`
- `gamma_P=0.1`, `P0=32`, `P1=4`
- `k_min=0.01`, `k_max=0.15`
- `T0_center=100`, `k_T=0.01`, `M0=600`, `k_M=0.01`
- `backbone_scale_v=3.0`, `backbone_scale_a=3.0`
- `t0=0.2`, `dt=0.001`, `A_CV=0.15`
- `subject_v_sd=0.20`, `subject_t0_sd=0.015`, `trial_v_sd=1.0`
- `gp_weight_v=0.35`, `gp_weight_a=0.35`

### `v3` 重要参数

- GP kernel for `v`: `ConstantKernel * RBF(length_scale=[40,150,300])`
- GP kernel for `a`: `ConstantKernel * RBF(length_scale=[60,200,400])`
- `gp_scale_v=0.8`, `gp_scale_a=0.5`
- `anchor_size=300`
- `max_time_s = max((T+W)/1000 + 0.5, 2.0)`

## 4. 各版本生成流程差异

- `v1`：先用 Sigmoid 骨架计算参数 → 生成 anchor 数据 → GP 拟合 → DDM 仿真。
- `v2`：直接用 `HybridDDMParameterGenerator` 预测参数，Sigmoid 和 GP 在同一函数里混合。
- `v2.4`：更像 demo/runner 版本，保留 `v,a` 混合；`t0` 和 `z` 简化处理。
- `v2.5`：引入配置、subject-level 噪声、GP 残差加权、invalid/timeout 记录，贴近真实实验模拟。
- `v3`：用 GP 先验采样残差场，按 trial 点插值；保留 Sigmoid baseline，侧重残差随机性。

## 5. 备注

- `Generate_Data_v2.3.ipynb` 为草稿，未纳入对比。
- 如果你希望，可继续追加 `v2.5` 与 `v3` 的具体函数调用链与生成脚本示意图。

---

*本文档由模型代码提取总结，适合快速理解从 v1 到 v3 的演进与参数差异。*
