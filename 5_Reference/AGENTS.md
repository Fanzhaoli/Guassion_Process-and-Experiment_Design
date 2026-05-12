# AGENTS.md - AI 助手操作指南

## 一、项目简介

本研究以 Sui 等人（2012）提出的 Self-Matching Task 为核心范式，聚焦练习次数（P）、刺激呈现时间（T）与反应窗口（W）等实验设计变量如何系统性地调控自我优势效应（Self-Prioritization Effect, SPE）。

研究框架：
- **实验设计空间**: Ω = (P, T, W, M)
- **建模工具**: 漂移扩散模型（DDM）+ 高斯过程（GP）
- **核心目标**: 优化实验设计，使模型生成的模拟数据能够逼近真实人类被试的反应

---

## 二、文件夹结构

```
Guassion-Process-Experiment-Design/
├── 1_Code/                           # 代码目录
│   ├── Python_for_Generate/          # 【数据生成】Python代码
│   │   ├── Generate_Data_v1.ipynb   # 基线模型（Sigmoid + DDM）
│   │   ├── Generate_Data_v2.ipynb   # 进阶模型初始版本
│   │   ├── Generate_Data_v2.1-2.4.ipynb  # 迭代版本
│   │   ├── Generate_Data_v3.ipynb   # 探索中（GP捕捉残差）
│   │   └── Py/
│   │       ├── Generate_Data_v2.4_runner.py    # 运行时脚本
│   │       ├── Generate_Data_v2.4_recovery.py  # 参数恢复检验
│   │
│   ├── Python_for_Check/             # 【模型检验】Python代码
│   │   ├── step1_generative_checks.ipynb       # Layer 1-2 检验
│   │   ├── Parameter_Recovery.ipynb           # Layer 3 检验
│   │   ├── GP-SPE-Explore-2D.ipynb           # GP 2D 边界探索
│   │   └── GP-SPE-Explore-3D.ipynb           # GP 3D 边界探索
│   │
│   └── R_for_Check/                 # 【模型检验】R代码
│       └── Check_Generate_Data.Rmd   # 模型对比分析
│
├── 2_Data/
│   ├── Generate_Data/               # 生成的模拟数据
│   └── Real_Data/
│       ├── EXP_data_combined.csv    # 整合后的真实数据（46被试）
│       └── EXP_data_group*.csv      # 分组原始数据
│
├── 3_Figures/                       # 图表输出
└── 4_Reports/                       # 报告/PPT
```

---

## 三、数据生成模型说明

| 模型类型 | 说明 | 对应文件 | 状态 |
|---------|------|----------|------|
| **基线模型** | Sigmoid + 标准DDM | v1系列 | 完成 |
| **进阶模型** | Sigmoid + GP | v2.x系列 | v2.4稳定，v3探索中 |
| **情境化模型** | 基于HDDM的条件化建模 | 计划中 | 待开发 |

---

## 四、运行方式

### 4.1 Python (Jupyter Notebook)

```bash
# 方法1：直接运行（如果Python已配置好）
jupyter notebook

# 方法2：使用虚拟环境（如果需要）
cd 1_Code
.venv\Scripts\python.exe -m notebook
```

常用Jupyter操作：
- Shift + Enter：运行当前单元格
- Ctrl + S：保存文件
- 运行时选择正确的 kernel（Python 3）

### 4.2 R (RStudio)

1. 打开 RStudio
2. File → Open File → 选择 `.Rmd` 文件
3. 点击 "Knit" 按钮运行

### 4.3 关键脚本直接运行

```bash
# 运行数据生成脚本
cd 1_Code/Python_for_Generate/Py
python Generate_Data_v2.4_runner.py
```

---

## 五、代码规范

### 5.1 注释
- 使用**中文**注释
- 复杂逻辑需添加说明

### 5.2 变量命名
- 使用**英文**变量名
- 保持原有命名习惯（参考已有代码）

### 5.3 代码风格
- 功能实现优先
- 换行按习惯即可
- 使用 pathlib 处理路径：`from pathlib import Path`

### 5.4 导入规范
```python
# 推荐：按顺序导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# sklearn 导入
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# scipy 导入
from scipy.special import expit as sigmoid
```

---

## 六、测试建议

### 6.1 A) 简单检查：运行不报错

每次修改代码后，运行以下检查：

```python
# 检查1：基本导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 检查2：运行生成函数
from Generate_Data_v2.4_runner import generate_dataset_s2
df, gen = generate_dataset_s2(n_subjects=10, trials_per_sub=20, seed=42)
print(f"生成试次数: {len(df)}")
print("运行成功！")
```

### 6.2 B) 功能检查：数值合理性验证

```python
import pandas as pd
import numpy as np

def validate_generated_data(df):
    """验证生成数据的合理性"""
    errors = []
    warnings = []
    
    # 1. RT应该是正数
    if (df['RT'] <= 0).any():
        errors.append("RT存在非正值")
    
    # 2. RT应该在合理范围 (0.2s ~ 3s)
    if df['RT'].min() < 0.2:
        warnings.append(f"RT最小值过小: {df['RT'].min():.3f}s")
    if df['RT'].max() > 3:
        warnings.append(f"RT最大值过大: {df['RT'].max():.3f}s")
    
    # 3. 正确率应该在 50% ~ 95%
    acc = (df['response'] == 1).mean()
    if acc < 0.5:
        errors.append(f"正确率过低: {acc*100:.1f}%")
    if acc > 0.95:
        warnings.append(f"正确率过高: {acc*100:.1f}%")
    
    # 4. SPE应该在合理范围 (-100ms ~ +50ms)
    self_rt = df[df['label']=='self'].groupby('subject')['RT'].mean()
    stranger_rt = df[df['label']=='stranger'].groupby('subject')['RT'].mean()
    common = self_rt.index.intersection(stranger_rt.index)
    spe_ms = (self_rt[common] - stranger_rt[common]).mean() * 1000
    
    if spe_ms < -150:
        warnings.append(f"SPE效应过大: {spe_ms:.1f}ms (可能过拟合)")
    if spe_ms > 100:
        warnings.append(f"SPE方向异常: {spe_ms:.1f}ms")
    
    # 输出结果
    if errors:
        print("❌ 错误:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("⚠️ 警告:")
        for w in warnings:
            print(f"  - {w}")
    if not errors and not warnings:
        print("✅ 所有检查通过!")
    
    return len(errors) == 0

# 使用示例
df = pd.read_csv('2_Data/Generate_Data/gp_ddm_v2.4_small.csv')
validate_generated_data(df)
```

---

## 七、已知问题与注意事项

### 7.1 已发现问题
- RStan 包安装失败（可选，非必须）
- `Generate_Data_v2.3.ipynb` 暂不完整

### 7.2 当前稳定版本
- 数据生成：`Generate_Data_v2.4_runner.py`
- 模型检验：需参考 `Python_for_Check/` 目录

### 7.3 真实数据说明
- 真实被试数：**46**（通过GroupInfo区分）
- 数据文件：`2_Data/Real_Data/EXP_data_combined.csv`
- 真实SPE范围：约 -90ms ~ +4ms（受P/T/W调节）

---

## 八、常用命令速查

```bash
# 进入项目目录
cd D:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design

# 启动Jupyter
jupyter notebook

# 运行Python脚本
python 1_Code/Python_for_Generate/Py/Generate_Data_v2.4_runner.py

# 查看生成的数据统计
python -c "import pandas as pd; df=pd.read_csv('2_Data/Generate_Data/gp_ddm_v2.4_small.csv'); print(df.describe())"
```

---

## 九、后续开发参考

如需修改或扩展代码，请参考：

1. **调整SPE效应**：修改 `Generate_Data_v2.4_runner.py` 中的 `alaph1`、`alaph2` 参数
2. **添加新的生成模型**：参考 v1/v2/v3 的结构
3. **模型检验**：参考 `Python_for_Check/` 中的检验流程

---

*最后更新：2026-03-05*
