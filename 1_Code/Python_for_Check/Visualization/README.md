# 高斯过程实验设计 - 数据可视化分析平台

## 项目简介

本项目是一个基于 HTML/JavaScript + Python HTTP 服务器的交互式数据可视化网页应用，用于分析 Shape-Label Association Task 实验的行为数据。该应用实现了以下核心功能：

1. **实验按键逻辑模拟模块** - 复现4试次循环的按键逻辑，支持不同被试ID的按键映射规则查询
2. **数据选择与浏览模块** - 从指定路径加载单个被试的真实行为数据，支持数据预览和RT分布直方图
3. **交互式CRF可视化模块** - 基于分位数分箱的累积反应函数(CRF)可视化，展示Self/Stranger差异及SPE效应
4. **探索性数据分析模块** - 实验设计参数(P/T/W)影响分析，数据质量敏感性分析

## 文件结构

```
Visualization/
├── README.md                    # 本文件
├── app_server.py                # Python HTTP 服务器 (后端API)
├── visualization_app.html       # 前端可视化应用主页面
├── chart.umd.min.js             # Chart.js 本地库 (图表渲染)
└── V1/                          # 旧版本Python可视化代码
    ├── check_data_logic.py
    ├── check_data_logic_V2.py
    ├── CRF_visualization.py
    └── CRF_visualization_V2.py
```

## 环境要求

- **Python**: 3.7 或更高版本
- **操作系统**: Windows / macOS / Linux
- **浏览器**: Chrome / Firefox / Edge (推荐 Chrome)
- **网络**: 无需联网 (所有依赖均为本地文件)

## 启动方法

### 1. 启动服务器

打开终端，进入项目目录，运行以下命令：

```bash
cd "d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\Visualization"
python app_server.py
```

服务器启动后会显示以下信息：

```
============================================================
  Experiment Data Visualization Server
  Local:  http://localhost:8899
  Health: http://localhost:8899/api/health
  Press Ctrl+C to stop
============================================================
  HTML file: d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\Visualization\visualization_app.html
  Data dir:  d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\2_Data\Real_Data\UnExtact\raw
============================================================
```

**注意**: 如果端口 8899 被占用，服务器会自动尝试 8890、8891 等端口。

### 2. 访问应用

在浏览器中打开以下地址：

```
http://localhost:8899
```

页面右上角会显示 **"已连接"** 表示连接成功。

## 功能模块说明

### 标签页1: 实验按键逻辑模拟

**功能**: 模拟并验证实验的按键逻辑

**使用方法**:

1. 输入被试编号 (Subject ID) 和选择组别 (Group ID)
2. 点击 **"查询实验参数"** 查看该被试的按键映射规则
3. 点击左侧四个按钮 (□+Self, □+Stranger, ○+Self, ○+Stranger) 模拟试次
4. 右侧显示当前试次的正确按键和匹配/不匹配条件

**核心逻辑**:

- 4试次循环: 按键模式 {f, j, j, f}，由 `mod(subjectID-1, 4)` 决定
- 按键映射: 4种被试间模式，由 `mod(subjectID, 4)` 决定
- 正确配对: 由 `mod(subjectID, 2)` 决定（偶数→□↔Self, 奇数→□↔Stranger）

### 标签页2: 数据选择与浏览

**功能**: 浏览和预览被试数据文件

**使用方法**:

1. 使用顶部的组别标签 (G1-G8) 筛选要显示的文件
2. 在搜索框中输入文件名关键词进行搜索
3. 点击左侧文件列表中的文件加载数据
4. 右侧显示数据预览表格 (前100行) 和 RT 分布直方图
5. 左下角显示数据概览统计 (总试次、正式试次、遗漏率、正确率)

**数据说明**:

- 数据文件位于: `2_Data/Real_Data/UnExtact/raw/EXP_data_group*.csv`
- 共88个数据文件，分布在8个组别

### 标签页3: 交互式CRF可视化

**功能**: 生成累积反应函数(CRF)图表，分析Self-Processing Effect (SPE)

**使用方法**:

1. **单个被试分析**: 从下拉菜单选择被试文件，直接生成CRF图
2. **聚合数据分析**:
   - 点击组别标签选择要分析的组别 (可多选)
   - 点击 **"加载选中组别数据"** 加载聚合数据
3. 使用筛选条件 (Matching/NonMatching, Self/Stranger, 仅正确试次) 过滤数据
4. 调整分位数滑块 (3-10) 改变分箱数量
5. 主图显示 Self (橙色) 和 Stranger (蓝色) 的CRF曲线
6. 下方显示 SPE差异图 (Self - Stranger)

**图表说明**:

- **横轴**: RT均值 (秒)，按分位数分箱
- **纵轴**: 上边界选择比例 (选择Matching的比例)
- **0.5虚线**: 无偏向基线
- **橙色曲线**: Self 条件
- **蓝色曲线**: Stranger 条件

### 标签页4: 设计空间分析

**功能**: 分析实验设计参数对数据质量的影响

**使用方法**:

1. 点击 **"加载所有数据进行分析"** 加载全部88个文件的数据
2. 使用筛选条件 (Good/Caution/Exclude) 控制显示哪些质量等级的组别
3. 查看三个分析图表:
   - **气泡图**: P/T/W参数空间中的组别分布 (气泡大小=P，X轴=T，Y轴=W)
   - **正确率图**: 各组正确率对比
   - **遗漏率图**: 各组数据质量 (遗漏率) 对比

**实验设计参数**:

| 组别 | P (练习试次) | T (刺激呈现) | W (反应窗口) | 质量等级 |
| ---- | ------------ | ------------ | ------------ | -------- |
| G1   | 0            | 0.03s        | 0.3s         | exclude  |
| G2   | 0            | 0.03s        | 0.6s         | exclude  |
| G3   | 120          | 0.03s        | 0.6s         | caution  |
| G4   | 120          | 0.08s        | 0.6s         | good     |
| G5   | 8            | 0.1s         | 1.1s         | good     |
| G6   | 120          | 0.5s         | 1.5s         | good     |
| G7   | 0            | 0.1s         | 1.1s         | good     |
| G8   | 120          | 0.03s        | 0.8s         | good     |

## 数据核对结果

数据核对脚本 `exp_Check/data_verification.py` 已完成对88个数据文件的验证：

- **通过验证**: 87个文件
- **异常文件**: 1个 (EXP_data_group2_11.csv，发现677个错误)

核对内容：

- 试次顺序与4试次循环逻辑一致性
- 按键映射规则正确性
- 匹配/不匹配条件判断正确性

## API 接口

服务器提供以下 REST API：

| 接口                                             | 方法 | 说明                             |
| ------------------------------------------------ | ---- | -------------------------------- |
| `/api/health`                                  | GET  | 健康检查                         |
| `/api/files`                                   | GET  | 获取所有数据文件列表             |
| `/api/data/all?group=`                         | GET  | 获取聚合数据 (可选group参数筛选) |
| `/api/data/file?name=`                         | GET  | 获取单个文件数据                 |
| `/api/experiment/params?subject=&group=`       | GET  | 获取实验参数                     |
| `/api/experiment/trial?subject=&shape=&label=` | GET  | 模拟单个试次                     |

## 常见问题

### Q1: 页面显示 "服务器离线"

**原因**: 未启动 Python 服务器或服务器已停止

**解决**: 重新运行 `python app_server.py`

### Q2: 图表无法显示

**原因**: 浏览器缓存问题或 Chart.js 加载失败

**解决**:

1. 刷新页面 (Ctrl+F5)
2. 检查 `chart.umd.min.js` 文件是否存在

### Q3: 数据文件无法加载

**原因**: 数据文件路径配置错误

**解决**: 检查 `app_server.py` 中的 `RAW_DIR` 路径是否正确指向数据目录

### Q4: 如何停止服务器

**解决**: 在终端中按 `Ctrl+C`

## 技术说明

### 前端技术栈

- **HTML5/CSS3**: 页面结构和样式
- **原生 JavaScript**: 交互逻辑
- **Chart.js**: 图表渲染 (本地文件，无需CDN)

### 后端技术栈

- **Python 3**: 服务器运行环境
- **http.server**: 内置HTTP服务器
- **csv**: 数据文件解析

### 数据流

```
浏览器 (visualization_app.html)
    ↕ HTTP请求
Python服务器 (app_server.py)
    ↕ 文件读取
CSV数据文件 (2_Data/Real_Data/UnExtact/raw/*.csv)
```

## 版本历史

- **v0.1** (2026-05-27): 初始版本，完成4个核心功能模块

## 相关文件

- **MATLAB实验代码**: `1_Code/Experiment/exp_matlab/experiment_formal_newcon.m`
- **数据核对脚本**: `1_Code/Experiment/exp_Check/data_verification.py`
- **原始数据**: `2_Data/Real_Data/UnExtact/raw/EXP_data_group*.csv`

## 联系与支持

如有问题或建议，请联系项目维护者。
