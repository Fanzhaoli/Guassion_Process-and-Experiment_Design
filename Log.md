# 版本更新日志

---

## v0.3 (2026-05-26)

### 问题修复
- **修复 Screen.mexw64 加载失败 (找不到指定的模块)**：
  - 根因：`LexActivator.dll`（Psychtoolbox 许可证管理 DLL）缺失。该 DLL 是 Screen.mexw64、WaitSecs.mexw64、GetSecs.mexw64 等所有 MEX 文件的静态导入依赖
  - 原因：用户使用的 `Psychtoolbox-3-master` 是 GitHub 开发者源代码仓库，不含 `LexActivator.dll` 商业组件
  - 下载了 LexActivator-Win.zip (v3.31.2) 并提取 `LexActivator.dll` (vc14/x64) 至项目目录
  - 更新 `setup_paths.m` 为完整自动化配置脚本（5步：确认目录→安装DLL→添加路径→配置运行时→保存路径）

### 功能完善
- `setup_paths.m` 现在自动处理 LexActivator.dll 的安装（优先本地复制，备选网络下载 `downloadlexactivator`）
- 脚本自动运行 `PsychStartup` 将 PsychPlugins 目录添加到系统 PATH，确保 MEX 文件能找到运行时 DLL

### 文件变更
- 修改: `1_Code/Experiment/exp_matlab/setup_paths.m`
- 新增: `1_Code/Experiment/exp_matlab/LexActivator.dll`

---

## v0.2 (2026-05-26)

### 新增功能
- 创建 `setup_paths.m` 环境配置脚本：一键将 Psychtoolbox-3 添加到 MATLAB 搜索路径并持久化保存

### 问题修复
- 确认 `Screen` 报错根因：MATLAB R2023b (E:\matlab2023b) 已安装，Psychtoolbox-3 已下载至 `D:\学习\Coding Learning\Matlab\Psychtoolbox-3-master\`，但未加入 MATLAB 搜索路径
- 确认 Psychtoolbox MEX 文件完整：Screen.mexw64、PsychHID.mexw64、GetSecs.mexw64、WaitSecs.mexw64 等均存在

### 功能完善
- 提供一键配置脚本 `setup_paths.m`，用户首次使用时运行一次即可永久配置环境

### 文件变更
- 新建: `1_Code/Experiment/exp_matlab/setup_paths.m`

---

## v0.1 (2026-05-26)

### 新增功能
- 创建 `EXP_NEW.m`，基于 `experiment_formal_newcon.m` 重命名并修复后的正式实验脚本

### 问题修复
- **Psychtoolbox 缺失检测**：脚本开头新增 `exist('Screen', 'file')` 检查，若 Psychtoolbox-3 未安装或不在 MATLAB 路径中，给出明确的中文错误提示和安装指引
- **checkEscape() 函数缺陷修复**：原函数调用 `sca` 关闭窗口后脚本仍继续执行，导致后续 `Screen()` 调用全部崩溃。修复后增加 `error()` 终止脚本，干净退出实验

### 功能完善
- 保留原 `experiment_formal_newcon.m` 的 9 组实验条件（conditions 1-9），支持 groupID 1-9

### 文件变更
- 新建: `1_Code/Experiment/exp_matlab/EXP_NEW.m`

---
