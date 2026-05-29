% setup_paths.m
% Psychtoolbox-3 完整环境配置脚本
% 首次使用前在 MATLAB 中运行本脚本一次，之后可直接运行 EXP_NEW 开始实验。

fprintf('=== Psychtoolbox 环境配置向导 ===\n\n');

%% 步骤0: 检查 MATLAB 架构
if ~isunix && ~strcmp(computer('arch'), 'win64')
    error('需要 64 位 MATLAB。当前架构: %s', computer('arch'));
end

%% 步骤1: 定位 Psychtoolbox 目录
ptbRoot = 'D:\学习\Coding Learning\Matlab\Psychtoolbox-3-master\Psychtoolbox';

if ~exist(ptbRoot, 'dir')
    error('Psychtoolbox 目录未找到: %s\n请修改脚本中的 ptbRoot 变量为实际路径。', ptbRoot);
end
fprintf('[1/5] Psychtoolbox 目录已确认: %s\n', ptbRoot);

%% 步骤2: 复制 LexActivator.dll（许可证管理 DLL）
lexSrc = fileparts(mfilename('fullpath'));
lexSrc = fullfile(lexSrc, 'LexActivator.dll');
lexDstDir = fullfile(ptbRoot, 'PsychBasic', 'PsychPlugins', 'Intel64');
lexDst = fullfile(lexDstDir, 'LexActivator.dll');

if exist(lexDst, 'file')
    fprintf('[2/5] LexActivator.dll 已存在于目标位置，跳过复制。\n');
elseif exist(lexSrc, 'file')
    if ~exist(lexDstDir, 'dir')
        mkdir(lexDstDir);
    end
    [status, msg] = copyfile(lexSrc, lexDst, 'f');
    if status
        fprintf('[2/5] LexActivator.dll 已复制到: %s\n', lexDst);
    else
        error('复制 LexActivator.dll 失败: %s', msg);
    end
else
    % DLL 不在本地，尝试从网络下载
    fprintf('[2/5] LexActivator.dll 未在本地找到，尝试从网络下载...\n');
    try
        addpath(fullfile(fileparts(ptbRoot), 'managementtools'));
        downloadlexactivator(0, 1);
        fprintf('[2/5] LexActivator DLL 下载并安装成功。\n');
    catch e
        error('下载 LexActivator DLL 失败: %s\n请检查网络连接后重试。', e.message);
    end
end

%% 步骤3: 添加 Psychtoolbox 到 MATLAB 路径
fprintf('[3/5] 正在添加 Psychtoolbox 到 MATLAB 路径...\n');

% 清除旧路径
w = warning('off', 'all');
pSearch = [filesep filesep 'Psychtoolbox[' filesep pathsep ']'];
while any(regexp(path, pSearch))
    paths = regexp(path, ['[^' pathsep ']+'], 'match');
    for i = 1:length(paths)
        s = char(paths{i});
        if contains(s, [filesep 'Psychtoolbox'])
            rmpath(s);
        end
    end
end
warning(w);

% 添加新路径
addpath(genpath(ptbRoot));
fprintf('[3/5] Psychtoolbox 路径已添加。\n');

%% 步骤4: 运行 PsychStartup 设置 DLL 搜索路径
fprintf('[4/5] 正在配置运行时 DLL 搜索路径...\n');
try
    PsychStartup;
    fprintf('[4/5] 运行时 DLL 路径已配置。\n');
catch e
    fprintf('[4/5] 警告: PsychStartup 执行出错，但可能不影响基本功能。\n');
    fprintf('错误信息: %s\n', e.message);
end

%% 步骤5: 持久化保存路径
fprintf('[5/5] 正在保存 MATLAB 搜索路径...\n');
try
    savepath;
    fprintf('[5/5] MATLAB 搜索路径已保存。下次启动 MATLAB 无需重新配置。\n');
catch
    fprintf('[5/5] 警告: 无法保存搜索路径（可能需要管理员权限）。\n');
    fprintf('每次启动 MATLAB 后需重新运行: addpath(genpath(''%s''));\n', ptbRoot);
end

fprintf('\n=== 环境配置完成 ===\n');
fprintf('现在可以运行 EXP_NEW 开始实验。\n');
fprintf('运行命令: >> EXP_NEW\n');
fprintf('\n提示: 如遇到许可证相关提示，请根据实验需求选择试用或输入许可证密钥。\n');
