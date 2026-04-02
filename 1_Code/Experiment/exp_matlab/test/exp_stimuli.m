Screen('Preference', 'SkipSyncTests', 2);
InitializeMatlabOpenGL;

%% 窗口
% 进行刺激调试的时候要全屏
% 创建窗口并设置屏幕
screenNumber = max(Screen('Screens')); % 获取屏幕编号，若有多个获取最大值
% 创建窗口
[window, windowRect] = Screen('OpenWindow', screenNumber, [128, 128, 128]); % 背景颜色：灰色
% 获取屏幕大小
[screenXpixels, screenYpixels] = Screen('WindowSize', window); 

% 屏幕中心坐标
centerX = screenXpixels / 2; % 屏幕中心X坐标
centerY = screenYpixels / 2; % 屏幕中心Y坐标

%% 颜色与位置
% 假设注视点和形状的颜色为白色
stimColor = [255, 255, 255];

% 注视点的大小
FixationSize = deg2pix(0.8, 16, windowRect(3), 70);

% 刺激与注视点的距离
distance = deg2pix(3.5, 16, windowRect(3), 70);

% 刺激的坐标位置
shapePositions = [
    centerX, centerY - distance; % 上方的形状
    centerX, centerY + distance % 下方的文字
];

%% 形状
% 形状的大小: 3.8° * 3.8°，计算这个条件下对应视角在屏幕中的像素数
halfShapeSize = deg2pix(1.9, 16, windowRect(3), 70);  % 半个形状的大小

%% 标签
% 目标文本
labelText = double('自我');
Screen('TextFont', window, 'Simsun'); % 宋体

%% 绘制刺激
% 绘制注视点
Screen('TextSize', window, FixationSize); 
DrawFormattedText(window, '+', 'center', 'center', stimColor);

% 绘制圆形
% 计算左、右、上、下边界
left = shapePositions(1, 1) - halfShapeSize;  % 左边界
right = shapePositions(1, 1) + halfShapeSize; % 右边界
top = shapePositions(1, 2) - halfShapeSize;   % 上边界
bottom = shapePositions(1, 2) + halfShapeSize; % 下边界

% 绘制圆形
Screen('FillOval', window, stimColor, [left, top, right, bottom]);  % 绘制圆形

% 绘制标签
% 设置字体大小
Screen('TextSize', window, 90);  % 更改字体大小确定像素和视角的差别
DrawFormattedText(window, labelText, 'center', shapePositions(2, 2), stimColor);

% 更新屏幕
Screen('Flip', window);

%% 计算
% 获取文本边界框
textBounds = Screen('TextBounds', window, labelText);
textWidth = textBounds(3) - textBounds(1);  % 计算文本的宽度
textHeight = textBounds(4) - textBounds(2);  % 计算文本的高度

% target
tarWidth = deg2pix(3.1, 16, windowRect(3), 70);
tarHeight = deg2pix(1.6, 16, windowRect(3), 70);

% 显示结果
disp(['font文本宽度: ', num2str(textWidth), ' 像素']); % 158.2031
disp(['font文本高度: ', num2str(textHeight), ' 像素']); % 82
disp(['tar文本宽度: ', num2str(tarWidth), ' 像素']); % 158
disp(['tar文本高度: ', num2str(tarHeight), ' 像素']); % 81

% 等待一段时间并关闭窗口
WaitSecs(1);
Screen('CloseAll');

% 辅助函数：计算标签的像素宽度和高度
function pixs = deg2pix(degree, inch, pwidth, vdist)
    % 计算标签宽度和高度（像素）
    % degree 和 inch 为标签视角和屏幕对角线长度
    % pwidth 是屏幕宽度的像素数，vdist 是视距

    screenWidth = inch * 2.54 / sqrt(1 + 11.81/15.75); % inch是屏幕对角线长度，假设16:9比例
    pix = screenWidth / pwidth;  % 计算每像素对应的物理距离
    pixs = round(2 * tan(degree / 2 * pi / 180) * vdist / pix);  % 计算角度对应的像素数
end



