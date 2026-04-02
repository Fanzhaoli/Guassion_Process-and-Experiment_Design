Screen('Preference', 'SkipSyncTests', 0);
InitializeMatlabOpenGL;

%% 被试基础信息
% 初始化一个空的表格
data = table();

% 基本信息收集
prompt = {'被试组别', '被试编号', '性别[1 = 女, 2 = 男]', '年龄', '惯用手[1 = 左, 2 = 右]'};
title = '实验信息'; % 标题
definput = {'','', '','',''};% 默认值
% 使用 inputdlg 函数弹出对话框，收集数据
userInput = inputdlg(prompt, title, 1, definput);

% 将收集到的数据转换为合适的类型并存入表格
groupID = str2double(userInput{1});    % 被试组别
subjectID = str2double(userInput{2});  % 被试编号
gender = str2double(userInput{3});     % 性别
age = str2double(userInput{4});        % 年龄
handedness = str2double(userInput{5}); % 惯用手

%% 屏幕窗口
% 隐藏鼠标，最后记得取消注释
HideCursor;

% 创建窗口并设置屏幕
screenNumber = max(Screen('Screens')); % 获取屏幕
isTestMode = false; % 设置测试模式，修改为true或false
% 根据是否是测试模式设置窗口大小
if isTestMode
    windowWidth = 800; % 测试模式下设置小窗口宽度
    windowHeight = 600; % 测试模式下设置小窗口高度
    rect = [0, 0, windowWidth, windowHeight]; % 设置窗口位置和大小
else
    rect = []; % 不设置具体的窗口大小，使用全屏
end
% 创建窗口
[window, windowRect] = Screen('OpenWindow', screenNumber, [128, 128, 128], rect);
% 获取屏幕大小
[screenXpixels, screenYpixels] = Screen('WindowSize', window); 
% 屏幕中心坐标
centerX = screenXpixels / 2; % 屏幕中心X坐标
centerY = screenYpixels / 2; % 屏幕中心Y坐标

% 获取显示器的帧间隔（秒/帧）
FlipIntv = Screen('GetFlipInterval', window);

%% 刺激设置
% 设置字体
Screen('TextFont', window, 'Simsun'); % 宋体

% 刺激颜色
stimColor = [255, 255, 255]; % 白色

% 注视点的大小
FixationSize = deg2pix(0.8, 16, windowRect(3), 70);

% 刺激与注视点的距离
distance = deg2pix(3.5, 16, windowRect(3), 70);

% 刺激的坐标位置
shapePositions = [
    centerX, centerY - distance; % 上方的形状
    centerX, centerY + distance; % 下方的文字
];

% 设置标签
labelEnglish = {'self', 'stranger'}; % 英文标签
labelChinese = {'自我', '生人'};  % 中文标签

% 设置形状
shapeEnglish = {'circle', 'square'}; % 英文标签
shapeChinese = {'圆形', '正方形'};  % 中文标签

% 形状的大小:3.8° * 3.8°, 计算这个条件下对应视角在屏幕中的像素数
halfShapeSize = deg2pix(1.9, 16, windowRect(3), 70);

% 定义形状和标签的正确匹配关系,_后为余数
correctPairs_0 = {
    'square', 'self';
    'circle', 'stranger'
};

correctPairs_1 = {
    'square', 'stranger';
    'circle', 'self'
};

% 根据被试编号决定刺激分配方式
if mod(subjectID, 2) == 0  % 偶数编号的被试
    correctOrder = correctPairs_0;  % 正确的匹配顺序
else   % 奇数编号的被试
    correctOrder = correctPairs_1;  % 反转正确的匹配顺序
end

% 预分配匹配和不匹配刺激的单元格数组
matchingStimuli = cell(0, 2);
nonMatchingStimuli = cell(0, 2);

% 生成匹配和不匹配刺激
for i = 1:size(correctOrder, 1)
    shape = correctOrder{i, 1};
    correctLabel = correctOrder{i, 2};
    
    % 生成匹配刺激
    matchingStimuli = [matchingStimuli; {shape, correctLabel}];
    
    % 生成不匹配刺激，排除当前的正确标签
    otherLabel = setdiff(labelEnglish, correctLabel);  % 排除当前的正确标签
    for j = 1:length(otherLabel)
        nonMatchingStimuli = [nonMatchingStimuli; {shape, otherLabel{j}}];
    end
end

% 合并匹配和不匹配的刺激
stimuli = [matchingStimuli; nonMatchingStimuli];

% 掩蔽刺激
selfMaskImages = {'self10-1.png', 'self10-2.png', 'self10-3.png', 'self10-4.png', 'self10-5.png'};
strangerMaskImages = {'stranger10-1.png', 'stranger10-2.png', 'stranger10-3.png', 'stranger10-4.png', 'stranger10-5.png'};

% 自定义实验条件(P,T,W)：每个条件的具体值
conditions = [
    0, 0.03, 0.3;   % 第一组条件 (P=0, T=30, W=300)
    0, 0.03, 0.6;  % 第二组条件 (P=0, T=30, W=600)
    120, 0.03, 0.6;  % 第三组条件 (P=120, T=30, W=600)
    120, 0.08, 0.6;  % 第四组条件 (P=120, T=80, W=600)
    8, 0.1, 1.1;  % 第五组条件 (P=8, T=100, W=1100)
    120, 0.5, 1.5;  % 第六组条件 (P=120, T=500, W=1500)
    0, 0.1, 1.1;  % 第七组条件 (用于程序崩溃后接着做)
];
currentCondition = conditions(groupID, :);  % 每组一个条件

% 练习试次数量 P
P = currentCondition(1);

% 刺激呈现时间 T
T = currentCondition(2);

% 反应窗口 W
W = currentCondition(3);

%% 联结阶段
% 指导语字体大小
Screen('TextSize', window, 30); 

% 获取形状配对规则信息
label1 = correctOrder{1, 2};  % 获取第一个形状配对的标签
label2 = correctOrder{2, 2};  % 获取第二个形状配对的标签
shape1 = correctOrder{1, 1};  % 获取第一个形状配对的标签
shape2 = correctOrder{2, 1};  % 获取第二个形状配对的标签

% 获取形状1的中文名
label1Index = find(strcmp(labelEnglish, label1));  % 获取标签1对应的英文标签索引
label1Chinese = labelChinese{label1Index};   % 获取标签1对应的中文形状
shape1Index = find(strcmp(shapeEnglish, shape1));  % 获取标签1对应的英文标签索引
shape1Chinese = shapeChinese{shape1Index};   % 获取形状1对应的中文形状

% 获取形状2的中文名
label2Index = find(strcmp(labelEnglish, label2));  % 获取形状2对应的英文标签索引
label2Chinese = labelChinese{label2Index};   % 获取形状2对应的中文形状
shape2Index = find(strcmp(shapeEnglish, shape2));  % 获取标签1对应的英文标签索引
shape2Chinese = shapeChinese{shape2Index};   % 获取形状1对应的中文形状

% 获取匹配按键
matchKey = getMatchKey(subjectID);

% 根据匹配按键定义不匹配按键
if matchKey == 'f'
    mismatchKey = 'j';
else
    mismatchKey = 'f';
end

% 联结阶段指导语1
aline1 = '你好，欢迎参加本实验。';
aline2 = '接下来首先你需要记忆几何形状（圆形、正方形）和身份标签（自我、陌生人）的配对关系。';
aline3 = '这一关系会呈现60s，60s之后会自动停止。';
aline4 = '明白了任务之后请按回车键记忆关系。';
aline5 = '如果对本实验还有不清楚之处，请立即向实验员咨询。';

% 定义每段文本的位置
DrawFormattedText(window, double(aline1), 'center', centerY - 200, stimColor); % 第一行
DrawFormattedText(window, double(aline2), 'center', centerY - 100, stimColor); % 第二行
DrawFormattedText(window, double(aline3), 'center', centerY, stimColor); % 第三行
DrawFormattedText(window, double(aline4), 'center', centerY + 100, stimColor); % 第四行
DrawFormattedText(window, double(aline5), 'center', centerY + 200, stimColor); % 第五行

% 刷新屏幕
Screen('Flip', window);

% 等待被试按空格键继续
waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        % 查找按下的键
        pressedKey = find(keyCode);  % 查找按下的键的索引
        if any(pressedKey == KbName('return'))  % 检查是否按下空格键
            waitForSpace = false;  % 按下空格键后跳出循环，继续实验
        end
    end
end


% 联结阶段指导语2
aline6 = '记忆以下规则60s：';
aline7 = sprintf('%s 是 %s , %s 是 %s',  label1Chinese, shape1Chinese, label2Chinese, shape2Chinese);

% 定义每段文本的位置
DrawFormattedText(window, double(aline6), 'center', centerY - 100, stimColor); % 第五行
DrawFormattedText(window, double(aline7), 'center', centerY, stimColor); % 第六行

% 刷新屏幕
Screen('Flip', window);

% 等待60秒后自动进入下一帧
WaitSecs(60);  % 等待60秒


% 联结阶段指导语3
aline8 = '接下来将进入正式实验。';
aline9 = '如果未记清楚记忆关系，请立即向实验员咨询。';
aline10 = '没有问题请按回车键进入实验。';

% 定义每段文本的位置
DrawFormattedText(window, double(aline8), 'center', centerY - 100, stimColor); % 第五行
DrawFormattedText(window, double(aline9), 'center', centerY, stimColor); % 第六行
DrawFormattedText(window, double(aline10), 'center', centerY + 100, stimColor); % 第七行

% 刷新屏幕
Screen('Flip', window);

% 等待被试按空格键继续
waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        % 查找按下的键
        pressedKey = find(keyCode);  % 查找按下的键的索引
        if any(pressedKey == KbName('return'))  % 检查是否按下回车键
            waitForSpace = false;  % 按下空格键后跳出循环，继续实验
        end
    end
end

% 继续后续实验流程

%% 匹配阶段-练习
% 指导语字体大小
Screen('TextSize', window, 30); 

% 练习指导语
bline1 = '屏幕中心的十字上方会呈现形状，同时下方会呈现标签。';
bline2 = '接下来的任务是判断屏幕上呈现的刺激对与刚刚学习的关系是否一致。';
bline3 = sprintf('匹配按 %s 键 , 不匹配按 %s 键', matchKey, mismatchKey);
bline4 = '有任何问题请立即向实验员咨询。';
bline5 = '没有问题请在记住配对关系后按空格键进入实验。';

DrawFormattedText(window, double(bline1), 'center', centerY - 200, stimColor); % 第一行
DrawFormattedText(window, double(bline2), 'center', centerY - 100, stimColor); % 第二行
DrawFormattedText(window, double(bline3), 'center', centerY, stimColor); % 第三行
DrawFormattedText(window, double(bline4), 'center', centerY + 100, stimColor); % 第四行
DrawFormattedText(window, double(bline5), 'center', centerY + 200, stimColor); % 第五行

% 刷新屏幕
Screen('Flip', window);

% 等待被试按空格键继续
waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        % 查找按下的键
        pressedKey = find(keyCode);  % 查找按下的键的索引
        if any(pressedKey == KbName('space'))  % 检查是否按下空格键
            waitForSpace = false;  % 按下空格键后跳出循环，继续实验
        end
    end
end

% 练习
% 动态生成文件名，根据subjectID命名
fileName = sprintf('exp_group%d_subject%d.txt', groupID, subjectID);

% 打开文件，用于保存数据，'a' 表示追加模式，确保每个 trial 保存一行数据
fileID = fopen(fileName, 'a+');
if fileID == -1
    error('无法打开文件');
end

% 在文件头部添加列标题（如果文件是空的）
if ftell(fileID) == 0
    fprintf(fileID, 'groupID\tsubjectID\tgender\tage\thandedness\tstage\ttrialID\tP\tT\tW\tShape\tLabel\tCorrectKey\tResponse\tRT\tCorrect\n');
end

% 初始化一个全局的trialIndex，用于记录当前试次
trialIndex = 1;

% 计算每次循环的试次数量
loopCount = P / 4; % 计算循环次数（P/4次）

% 循环呈现每个trial
for k = 1:loopCount  % 循环次数为 P/4

    % 打乱刺激顺序
    stimuli_bin = stimuli(randperm(size(stimuli, 1)), :);

    % 循环呈现每个trial
    for i = 1:length(stimuli)
    
        % 每个trial开始时，立即检查Esc键
        checkEscape();

        % 获取当前刺激的形状和标签
        currentShape = stimuli_bin{i, 1};
        currentLabel = stimuli_bin{i, 2};
        
        % 获取当前被试的配对规则
        pairingRules = getPairingRules(subjectID);
        
        % 使用获取到的配对规则继续进行后续的实验流程
        disp(pairingRules);  % 输出配对规则以检查
        
        % 获取当前形状和标签的按键
        correctKey = pairingRules.(currentShape).(currentLabel);
        
        % 1. 显示中央注视点 (+)，持续500ms
        % 注视点持续时间
        fixationDuration = 0.5;
        
        % 根据注视点持续时间计算所需的帧数
        fixationFrames = round(fixationDuration / FlipIntv);  % 计算所需的帧数

        % 应用注视点大小
        Screen('TextSize', window, FixationSize);  

        % 预先生成所有要显示的注视点图像
        for j = 1:fixationFrames
            % 绘制注视点（在内存缓冲区中）
            DrawFormattedText(window, '+', 'center', 'center', stimColor); 
        end

        % 刷新屏幕，显示注视点
        fixationFlipTime = Screen('Flip', window);

        while GetSecs() - fixationFlipTime < fixationDuration
            % 等待fixationDuration时间
        end
    
        % 注视点呈现检查Esc键
        checkEscape(); 
        
        % 2. 显示形状和标签，持续500ms
        % 根据刺激呈现时间计算所需的帧数
        stimuliFrames = round(T / FlipIntv);  % 计算所需的帧数

        % 预先生成所有要显示的刺激图像
        for j = 1:stimuliFrames
            % 绘制注视点（在内存缓冲区中）
            % 应用注视点大小
            Screen('TextSize', window, FixationSize); 
            DrawFormattedText(window, '+', 'center', 'center', stimColor);
        
            % 绘制形状
            % 计算左、右、上、下边界
            left = shapePositions(1, 1) - halfShapeSize;  % 左边界
            right = shapePositions(1, 1) + halfShapeSize; % 右边界
            top = shapePositions(1, 2) - halfShapeSize;   % 上边界
            bottom = shapePositions(1, 2) + halfShapeSize; % 下边界
            
            switch currentShape
                case 'circle'
                    Screen('FillOval', window, stimColor, [left, top, right, bottom]);  % 绘制圆形
                case 'square'
                    Screen('FillRect', window, stimColor, [left, top, right, bottom]);  % 绘制方形
            end
            
            % 绘制标签（显示中文标签）
            Screen('TextSize', window, 90);
            labelIndex = find(strcmp(labelEnglish, currentLabel));  % 获取对应的中文标签索引
            DrawFormattedText(window, double(labelChinese{labelIndex}), 'center', shapePositions(2, 2), stimColor); % 标签位置：注视点下方
        end
      
        % 刷新屏幕并记录刺激呈现时间
        stimulusFlipTime = Screen('Flip', window); % 显示注视点、形状和标签，并记录刺激呈现时间
        
        % 等待刺激呈现时间 T
        while GetSecs() - stimulusFlipTime < T
            % 等待持续时间
        end

        % 刺激呈现检查Esc键
        checkEscape(); 

        % 3. 插入掩蔽刺激 200 ms
        % 根据掩蔽条件选择不同的图片
        if strcmp(currentLabel, 'stranger')
            selectedMaskImage = strangerMaskImages{randi(numel(strangerMaskImages))};
            maskImage = imread(selectedMaskImage);  % 加载 stranger.png 图片
        elseif strcmp(currentLabel, 'self')
            selectedMaskImage = selfMaskImages{randi(numel(selfMaskImages))};
            maskImage = imread(selectedMaskImage);  % 加载 self.png 图片
        end
        
        % 获取图片的原始大小
        [maskHeight, maskWidth, ~] = size(maskImage);  % 获取图片的高度和宽度

        % 位置
        maskRect = [0 0 maskWidth maskHeight]; % 噪声矩形框
        maskRect = CenterRectOnPoint(maskRect, centerX, centerY + distance); % 在标签的位置显示
        
        % 将图片转换为纹理
        maskImageTexture = Screen('MakeTexture', window, maskImage);  % 将图片转换为纹理

        % 显示掩蔽刺激，持续时间 maskDuration
        maskDuration = 0.2; % 掩蔽刺激持续时间 (秒)
        Screen('DrawTexture', window, maskImageTexture, [], maskRect); % 在屏幕上绘制噪声掩蔽
        maskFlipTime = Screen('Flip', window); % 显示掩蔽刺激
        while GetSecs() - maskFlipTime < maskDuration
              % 等待掩蔽刺激持续时间
        end

        % 掩蔽刺激呈现检查Esc键
        checkEscape(); 
    
        % 4. 显示空白屏，等待键盘反应，持续反应窗口 W - maskDuration
        % 显示空白屏并记录flip时间
        responseFlipTime = Screen('Flip', window); 
    
        response = NaN;      % 初始化响应为NaN
        responseTime = NaN;  % 初始化反应时间为空
        
        % 循环等待按键输入或超时
        while GetSecs() - responseFlipTime < W - maskDuration
            % 在空白屏显示期间等待按键
            [keyIsDown, ~, keyCode] = KbCheck;
            
            if keyIsDown
                % 获取按下的所有键的索引
                pressedKeys = find(keyCode);
                
                if ~isempty(pressedKeys)
                    % 只处理第一个按下的键
                    pressedKey = KbName(pressedKeys(1));  % 只处理第一个按下的键
                    
                    if ismember(pressedKey, {'f', 'j'})
                        response = pressedKey;  % 记录按键
                        responseTime = GetSecs() - stimulusFlipTime;  % 计算反应时
                        break;  % 跳出循环，停止等待
                    else
                        % 如果按下的不是'f'或'j'，则不记录，继续等待
                        continue;  % 不做任何操作，继续等待
                    end
                end
            end
        end

        % 空白屏呈现检查Esc键
        checkEscape(); 

        % 如果没有按键响应，则保持response为NaN（此时response默认是NaN，无需额外赋值）

        % 列:1.被试编号, 2.性别, 3.年龄, 4.利手, 5.实验阶段, 6.试次数, 7.试次形状, 8.试次标签, 9.正确的键, 10.被试按键, 11.反应时, 12.是否正确
        % 存储试次数据
        data{trialIndex, 1} = groupID;
        data{trialIndex, 2} = subjectID;
        data{trialIndex, 3} = gender;
        data{trialIndex, 4} = age;
        data{trialIndex, 5} = handedness;
        data{trialIndex, 6} = "practice"; % 存储实验阶段
        data{trialIndex, 7} = trialIndex; % 存储试次编号
        data{trialIndex, 8} = P; % 存储练习试次数量
        data{trialIndex, 9} = T; % 存储刺激呈现时间
        data{trialIndex, 10} = W;   % 存储反应窗口
        data{trialIndex, 11} = currentShape; % 存储试次形状
        data{trialIndex, 12} = string(currentLabel); % 存储试次标签
        data{trialIndex, 13} = char(correctKey);   % 存储正确按键
        data{trialIndex, 14} = char(response);     % 存储被试按键
        data{trialIndex, 15} = responseTime; % 存储反应时（单位：秒）
    
        % 判断是否正确
        if isempty(response)  % 如果没有响应
            data{trialIndex, 16} = NaN; % 无反应
        elseif strcmp(response, correctKey)  % 如果响应正确
            data{trialIndex, 16} = 1; % 正确
        else  % 如果响应错误
            data{trialIndex, 16} = 0; % 错误
        end
    
        % 5. 显示反馈（正确或错误或无响应）
        if isnan(data{trialIndex, 15})  % 无响应，即未在W内做出反应
            feedbackText = double('过慢！');
            feedbackColor = [255, 0, 0];  % 红色
        elseif data{trialIndex, 15} < 0.15  % 反应时小于150ms记为过快
            feedbackText = double('过快！');
            feedbackColor = [255, 0, 0];  % 红色
        elseif data{trialIndex, 16} == 1  % 正确
            feedbackText = double('正确！');
            feedbackColor = [0, 255, 0];  % 绿色
        elseif data{trialIndex, 16} == 0  % 错误
            feedbackText = double('错误！');
            feedbackColor = [255, 0, 0];  % 红色
        end
        
        % 显示反馈
        DrawFormattedText(window, feedbackText, 'center', 'center', feedbackColor); 
        feedbackFlipTime = Screen('Flip', window); % 显示反馈
        WaitSecs(0.5); % 呈现500 ms

        % 反馈检查Esc键
        checkEscape(); 

        % 写入数据到文件
        fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3f\t%d\n', ...
            data{trialIndex, 1}, data{trialIndex, 2}, data{trialIndex, 3}, data{trialIndex, 4}, ...
            data{trialIndex, 5}, data{trialIndex, 6}, data{trialIndex, 7}, data{trialIndex, 8}, ...
            data{trialIndex, 9}, data{trialIndex, 10}, data{trialIndex, 11}, data{trialIndex, 12}, ...
            data{trialIndex, 13}, data{trialIndex, 14}, data{trialIndex, 15}, data{trialIndex, 16});

        % 更新trialIndex
        trialIndex = trialIndex + 1;  % 确保在每次循环后trialIndex递增

    end
end

%% 练习与正式实验之间的休息
% 指导语字体大小
Screen('TextSize', window, 30); 

% 每个block结束后，提示休息并等待按任意键继续
restMessage1 = '已完成练习，';
restMessage2 = '休息后按空格键进入正式实验。';
DrawFormattedText(window, double(restMessage1), 'center', 'center', stimColor); % 第一行
DrawFormattedText(window, double(restMessage2), 'center', centerY + 100, stimColor); % 第二行
Screen('Flip', window);  % 显示休息提示

% 等待被试按空格键继续
waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        % 查找按下的键
        pressedKey = find(keyCode);  % 查找按下的键的索引
        if any(pressedKey == KbName('space'))  % 检查是否按下空格键
            waitForSpace = false;  % 按下空格键后跳出循环，继续实验
        end
    end
end

%% 匹配阶段-正式实验
% 设置block的数量
block = 10;
blockTrials = 52;

% 循环多个block
for b = 1:block

    % 记录当前block的正确回答数
    correctCount = 0;

    % 初始化当前block的反应时间数组
    responseTimes = [];

    % 循环一个block
    for y = 1:(blockTrials/length(stimuli))
    
        % 打乱刺激顺序
        stimuli_bin = stimuli(randperm(size(stimuli, 1)), :);

        % 循环bin
        for i = 1:length(stimuli)

            % 每个trial开始时，立即检查Esc键
            checkEscape();

            % 获取当前刺激的形状和标签
            currentShape = stimuli_bin{i, 1};
            currentLabel = stimuli_bin{i, 2};
            
            % 获取当前被试的配对规则
            pairingRules = getPairingRules(subjectID);
            
            % 使用获取到的配对规则继续进行后续的实验流程
            disp(pairingRules);  % 输出配对规则以检查
            
            % 获取当前形状和标签的按键
            correctKey = pairingRules.(currentShape).(currentLabel);
            
            % 1. 显示中央注视点 (+)，持续500ms
            % 注视点持续时间
            fixationDuration = 0.5;
            
            % 根据注视点持续时间计算所需的帧数
            fixationFrames = round(fixationDuration / FlipIntv);  % 计算所需的帧数
    
            % 应用注视点大小
            Screen('TextSize', window, FixationSize); 
            DrawFormattedText(window, '+', 'center', 'center', stimColor);
    
            % 预先生成所有要显示的注视点图像
            for j = 1:fixationFrames
                % 绘制注视点（在内存缓冲区中）
                DrawFormattedText(window, '+', 'center', 'center', stimColor); 
            end
    
            % 刷新屏幕，显示注视点
            fixationFlipTime = Screen('Flip', window);
    
            while GetSecs() - fixationFlipTime < fixationDuration
                % 等待fixationDuration时间
            end

            % 注视点呈现检查Esc键
            checkEscape(); 
        
            % 2. 显示形状和标签，持续500ms
            % 根据刺激呈现时间计算所需的帧数
            stimuliFrames = round(T / FlipIntv);  % 计算所需的帧数
    
            % 预先生成所有要显示的刺激图像
            for j = 1:stimuliFrames

                % 应用注视点大小
                Screen('TextSize', window, FixationSize); 
                % 绘制注视点（在内存缓冲区中）
                DrawFormattedText(window, '+', 'center', 'center', stimColor); 
            
                % 绘制形状
                % 计算左、右、上、下边界
                left = shapePositions(1, 1) - halfShapeSize;  % 左边界
                right = shapePositions(1, 1) + halfShapeSize; % 右边界
                top = shapePositions(1, 2) - halfShapeSize;   % 上边界
                bottom = shapePositions(1, 2) + halfShapeSize; % 下边界
                
                switch currentShape
                    case 'circle'
                        Screen('FillOval', window, stimColor, [left, top, right, bottom]);  % 绘制圆形
                    case 'square'
                        Screen('FillRect', window, stimColor, [left, top, right, bottom]);  % 绘制方形
                end
                
                % 绘制标签（显示中文标签）
                Screen('TextSize', window, 90);
                labelIndex = find(strcmp(labelEnglish, currentLabel));  % 获取对应的中文标签索引
                DrawFormattedText(window, double(labelChinese{labelIndex}), 'center', shapePositions(2, 2), stimColor); % 标签位置：注视点下方
            end
          
            % 刷新屏幕并记录刺激呈现时间
            stimulusFlipTime = Screen('Flip', window); % 显示注视点、形状和标签，并记录刺激呈现时间
            
            % 等待刺激呈现时间 T
            while GetSecs() - stimulusFlipTime < T
                % 等待持续时间
            end

            % 刺激呈现检查Esc键
            checkEscape(); 

            % 3. 插入掩蔽刺激 200 ms
            % 根据掩蔽条件选择不同的图片
            if strcmp(currentLabel, 'stranger')
                selectedMaskImage = strangerMaskImages{randi(numel(strangerMaskImages))};
                maskImage = imread(selectedMaskImage);  % 加载 stranger.png 图片
            elseif strcmp(currentLabel, 'self')
                selectedMaskImage = selfMaskImages{randi(numel(selfMaskImages))};
                maskImage = imread(selectedMaskImage);  % 加载 self.png 图片
            end
            
            % 获取图片的原始大小
            [maskHeight, maskWidth, ~] = size(maskImage);  % 获取图片的高度和宽度
    
            % 位置
            maskRect = [0 0 maskWidth maskHeight]; % 噪声矩形框
            maskRect = CenterRectOnPoint(maskRect, centerX, centerY + distance); % 在标签的位置显示
            
            % 将图片转换为纹理
            maskImageTexture = Screen('MakeTexture', window, maskImage);  % 将图片转换为纹理
    
            % 显示掩蔽刺激，持续时间 maskDuration
            maskDuration = 0.2; % 掩蔽刺激持续时间 (秒)
            Screen('DrawTexture', window, maskImageTexture, [], maskRect); % 在屏幕上绘制噪声掩蔽
            maskFlipTime = Screen('Flip', window); % 显示掩蔽刺激
            while GetSecs() - maskFlipTime < maskDuration
                  % 等待掩蔽刺激持续时间
            end
    
            % 掩蔽刺激呈现检查Esc键
            checkEscape(); 
        
            % 4. 显示空白屏，等待键盘反应，持续反应窗口 W - maskDuration
            % 显示空白屏并记录flip时间
            responseFlipTime = Screen('Flip', window); 
        
            response = NaN;      % 初始化响应为NaN
            responseTime = NaN;  % 初始化反应时间为空
            
            % 循环等待按键输入或超时
            while GetSecs() - responseFlipTime < W - maskDuration
                % 在空白屏显示期间等待按键
                [keyIsDown, ~, keyCode] = KbCheck;
                
                if keyIsDown
                    % 获取按下的所有键的索引
                    pressedKeys = find(keyCode);
                    
                    if ~isempty(pressedKeys)
                        % 只处理第一个按下的键
                        pressedKey = KbName(pressedKeys(1));  % 只处理第一个按下的键
                        
                        if ismember(pressedKey, {'f', 'j'})
                            response = pressedKey;  % 记录按键
                            responseTime = GetSecs() - stimulusFlipTime;  % 计算反应时
                            break;  % 跳出循环，停止等待
                        else
                            % 如果按下的不是'f'或'j'，则不记录，继续等待
                            continue;  % 不做任何操作，继续等待
                        end
                    end
                end
            end

            % 空白屏呈现检查Esc键
            checkEscape(); 
    
            % 如果没有按键响应，则保持response为NaN（此时response默认是NaN，无需额外赋值）
    
            % 列:1.被试组别, 2.编号, 2.性别, 3.年龄, 4.利手, 5.实验阶段, 6.试次数, 7.试次形状, 8.试次标签, 9.正确的键, 10.被试按键, 11.反应时, 12.是否正确
            % 存储试次数据
            data{trialIndex, 1} = groupID;
            data{trialIndex, 2} = subjectID;
            data{trialIndex, 3} = gender;
            data{trialIndex, 4} = age;
            data{trialIndex, 5} = handedness;
            data{trialIndex, 6} = "formal"; % 存储实验阶段
            data{trialIndex, 7} = trialIndex; % 存储试次编号
            data{trialIndex, 8} = P; % 存储练习试次数量
            data{trialIndex, 9} = T; % 存储刺激呈现时间
            data{trialIndex, 10} = W; % 存储反应窗口
            data{trialIndex, 11} = currentShape; % 存储试次形状
            data{trialIndex, 12} = string(currentLabel); % 存储试次标签
            data{trialIndex, 13} = char(correctKey);   % 存储正确按键
            data{trialIndex, 14} = char(response);     % 存储被试按键
            data{trialIndex, 15} = responseTime; % 存储反应时（单位：秒）
        
            % 判断是否正确
            if isempty(response)  % 如果没有响应
                data{trialIndex, 16} = NaN; % 无反应
            elseif strcmp(response, correctKey)  % 如果响应正确
                data{trialIndex, 16} = 1; % 正确
                correctCount = correctCount + 1; % 正确计数加1
            else  % 如果响应错误
                data{trialIndex, 16} = 0; % 错误
            end

            % 记录每次反应时间
            if ~isnan(responseTime)
                responseTimes = [responseTimes, responseTime]; % 将有效的反应时间加入数组
            end
            
            % 写入数据到文件
            fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3f\t%d\n', ...
                data{trialIndex, 1}, data{trialIndex, 2}, data{trialIndex, 3}, data{trialIndex, 4}, ...
                data{trialIndex, 5}, data{trialIndex, 6}, data{trialIndex, 7}, data{trialIndex, 8}, ...
                data{trialIndex, 9}, data{trialIndex, 10}, data{trialIndex, 11}, data{trialIndex, 12}, ...
                data{trialIndex, 13}, data{trialIndex, 14}, data{trialIndex, 15}, data{trialIndex, 16});
    
            % 更新trialIndex
            trialIndex = trialIndex + 1;  % 确保在每次循环后trialIndex递增
    
        end
    end

    % 计算当前block的正确率
    accuracy = correctCount / blockTrials;  % 正确率 = 正确回答数 / block总试次数
    % 平均反应时
    avgRT = mean(responseTimes);  
    
    % 指导语字体大小
    Screen('TextSize', window, 30); 
    
    % 显示当前block的正确率和平均反应时
    message1 = sprintf('正确率: %.2f%% ', accuracy * 100);
    message2 = sprintf('平均反应时: %.3f 秒', avgRT);
    message3 = sprintf('按回车键进入休息');

    DrawFormattedText(window, double(message1), 'center', centerY - 150, stimColor); % 显示正确率
    DrawFormattedText(window, double(message2), 'center', centerY, stimColor); % 显示正确率
    DrawFormattedText(window, double(message3), 'center', centerY + 150, stimColor); % 显示正确率
    
    Screen('Flip', window); % 刷新屏幕显示

    % 等待被试按回车键继续
    waitForSpace = true;
    while waitForSpace
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            % 查找按下的键
            pressedKey = find(keyCode);  % 查找按下的键的索引
            if any(pressedKey == KbName('return'))  % 检查是否按下空格键
                waitForSpace = false;  % 按下空格键后跳出循环，继续实验
            end
        end
    end

    % 指导语字体大小
    Screen('TextSize', window, 30); 

    % 每个block结束后，提示休息并等待按空格键继续
    restMessage = double('休息中，请按空格键继续。');
    DrawFormattedText(window, restMessage, 'center', 'center', stimColor);
    Screen('Flip', window);  % 显示休息提示
    
    % 等待被试按空格键继续
    waitForSpace = true;
    while waitForSpace
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            % 查找按下的键
            pressedKey = find(keyCode);  % 查找按下的键的索引
            if any(pressedKey == KbName('space'))  % 检查是否按下空格键
                waitForSpace = false;  % 按下空格键后跳出循环，继续实验
            end
        end
    end
end

%% 结束
dline1 = '已完成所有实验，';
dline2 = '感谢您的参与。';
dline3 = '按回车键结束实验。';

DrawFormattedText(window, double(dline1), 'center', centerY - 100, stimColor); % 第一行
DrawFormattedText(window, double(dline2), 'center', centerY, stimColor); % 第二行
DrawFormattedText(window, double(dline3), 'center', centerY + 100, stimColor); % 第三行

Screen('Flip', window);

% 等待被试按空格键继续
waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        % 查找按下的键
        pressedKey = find(keyCode);  % 查找按下的键的索引
        if any(pressedKey == KbName('return'))  % 检查是否按下空格键
            waitForSpace = false;  % 按下空格键后跳出循环，继续实验
        end
    end
end

%% 保存数据
columnNames = {'groupID', 'subjectID', 'gender', 'age', 'handedness', 'stage', 'trialID', 'P', 'T', 'W', 'Shape', 'Label', 'CorrectKey', 'Response','RT', 'Correct'};
data.Properties.VariableNames = columnNames;
% 设置保存文件的路径（根据被试编号命名文件）
filename = ['EXP_data_group' num2str(groupID) '_' num2str(subjectID) '.csv'];
% 保存为 CSV 文件
writetable(data, filename);
disp(['Data has been saved to ', filename]);

% 关闭窗口
Screen('CloseAll');

%% Functions
function pairingRules = getPairingRules(subjectID)
    % 根据被试编号决定刺激分配方式    
    % 通过 mod(subjectID, 4) 来判断被试编号的规则，编号可以是 0, 1, 2, 3
    modResult = mod(subjectID, 4);
    
    % 定义所有规则，四种情况
    rules = { ...
        % 4的倍数, f是匹配, j是不匹配
        'square', struct('self', 'f', 'stranger', 'j'), 'circle', struct('self', 'j', 'stranger', 'f'); ...
        % 4k+1 的奇数, f是匹配, j是不匹配
        'square', struct('self', 'j', 'stranger', 'f'), 'circle', struct('self', 'f', 'stranger', 'j'); ...
        % 其他偶数, j是匹配, f是不匹配
        'square', struct('self', 'j', 'stranger', 'f'), 'circle', struct('self', 'f', 'stranger', 'j'); ...
        % 4k+3 的奇数, j是匹配, f是不匹配
        'square', struct('self', 'f', 'stranger', 'j'), 'circle', struct('self', 'j', 'stranger', 'f')  ... 
    };

    % 通过 modResult 查找对应的规则
    pairingRules = struct(rules{modResult + 1, :});
end

function key = getMatchKey(subjectID)
    % 根据 subjectID 返回匹配按键，按循环模式 {f, j, j, f}
    % 输入: subjectID (正整数)
    % 输出: key (匹配按键 'f' 或 'j')
    
    % 检查输入是否为正整数
    if subjectID <= 0 || mod(subjectID, 1) ~= 0
        error('subjectID 必须是正整数');
    end
    
    % 定义循环模式
    matchKeys = {'f', 'j', 'j', 'f'};
    
    % 通过取模计算模式索引
    index = mod(subjectID - 1, length(matchKeys)) + 1; % 映射到 1-4 范围
    key = matchKeys{index};
end

function pixs = deg2pix(degree,inch,pwidth,vdist) 

screenWidth = inch*2.54/sqrt(1+11.81/15.75);
pix=screenWidth/pwidth; 
pixs = round(2*tan((degree/2)*pi/180) * vdist / pix); 

end

function checkEscape()
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(KbName('esc'))  % 如果按下Esc键
        disp('实验结束！');
        sca;  % 关闭屏幕
    end
end





 



           
          

         
 