if ~exist('Screen', 'file')
    error(['Psychtoolbox-3 未安装或不在 MATLAB 搜索路径中。\n', ...
           '请访问 http://psychtoolbox.org/download 下载并安装 Psychtoolbox-3。\n', ...
           '安装后在 MATLAB 命令窗口运行: SetupPsychtoolbox\n', ...
           '或手动添加 Psychtoolbox 文件夹到 MATLAB 路径。']);
end

Screen('Preference', 'SkipSyncTests', 0);
InitializeMatlabOpenGL;

%% 被试基础信息
data = table();

prompt = {'被试组别', '被试编号', '性别[1 = 女, 2 = 男]', '年龄', '惯用手[1 = 左, 2 = 右]'};
title = '实验信息';
definput = {'','', '','',''};
userInput = inputdlg(prompt, title, 1, definput);

groupID = str2double(userInput{1});
subjectID = str2double(userInput{2});
gender = str2double(userInput{3});
age = str2double(userInput{4});
handedness = str2double(userInput{5});

%% 屏幕窗口
HideCursor;

screenNumber = max(Screen('Screens'));
isTestMode = false;
if isTestMode
    windowWidth = 800;
    windowHeight = 600;
    rect = [0, 0, windowWidth, windowHeight];
else
    rect = [];
end
[window, windowRect] = Screen('OpenWindow', screenNumber, [128, 128, 128], rect);
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
centerX = screenXpixels / 2;
centerY = screenYpixels / 2;

FlipIntv = Screen('GetFlipInterval', window);

%% 刺激设置
Screen('TextFont', window, 'Simsun');

stimColor = [255, 255, 255];

FixationSize = deg2pix(0.8, 16, windowRect(3), 70);

distance = deg2pix(3.5, 16, windowRect(3), 70);

shapePositions = [
    centerX, centerY - distance;
    centerX, centerY + distance;
];

labelEnglish = {'self', 'stranger'};
labelChinese = {'自我', '生人'};

shapeEnglish = {'circle', 'square'};
shapeChinese = {'圆形', '正方形'};

halfShapeSize = deg2pix(1.9, 16, windowRect(3), 70);

correctPairs_0 = {
    'square', 'self';
    'circle', 'stranger'
};

correctPairs_1 = {
    'square', 'stranger';
    'circle', 'self'
};

if mod(subjectID, 2) == 0
    correctOrder = correctPairs_0;
else
    correctOrder = correctPairs_1;
end

matchingStimuli = cell(0, 2);
nonMatchingStimuli = cell(0, 2);

for i = 1:size(correctOrder, 1)
    shape = correctOrder{i, 1};
    correctLabel = correctOrder{i, 2};
    
    matchingStimuli = [matchingStimuli; {shape, correctLabel}];
    
    otherLabel = setdiff(labelEnglish, correctLabel);
    for j = 1:length(otherLabel)
        nonMatchingStimuli = [nonMatchingStimuli; {shape, otherLabel{j}}];
    end
end

stimuli = [matchingStimuli; nonMatchingStimuli];

selfMaskImages = {'self10-1.png', 'self10-2.png', 'self10-3.png', 'self10-4.png', 'self10-5.png'};
strangerMaskImages = {'stranger10-1.png', 'stranger10-2.png', 'stranger10-3.png', 'stranger10-4.png', 'stranger10-5.png'};

conditions = [
    0, 0.03, 0.3;
    0, 0.03, 0.6;
    120, 0.03, 0.6;
    120, 0.08, 0.6;
    8, 0.1, 1.1;
    120, 0.5, 1.5;
    0, 0.1, 1.1;
    120, 0.03, 0.8;
    120, 0.08, 0.8;
];
currentCondition = conditions(groupID, :);

P = currentCondition(1);
T = currentCondition(2);
W = currentCondition(3);

%% 联结阶段
Screen('TextSize', window, 30);

label1 = correctOrder{1, 2};
label2 = correctOrder{2, 2};
shape1 = correctOrder{1, 1};
shape2 = correctOrder{2, 1};

label1Index = find(strcmp(labelEnglish, label1));
label1Chinese = labelChinese{label1Index};
shape1Index = find(strcmp(shapeEnglish, shape1));
shape1Chinese = shapeChinese{shape1Index};

label2Index = find(strcmp(labelEnglish, label2));
label2Chinese = labelChinese{label2Index};
shape2Index = find(strcmp(shapeEnglish, shape2));
shape2Chinese = shapeChinese{shape2Index};

matchKey = getMatchKey(subjectID);

if matchKey == 'f'
    mismatchKey = 'j';
else
    mismatchKey = 'f';
end

aline1 = '你好，欢迎参加本实验。';
aline2 = '接下来首先你需要记忆几何形状（圆形、正方形）和身份标签（自我、陌生人）的配对关系。';
aline3 = '这一关系会呈现60s，60s之后会自动停止。';
aline4 = '明白了任务之后请按回车键记忆关系。';
aline5 = '如果对本实验还有不清楚之处，请立即向实验员咨询。';

DrawFormattedText(window, double(aline1), 'center', centerY - 200, stimColor);
DrawFormattedText(window, double(aline2), 'center', centerY - 100, stimColor);
DrawFormattedText(window, double(aline3), 'center', centerY, stimColor);
DrawFormattedText(window, double(aline4), 'center', centerY + 100, stimColor);
DrawFormattedText(window, double(aline5), 'center', centerY + 200, stimColor);

Screen('Flip', window);

waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        pressedKey = find(keyCode);
        if any(pressedKey == KbName('return'))
            waitForSpace = false;
        end
    end
end

aline6 = '记忆以下规则60s：';
aline7 = sprintf('%s 是 %s , %s 是 %s',  label1Chinese, shape1Chinese, label2Chinese, shape2Chinese);

DrawFormattedText(window, double(aline6), 'center', centerY - 100, stimColor);
DrawFormattedText(window, double(aline7), 'center', centerY, stimColor);

Screen('Flip', window);

WaitSecs(60);

aline8 = '接下来将进入正式实验。';
aline9 = '如果未记清楚记忆关系，请立即向实验员咨询。';
aline10 = '没有问题请按回车键进入实验。';

DrawFormattedText(window, double(aline8), 'center', centerY - 100, stimColor);
DrawFormattedText(window, double(aline9), 'center', centerY, stimColor);
DrawFormattedText(window, double(aline10), 'center', centerY + 100, stimColor);

Screen('Flip', window);

waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        pressedKey = find(keyCode);
        if any(pressedKey == KbName('return'))
            waitForSpace = false;
        end
    end
end

%% 匹配阶段-练习
Screen('TextSize', window, 30);

bline1 = '屏幕中心的十字上方会呈现形状，同时下方会呈现标签。';
bline2 = '接下来的任务是判断屏幕上呈现的刺激对与刚刚学习的关系是否一致。';
bline3 = sprintf('匹配按 %s 键 , 不匹配按 %s 键', matchKey, mismatchKey);
bline4 = '有任何问题请立即向实验员咨询。';
bline5 = '没有问题请在记住配对关系后按空格键进入实验。';

DrawFormattedText(window, double(bline1), 'center', centerY - 200, stimColor);
DrawFormattedText(window, double(bline2), 'center', centerY - 100, stimColor);
DrawFormattedText(window, double(bline3), 'center', centerY, stimColor);
DrawFormattedText(window, double(bline4), 'center', centerY + 100, stimColor);
DrawFormattedText(window, double(bline5), 'center', centerY + 200, stimColor);

Screen('Flip', window);

waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        pressedKey = find(keyCode);
        if any(pressedKey == KbName('space'))
            waitForSpace = false;
        end
    end
end

fileName = sprintf('exp_group%d_subject%d.txt', groupID, subjectID);

fileID = fopen(fileName, 'a+');
if fileID == -1
    error('无法打开文件');
end

if ftell(fileID) == 0
    fprintf(fileID, 'groupID\tsubjectID\tgender\tage\thandedness\tstage\ttrialID\tP\tT\tW\tShape\tLabel\tCorrectKey\tResponse\tRT\tCorrect\n');
end

trialIndex = 1;

loopCount = P / 4;

for k = 1:loopCount

    stimuli_bin = stimuli(randperm(size(stimuli, 1)), :);

    for i = 1:length(stimuli)
    
        checkEscape();

        currentShape = stimuli_bin{i, 1};
        currentLabel = stimuli_bin{i, 2};
        
        pairingRules = getPairingRules(subjectID);
        
        disp(pairingRules);
        
        correctKey = pairingRules.(currentShape).(currentLabel);
        
        fixationDuration = 0.5;
        
        fixationFrames = round(fixationDuration / FlipIntv);

        Screen('TextSize', window, FixationSize);  

        for j = 1:fixationFrames
            DrawFormattedText(window, '+', 'center', 'center', stimColor); 
        end

        fixationFlipTime = Screen('Flip', window);

        while GetSecs() - fixationFlipTime < fixationDuration
        end
    
        checkEscape(); 
        
        stimuliFrames = round(T / FlipIntv);

        for j = 1:stimuliFrames
            Screen('TextSize', window, FixationSize); 
            DrawFormattedText(window, '+', 'center', 'center', stimColor);
        
            left = shapePositions(1, 1) - halfShapeSize;
            right = shapePositions(1, 1) + halfShapeSize;
            top = shapePositions(1, 2) - halfShapeSize;
            bottom = shapePositions(1, 2) + halfShapeSize;
            
            switch currentShape
                case 'circle'
                    Screen('FillOval', window, stimColor, [left, top, right, bottom]);
                case 'square'
                    Screen('FillRect', window, stimColor, [left, top, right, bottom]);
            end
            
            Screen('TextSize', window, 90);
            labelIndex = find(strcmp(labelEnglish, currentLabel));
            DrawFormattedText(window, double(labelChinese{labelIndex}), 'center', shapePositions(2, 2), stimColor);
        end
      
        stimulusFlipTime = Screen('Flip', window);
        
        while GetSecs() - stimulusFlipTime < T
        end

        checkEscape(); 

        if strcmp(currentLabel, 'stranger')
            selectedMaskImage = strangerMaskImages{randi(numel(strangerMaskImages))};
            maskImage = imread(selectedMaskImage);
        elseif strcmp(currentLabel, 'self')
            selectedMaskImage = selfMaskImages{randi(numel(selfMaskImages))};
            maskImage = imread(selectedMaskImage);
        end
        
        [maskHeight, maskWidth, ~] = size(maskImage);

        maskRect = [0 0 maskWidth maskHeight];
        maskRect = CenterRectOnPoint(maskRect, centerX, centerY + distance);
        
        maskImageTexture = Screen('MakeTexture', window, maskImage);

        maskDuration = 0.2;
        Screen('DrawTexture', window, maskImageTexture, [], maskRect);
        maskFlipTime = Screen('Flip', window);
        while GetSecs() - maskFlipTime < maskDuration
        end

        checkEscape(); 
    
        responseFlipTime = Screen('Flip', window); 
    
        response = NaN;
        responseTime = NaN;
        
        while GetSecs() - responseFlipTime < W - maskDuration
            [keyIsDown, ~, keyCode] = KbCheck;
            
            if keyIsDown
                pressedKeys = find(keyCode);
                
                if ~isempty(pressedKeys)
                    pressedKey = KbName(pressedKeys(1));
                    
                    if ismember(pressedKey, {'f', 'j'})
                        response = pressedKey;
                        responseTime = GetSecs() - stimulusFlipTime;
                        break;
                    else
                        continue;
                    end
                end
            end
        end

        checkEscape(); 

        data{trialIndex, 1} = groupID;
        data{trialIndex, 2} = subjectID;
        data{trialIndex, 3} = gender;
        data{trialIndex, 4} = age;
        data{trialIndex, 5} = handedness;
        data{trialIndex, 6} = "practice";
        data{trialIndex, 7} = trialIndex;
        data{trialIndex, 8} = P;
        data{trialIndex, 9} = T;
        data{trialIndex, 10} = W;
        data{trialIndex, 11} = currentShape;
        data{trialIndex, 12} = string(currentLabel);
        data{trialIndex, 13} = char(correctKey);
        data{trialIndex, 14} = char(response);
        data{trialIndex, 15} = responseTime;
    
        if isempty(response)
            data{trialIndex, 16} = NaN;
        elseif strcmp(response, correctKey)
            data{trialIndex, 16} = 1;
        else
            data{trialIndex, 16} = 0;
        end
    
        if isnan(data{trialIndex, 15})
            feedbackText = double('过慢！');
            feedbackColor = [255, 0, 0];
        elseif data{trialIndex, 15} < 0.15
            feedbackText = double('过快！');
            feedbackColor = [255, 0, 0];
        elseif data{trialIndex, 16} == 1
            feedbackText = double('正确！');
            feedbackColor = [0, 255, 0];
        elseif data{trialIndex, 16} == 0
            feedbackText = double('错误！');
            feedbackColor = [255, 0, 0];
        end
        
        DrawFormattedText(window, feedbackText, 'center', 'center', feedbackColor); 
        feedbackFlipTime = Screen('Flip', window);
        WaitSecs(0.5);

        checkEscape(); 

        fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3f\t%d\n', ...
            data{trialIndex, 1}, data{trialIndex, 2}, data{trialIndex, 3}, data{trialIndex, 4}, ...
            data{trialIndex, 5}, data{trialIndex, 6}, data{trialIndex, 7}, data{trialIndex, 8}, ...
            data{trialIndex, 9}, data{trialIndex, 10}, data{trialIndex, 11}, data{trialIndex, 12}, ...
            data{trialIndex, 13}, data{trialIndex, 14}, data{trialIndex, 15}, data{trialIndex, 16});

        trialIndex = trialIndex + 1;

    end
end

%% 练习与正式实验之间的休息
Screen('TextSize', window, 30);

restMessage1 = '已完成练习，';
restMessage2 = '休息后按空格键进入正式实验。';
DrawFormattedText(window, double(restMessage1), 'center', 'center', stimColor);
DrawFormattedText(window, double(restMessage2), 'center', centerY + 100, stimColor);
Screen('Flip', window);

waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        pressedKey = find(keyCode);
        if any(pressedKey == KbName('space'))
            waitForSpace = false;
        end
    end
end

%% 匹配阶段-正式实验
block = 10;
blockTrials = 52;

for b = 1:block

    correctCount = 0;

    responseTimes = [];

    for y = 1:(blockTrials/length(stimuli))
    
        stimuli_bin = stimuli(randperm(size(stimuli, 1)), :);

        for i = 1:length(stimuli)

            checkEscape();

            currentShape = stimuli_bin{i, 1};
            currentLabel = stimuli_bin{i, 2};
            
            pairingRules = getPairingRules(subjectID);
            
            disp(pairingRules);
            
            correctKey = pairingRules.(currentShape).(currentLabel);
            
            fixationDuration = 0.5;
            
            fixationFrames = round(fixationDuration / FlipIntv);
    
            Screen('TextSize', window, FixationSize); 
            DrawFormattedText(window, '+', 'center', 'center', stimColor);
    
            for j = 1:fixationFrames
                DrawFormattedText(window, '+', 'center', 'center', stimColor); 
            end
    
            fixationFlipTime = Screen('Flip', window);
    
            while GetSecs() - fixationFlipTime < fixationDuration
            end

            checkEscape(); 
        
            stimuliFrames = round(T / FlipIntv);
    
            for j = 1:stimuliFrames

                Screen('TextSize', window, FixationSize); 
                DrawFormattedText(window, '+', 'center', 'center', stimColor); 
            
                left = shapePositions(1, 1) - halfShapeSize;
                right = shapePositions(1, 1) + halfShapeSize;
                top = shapePositions(1, 2) - halfShapeSize;
                bottom = shapePositions(1, 2) + halfShapeSize;
                
                switch currentShape
                    case 'circle'
                        Screen('FillOval', window, stimColor, [left, top, right, bottom]);
                    case 'square'
                        Screen('FillRect', window, stimColor, [left, top, right, bottom]);
                end
                
                Screen('TextSize', window, 90);
                labelIndex = find(strcmp(labelEnglish, currentLabel));
                DrawFormattedText(window, double(labelChinese{labelIndex}), 'center', shapePositions(2, 2), stimColor);
            end
          
            stimulusFlipTime = Screen('Flip', window);
            
            while GetSecs() - stimulusFlipTime < T
            end

            checkEscape(); 

            if strcmp(currentLabel, 'stranger')
                selectedMaskImage = strangerMaskImages{randi(numel(strangerMaskImages))};
                maskImage = imread(selectedMaskImage);
            elseif strcmp(currentLabel, 'self')
                selectedMaskImage = selfMaskImages{randi(numel(selfMaskImages))};
                maskImage = imread(selectedMaskImage);
            end
            
            [maskHeight, maskWidth, ~] = size(maskImage);
    
            maskRect = [0 0 maskWidth maskHeight];
            maskRect = CenterRectOnPoint(maskRect, centerX, centerY + distance);
            
            maskImageTexture = Screen('MakeTexture', window, maskImage);
    
            maskDuration = 0.2;
            Screen('DrawTexture', window, maskImageTexture, [], maskRect);
            maskFlipTime = Screen('Flip', window);
            while GetSecs() - maskFlipTime < maskDuration
            end
    
            checkEscape(); 
        
            responseFlipTime = Screen('Flip', window); 
        
            response = NaN;
            responseTime = NaN;
            
            while GetSecs() - responseFlipTime < W - maskDuration
                [keyIsDown, ~, keyCode] = KbCheck;
                
                if keyIsDown
                    pressedKeys = find(keyCode);
                    
                    if ~isempty(pressedKeys)
                        pressedKey = KbName(pressedKeys(1));
                        
                        if ismember(pressedKey, {'f', 'j'})
                            response = pressedKey;
                            responseTime = GetSecs() - stimulusFlipTime;
                            break;
                        else
                            continue;
                        end
                    end
                end
            end

            checkEscape(); 
    
            data{trialIndex, 1} = groupID;
            data{trialIndex, 2} = subjectID;
            data{trialIndex, 3} = gender;
            data{trialIndex, 4} = age;
            data{trialIndex, 5} = handedness;
            data{trialIndex, 6} = "formal";
            data{trialIndex, 7} = trialIndex;
            data{trialIndex, 8} = P;
            data{trialIndex, 9} = T;
            data{trialIndex, 10} = W;
            data{trialIndex, 11} = currentShape;
            data{trialIndex, 12} = string(currentLabel);
            data{trialIndex, 13} = char(correctKey);
            data{trialIndex, 14} = char(response);
            data{trialIndex, 15} = responseTime;
        
            if isempty(response)
                data{trialIndex, 16} = NaN;
            elseif strcmp(response, correctKey)
                data{trialIndex, 16} = 1;
                correctCount = correctCount + 1;
            else
                data{trialIndex, 16} = 0;
            end

            if ~isnan(responseTime)
                responseTimes = [responseTimes, responseTime];
            end
            
            fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3f\t%d\n', ...
                data{trialIndex, 1}, data{trialIndex, 2}, data{trialIndex, 3}, data{trialIndex, 4}, ...
                data{trialIndex, 5}, data{trialIndex, 6}, data{trialIndex, 7}, data{trialIndex, 8}, ...
                data{trialIndex, 9}, data{trialIndex, 10}, data{trialIndex, 11}, data{trialIndex, 12}, ...
                data{trialIndex, 13}, data{trialIndex, 14}, data{trialIndex, 15}, data{trialIndex, 16});
    
            trialIndex = trialIndex + 1;
    
        end
    end

    accuracy = correctCount / blockTrials;
    avgRT = mean(responseTimes);  
    
    Screen('TextSize', window, 30); 
    
    message1 = sprintf('正确率: %.2f%% ', accuracy * 100);
    message2 = sprintf('平均反应时: %.3f 秒', avgRT);
    message3 = sprintf('按回车键进入休息');

    DrawFormattedText(window, double(message1), 'center', centerY - 150, stimColor);
    DrawFormattedText(window, double(message2), 'center', centerY, stimColor);
    DrawFormattedText(window, double(message3), 'center', centerY + 150, stimColor);
    
    Screen('Flip', window);

    waitForSpace = true;
    while waitForSpace
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            pressedKey = find(keyCode);
            if any(pressedKey == KbName('return'))
                waitForSpace = false;
            end
        end
    end

    Screen('TextSize', window, 30); 

    restMessage = double('休息中，请按空格键继续。');
    DrawFormattedText(window, restMessage, 'center', 'center', stimColor);
    Screen('Flip', window);
    
    waitForSpace = true;
    while waitForSpace
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            pressedKey = find(keyCode);
            if any(pressedKey == KbName('space'))
                waitForSpace = false;
            end
        end
    end
end

%% 结束
dline1 = '已完成所有实验，';
dline2 = '感谢您的参与。';
dline3 = '按回车键结束实验。';

DrawFormattedText(window, double(dline1), 'center', centerY - 100, stimColor);
DrawFormattedText(window, double(dline2), 'center', centerY, stimColor);
DrawFormattedText(window, double(dline3), 'center', centerY + 100, stimColor);

Screen('Flip', window);

waitForSpace = true;
while waitForSpace
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown
        pressedKey = find(keyCode);
        if any(pressedKey == KbName('return'))
            waitForSpace = false;
        end
    end
end

%% 保存数据
columnNames = {'groupID', 'subjectID', 'gender', 'age', 'handedness', 'stage', 'trialID', 'P', 'T', 'W', 'Shape', 'Label', 'CorrectKey', 'Response','RT', 'Correct'};
data.Properties.VariableNames = columnNames;
filename = ['EXP_data_group' num2str(groupID) '_' num2str(subjectID) '.csv'];
writetable(data, filename);
disp(['Data has been saved to ', filename]);

Screen('CloseAll');

%% Functions
function pairingRules = getPairingRules(subjectID)
    modResult = mod(subjectID, 4);
    
    rules = { ...
        'square', struct('self', 'f', 'stranger', 'j'), 'circle', struct('self', 'j', 'stranger', 'f'); ...
        'square', struct('self', 'j', 'stranger', 'f'), 'circle', struct('self', 'f', 'stranger', 'j'); ...
        'square', struct('self', 'j', 'stranger', 'f'), 'circle', struct('self', 'f', 'stranger', 'j'); ...
        'square', struct('self', 'f', 'stranger', 'j'), 'circle', struct('self', 'j', 'stranger', 'f')  ... 
    };

    pairingRules = struct(rules{modResult + 1, :});
end

function key = getMatchKey(subjectID)
    if subjectID <= 0 || mod(subjectID, 1) ~= 0
        error('subjectID 必须是正整数');
    end
    
    matchKeys = {'f', 'j', 'j', 'f'};
    
    index = mod(subjectID - 1, length(matchKeys)) + 1;
    key = matchKeys{index};
end

function pixs = deg2pix(degree,inch,pwidth,vdist) 

screenWidth = inch*2.54/sqrt(1+11.81/15.75);
pix=screenWidth/pwidth; 
pixs = round(2*tan((degree/2)*pi/180) * vdist / pix); 

end

function checkEscape()
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(KbName('esc'))
        sca;
        error('实验被用户终止（按下 Esc 键）。');
    end
end
