

clc
clear all
close all

addpath('.\data');

load inputn_test.mat
load inputn_train.mat
load output_test.mat
load output_train.mat
load output_test_class.mat

W10 = load('W10result.txt');
W21 = load('W21result.txt');
W10 = reshape(W10, [401, 32])';
W21 = reshape(W21, [32, 10])';

% 信号
classNum = 10;
trainNum = 4160;
testNum = 840;

% break
% 隐藏层输出 
hOut = 1 ./ (1 + exp(- W10 * inputn_test));

% 输出层输出
fore = W21 * hOut;

%% 结果分析
% 根据网络输出找出数据属于哪类
[output_fore, ~] = find(bsxfun(@eq, fore, max(fore)) ~= 0);

%BP网络预测误差
error = output_fore' - output_test_class;

%% 计算正确率
% 找出每类判断错误的测试样本数
kError = zeros(1, classNum);  
outPutError = bsxfun(@and, output_test, error);
[indexError, ~] = find(outPutError ~= 0);

for class = 1:classNum
    kError(class) = sum(indexError == class);
end

% 找出每类的总测试样本数
kReal = zeros(1, classNum);
[indexRight, ~] = find(output_test ~= 0);
for class = 1:classNum
    kReal(class) = sum(indexRight == class);
end

% 正确率
rightridio = (kReal-kError) ./ kReal
meanRightRidio = mean(rightridio)
%}

%% 画图

% 画出误差图
figure
stem(error, '.')
title('BP网络分类误差', 'fontsize',12)
xlabel('信号', 'fontsize',12)
ylabel('分类误差', 'fontsize',12)


