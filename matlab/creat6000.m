
%将数据保存为文本文件，方便vs调用

clc
clear all
close all

addpath('.\data\6000\');

load inputn_test.mat
load inputn_train.mat
load output_test.mat
load output_train.mat
load W10.mat
load W21.mat

saveMatToText(inputn_test', '.\data\6000\inputn_test.txt');
saveMatToText(inputn_train', '.\data\6000\inputn_train.txt');
saveMatToText(output_test', '.\data\6000\output_test.txt');
saveMatToText(output_train', '.\data\6000\output_train.txt');
saveMatToText(W10, '.\data\6000\W10.txt');
saveMatToText(W21, '.\data\6000\W21.txt');

