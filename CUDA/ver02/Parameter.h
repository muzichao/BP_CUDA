#ifndef PARAMETER_H
#define PARAMETER_H

#include <iostream>

#define classNum 10 // 类别数
#define trainNum 4160 // 训练样本数
#define testNum 840 // 测试样本数

#define inLayout 401 // 输入层数
#define hideLayout 32 // 中间层数
#define outLayout classNum // 输出层数

#define initWeightMax sqrt(6.0f / (inLayout + hideLayout)) // 初始权重最大值

#define eta (0.2f) // 学习率

#define iterMax 50 // 最大迭代此时

#define batchNum 32 // 批处理样本数

#define BLOCKSIZE 16 // 线程块维度
#define BLOCKSIZE_32 32 // 线程块维度

#endif //PARAMETER_H