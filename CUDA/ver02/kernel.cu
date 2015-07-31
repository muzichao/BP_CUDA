#include "Parameter.h"
#include "ReadSaveImage.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "BPCUDAmain.h"

using std::cout;
using std::endl;
using std::string;

int main()
{
	string strInputTrain = "D:\\Document\\vidpic\\CUDA\\BP\\data\\6000\\inputn_train.txt";
	string strInputTest = "D:\\Document\\vidpic\\CUDA\\BP\\data\\6000\\inputn_test.txt";
	string strOutputTrain = "D:\\Document\\vidpic\\CUDA\\BP\\data\\6000\\output_train.txt";
	string strOutputTest = "D:\\Document\\vidpic\\CUDA\\BP\\data\\6000\\output_test.txt";

	float *inputTrain = (float*)malloc(trainNum * inLayout * sizeof(float));
	float *inputTest = (float*)malloc(testNum * inLayout * sizeof(float));
	float *outputTrain = (float*)malloc(trainNum * outLayout * sizeof(float));
	float *outputTest = (float*)malloc(testNum * outLayout * sizeof(float));

	ReadFile(inputTrain, strInputTrain, trainNum * inLayout);
	ReadFile(inputTest, strInputTest, testNum * inLayout);
	ReadFile(outputTrain, strOutputTrain, trainNum * outLayout);
	ReadFile(outputTest, strOutputTest, testNum * outLayout);

	BpMain(inputTrain, inputTest, outputTrain, outputTest);

	free(inputTrain);
	free(inputTest);
	free(outputTrain);
	free(outputTest);

	cudaDeviceReset();
	return 0;
}