#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

__global__ void softmax_kernel_v1(float *input, float *output, int M, int N);
void softmax_gpu(float *input, float *output, int M, int N);