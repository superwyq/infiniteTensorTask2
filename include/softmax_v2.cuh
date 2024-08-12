#pragma once
#include <iostream>
#include <cuda_runtime.h>

__global__ void softmax_kernel_v2(float *input, float *output, int M, int N);