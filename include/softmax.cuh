#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <softmax_v1.cuh>
#include <config.cuh>
#include "baseline.cuh"

void softmax_gpu(float *input, float *output, int M, int N);