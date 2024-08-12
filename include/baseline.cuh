#pragma once
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include "config.cuh"

double get_walltime();
__global__ void softmax(float *input, float *output, int M, int N);
void cpu_softmax(float *cpu_input, float *cpu_output, int M, int N);