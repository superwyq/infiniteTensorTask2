#include "softmax.cuh"
#include "baseline.cuh"

int main()
{
    cudaSetDevice(0);
    cudaFree(0); //初始化cuda，防止第一次调用函数时间过长
    float *input, *output, *cpu_output;
    int M = 1024;
    int N = 1024;

    input = (float *)malloc(M * N * sizeof(float));
    output = (float *)malloc(M * N * sizeof(float));
    cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++)
    {
        input[i] = (i % 10);
    }
    softmax_gpu(input, output, M, N);
    cpu_softmax(input, cpu_output, M, N);
    float max_diff = 0.0;
    for(int i = 0; i < M * N; i++)
    {
        max_diff = fmaxf(max_diff, fabs(output[i] - cpu_output[i]));
    }
    printf("Max diff: %f\n", max_diff);
    printf("\n");
    
    free(input);
    free(output);
    free(cpu_output);
    return 0;
    
}