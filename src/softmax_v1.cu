#include "softmax_v1.cuh"
#include "baseline.cuh"
#include "config.cuh"


__global__ void softmax_kernel_v1(float *input, float *output, int M, int N)
{
    /// @brief: softmax_kernel_v1
    /// @param: float *input, float *output, int M, int N
    /// @return: void
    /// @note:
    ///     最基础的实现，没有进行优化
    ///     求最大值和求和的时候需要进行2*M*N次访存
    ///     求softmax的时候需要进行2*M*N次访存
    ///     总共需要进行4*M*N次访存，有效带宽为：4*M*N*sizeof(float) / time
    int row = blockIdx.x; //每个block负责一行数据
    int col = threadIdx.x; //当下thread负责的数据
    int base = row * N; //每行数据的起始位置
    if(col >= N || base + col >= M * N) //如果线程id超出了数据列数，或者全局id超出了数据总数，直接返回
        return;
    
    __shared__ float sDataSum[BLOCK_DIM]; //用于规约求和
    __shared__ float sDataMax[BLOCK_DIM]; //用于规约求最大值

    float tempMax = -INFINITY; //线程局部变量，用于存储每个线程负责的数据中的最大值，局部变量更快一些
    for(int i = col; i < N; i += BLOCK_DIM) //将数据加载到共享内存，每个thread负责第col + n*BLOCK_DIM个数据
    {
        tempMax = fmaxf(tempMax, input[base + i]);
    }
    sDataMax[col] = tempMax;
    __syncthreads();
    for(int i = BLOCK_DIM / 2; i > 0; i >>= 1)
    {
        if(col < i) //每次只用前一半的线程参与规约
        {
            sDataMax[col] = fmaxf(sDataMax[col], sDataMax[col + i]);
        }
    }
    __shared__ float globalMax; //全局最大值
    if(col == 0) //每个block的第一个线程负责将自己的最大值存储到全局变量中
    {
        globalMax = sDataMax[0];
    }
    __syncthreads();


    float tempSum = 0; //同上，用于存储每个线程负责的数据的和
    for(int i = col; i < N; i += BLOCK_DIM) //将数据加载到共享内存，每个thread负责第col + n*BLOCK_DIM个数据
    {
        tempSum += expf(input[base + i] - globalMax);
    }
    sDataSum[col] = tempSum;
    __syncthreads();

    for(int i = BLOCK_DIM / 2; i > 0; i >>= 1)
    {
        if(col < i) //每次只用前一半的线程参与规约
        {
            sDataSum[col] += sDataSum[col + i];
        }
        __syncthreads();
    }
    
    __shared__ float globalSum;
    if(col == 0){ //全局和是所有线程规约后的第一个元素
        globalSum = sDataSum[0];
    }
    __syncthreads();
    for(int i = col; i < N; i+= BLOCK_DIM) //和上面一样，每个线程负责第col + n*BLOCK_DIM个数据
    {
        output[base + i] = expf(input[base + i] - globalMax) / globalSum;
    }
}


