#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void CalTheoreticalBandWidth()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);               // 获取设备上的GPU个数

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);    // 获取当前GPU的相关属性


        std::cout << "GPU: " << i << std::endl;
        std::cout << "Name: " << deviceProp.name << std::endl;
        std::cout << "Bit width: " << deviceProp.memoryBusWidth << " bit" << std::endl;
        std::cout << "Memory clock rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;

        int bw = static_cast<size_t>(deviceProp.memoryClockRate) * 1000 * deviceProp.memoryBusWidth / 8 * 2 / 1000000000;
        
        std::cout << "Theoretical band width = " << bw << " GB/s" << std::endl;
    }
}

int main()
{
    CalTheoreticalBandWidth();
    return 0;
}
