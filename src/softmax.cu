#include "softmax.cuh"

void softmax_gpu(float *input, float *output, int M, int N)
{
    double st, ela;
    st = get_walltime();

    // auto start_cpu = std::chrono::high_resolution_clock::now();
    dim3 block_dim(BLOCK_DIM, 1);
    dim3 grid_dim(M, 1);
    float* d_input,*d_output;
    cudaMalloc((void**)&d_input, M * N * sizeof(float));
    cudaMalloc((void**)&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    float ker_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    KERNEL_VERSION<<<grid_dim, block_dim>>>(d_input, d_output, M, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);

    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    // auto end_cpu = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // std::cout << "total use time: " << duration.count() * 1.0 << std::endl;
    ela = get_walltime() - st;
    std::cout << "********** softmax_v1**********" <<std::endl; 
    std::cout << "Data size: " << M << " * " << N << std::endl;
    std::cout << "use time: " << ela << std::endl;
    std::cout << "kernel time: " << ker_time / 1000. << std::endl;
    printf("Bandwidth: %f GB/s\n", 4 * M * N * sizeof(float) * 1000.0 / (1<<30) / ker_time );
}