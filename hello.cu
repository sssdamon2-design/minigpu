#include <cstdio>
#include <cuda_runtime.h>

// GPU 上执行的函数
__global__ void hello_kernel()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    // 启动 1 个 Block，每个 Block 有 4 个 Thread
    hello_kernel<<<1, 4>>>();

    // 等待 GPU 执行完成，并检查错误
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
