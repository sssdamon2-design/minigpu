
#include <cuda_runtime.h>

// 复用 V2 的 16x16 Shared Memory Tiling
__global__ void gemm_tiled16(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
{
    constexpr int TILE = 16;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] =
            (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        Bs[ty][tx] =
            (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 将 CUDA Kernel 暴露给 C++ 层
void launch_gemm_tiled16(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_tiled16<<<grid, block, 0, stream>>>(
        A, B, C, M, K, N);
}
