#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA 错误检查
void check_cuda(cudaError_t err, const char* step)
{
    if (err != cudaSuccess)
    {
        printf("%s failed: %s\n",
               step, cudaGetErrorString(err));
        std::exit(1);
    }
}

// cuBLAS 错误检查
void check_cublas(cublasStatus_t status, const char* step)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("%s failed: cuBLAS status=%d\n",
               step, static_cast<int>(status));
        std::exit(1);
    }
}

int main(int argc, char* argv[])
{
    int M = 3;
    int K = 5;
    int N = 4;

    if (argc == 4)
    {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);

        if (M <= 0 || K <= 0 || N <= 0)
        {
            printf("Error: M, K, N must be positive\n");
            return 1;
        }
    }
    else if (argc != 1)
    {
        printf("Usage: %s [M K N]\n", argv[0]);
        return 1;
    }

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu(M * N, 0.0f);

    // 初始化矩阵
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (i % 7 - 3) * 0.25f;

    for (int i = 0; i < K * N; ++i)
        h_B[i] = (i % 9 - 4) * 0.125f;

    // CPU Reference
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
                sum += h_A[row * K + k]
                     * h_B[k * N + col];

            h_C_cpu[row * N + col] = sum;
        }
    }

    // GPU 显存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    check_cuda(cudaMalloc((void**)&d_A, bytes_A), "malloc A");
    check_cuda(cudaMalloc((void**)&d_B, bytes_B), "malloc B");
    check_cuda(cudaMalloc((void**)&d_C, bytes_C), "malloc C");

    check_cuda(cudaMemcpy(d_A, h_A.data(), bytes_A,
                          cudaMemcpyHostToDevice), "copy A");

    check_cuda(cudaMemcpy(d_B, h_B.data(), bytes_B,
                          cudaMemcpyHostToDevice), "copy B");

    // 创建 cuBLAS Handle
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 计算 Row-major C = A * B
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N
        ),
        "cublasSgemm"
    );

    check_cuda(cudaDeviceSynchronize(), "SGEMM sync");

    check_cuda(cudaMemcpy(h_C_gpu.data(), d_C, bytes_C,
                          cudaMemcpyDeviceToHost), "copy C");

    // 验证结果
    bool pass = true;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float cpu = h_C_cpu[i];
        float gpu = h_C_gpu[i];
        float diff = std::fabs(cpu - gpu);

        if (diff > max_diff)
            max_diff = diff;

        if (!std::isfinite(gpu) ||
            diff > 1e-5f + 1e-4f * std::fabs(cpu))
        {
            pass = false;
        }
    }

    printf("Shape: M=%d K=%d N=%d\n", M, K, N);
    printf("Max Difference: %.8f\n", max_diff);
    printf("cuBLAS Test: %s\n", pass ? "PASS" : "FAIL");


    // ========================================
// V2.6.3 cuBLAS SGEMM Benchmark
// ========================================

if (pass)
{
    const int warmup = 20;
    const int repeat = 1000;

    cudaEvent_t start, stop;

    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    // 1. 预热
    for (int i = 0; i < warmup; ++i)
    {
        check_cublas(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N
            ),
            "cuBLAS warmup"
        );
    }

    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    // 2. 开始计时
    check_cuda(cudaEventRecord(start), "record start");

    // 3. 重复执行
    for (int i = 0; i < repeat; ++i)
    {
        check_cublas(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N
            ),
            "cuBLAS benchmark"
        );
    }

    // 4. 结束计时
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    // 5. 计算平均耗时
    float total_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&total_ms, start, stop),
        "elapsed time"
    );

    float avg_ms = total_ms / repeat;

    // 6. 计算 GFLOPS
    double flops = 2.0 * M * N * K;

    double gflops = flops / (avg_ms * 1e6);

    printf("\n===== cuBLAS Benchmark =====\n");

    printf("Warmup: %d, Repeat: %d\n", warmup, repeat);
    printf("cuBLAS Latency: %.6f ms\n", avg_ms);
    printf("cuBLAS Performance: %.3f GFLOPS\n", gflops);

    // 7. 释放 Event
    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");
}
    check_cublas(cublasDestroy(handle), "cublasDestroy");

    check_cuda(cudaFree(d_A), "free A");
    check_cuda(cudaFree(d_B), "free B");
    check_cuda(cudaFree(d_C), "free C");

    return pass ? 0 : 1;
}
