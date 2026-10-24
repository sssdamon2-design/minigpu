
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

// ========================================
// 1. 错误检查与参数解析
// ========================================

void check_cuda(cudaError_t err, const char* step)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n",
                     step, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t status, const char* step)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed: cuBLAS status=%d\n",
                     step, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

bool read_positive_int(const char* text, int& result)
{
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > INT_MAX)
        return false;

    result = static_cast<int>(value);
    return true;
}

// ========================================
// 2. V2.1 Naive GEMM
// ========================================

__global__ void gemm_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= M * N)
        return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

// ========================================
// 3. V2.2 Shared Memory Tiled GEMM
// ========================================

template<int TILE>
__global__ void gemm_tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    for (int t = 0; t < (K - 1) / TILE + 1; ++t)
    {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] =
            (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;

        Bs[ty][tx] =
            (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ========================================
// 4. V2.4 Register Blocking 1x2
// 同一份 Kernel 支持 TILE=16 和 TILE=32
// ========================================

template<int TILE>
__global__ void gemm_register2_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;

    int col0 = blockIdx.x * (2 * TILE) + tx;
    int col1 = col0 + TILE;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][2 * TILE];

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int t = 0; t < (K - 1) / TILE + 1; ++t)
    {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] =
            (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;

        Bs[ty][tx] =
            (bRow < K && col0 < N)
            ? B[bRow * N + col0] : 0.0f;

        Bs[ty][tx + TILE] =
            (bRow < K && col1 < N)
            ? B[bRow * N + col1] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
        {
            float a = As[ty][k];

            sum0 += a * Bs[k][tx];
            sum1 += a * Bs[k][tx + TILE];
        }

        __syncthreads();
    }

    if (row < M && col0 < N)
        C[row * N + col0] = sum0;

    if (row < M && col1 < N)
        C[row * N + col1] = sum1;
}

// ========================================
// 5. CPU Reference
// ========================================

void gemm_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum +=
                    A[static_cast<size_t>(row) * K + k] *
                    B[static_cast<size_t>(k) * N + col];
            }

            C[static_cast<size_t>(row) * N + col] = sum;
        }
    }
}

// ========================================
// 6. 统一正确性验证
// ========================================

bool verify(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& output)
{
    constexpr float atol = 1e-5f;
    constexpr float rtol = 1e-4f;

    float max_diff = 0.0f;

    size_t bad = 0;
    size_t first_bad = 0;

    for (size_t i = 0; i < reference.size(); ++i)
    {
        float ref = reference[i];
        float got = output[i];
        float diff = std::fabs(ref - got);

        if (std::isfinite(diff) && diff > max_diff)
            max_diff = diff;

        if (!std::isfinite(ref) ||
            !std::isfinite(got) ||
            diff > atol + rtol * std::fabs(ref))
        {
            if (bad == 0)
                first_bad = i;

            ++bad;
        }
    }

    bool ok = (bad == 0);

    std::printf(
        "%-16s %s  max_diff=%.8g  bad=%zu\n",
        name,
        ok ? "PASS" : "FAIL",
        max_diff,
        bad
    );

    if (!ok)
    {
        std::printf(
            "    first error at [%zu]: CPU=%.8g, GPU=%.8g\n",
            first_bad,
            reference[first_bad],
            output[first_bad]
        );
    }

    return ok;
}

// ========================================
// 7. 统一 CUDA Event Benchmark
// 预热20次，测量1000次，3轮取中位数
// ========================================

template<class Launch>
double benchmark_events(
    Launch launch,
    cudaEvent_t start,
    cudaEvent_t stop,
    int warmup,
    int repeat,
    int trials)
{
    // 预热
    for (int i = 0; i < warmup; ++i)
        launch();

    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<double> samples;

    for (int trial = 0; trial < trials; ++trial)
    {
        check_cuda(cudaEventRecord(start), "event start");

        for (int i = 0; i < repeat; ++i)
            launch();

        check_cuda(cudaGetLastError(), "benchmark launch");

        check_cuda(cudaEventRecord(stop), "event stop");
        check_cuda(cudaEventSynchronize(stop), "event synchronize");

        float total_ms = 0.0f;

        check_cuda(
            cudaEventElapsedTime(&total_ms, start, stop),
            "elapsed time"
        );

        samples.push_back(
            static_cast<double>(total_ms) / repeat
        );
    }

    // 取中位数
    std::sort(samples.begin(), samples.end());

    return samples[samples.size() / 2];
}

// ========================================
// 8. Main：统一 GEMM 测试入口
// ========================================

int main(int argc, char* argv[])
{
    int M = 512;
    int K = 512;
    int N = 512;

    if (argc == 4)
    {
        if (!read_positive_int(argv[1], M) ||
            !read_positive_int(argv[2], K) ||
            !read_positive_int(argv[3], N))
        {
            std::fprintf(stderr,
                         "M, K, N must be positive integers.\n");
            return 1;
        }
    }
    else if (argc != 1)
    {
        std::fprintf(stderr, "Usage: %s [M K N]\n", argv[0]);
        return 1;
    }

    // 当前 Kernel 使用 int 进行索引
    long long mk = 1LL * M * K;
    long long kn = 1LL * K * N;
    long long mn = 1LL * M * N;

    if (mk > INT_MAX || kn > INT_MAX || mn > INT_MAX)
    {
        std::fprintf(stderr,
                     "Matrix too large for current int indexing.\n");
        return 1;
    }

    const size_t countA = static_cast<size_t>(mk);
    const size_t countB = static_cast<size_t>(kn);
    const size_t countC = static_cast<size_t>(mn);

    const size_t bytesA = countA * sizeof(float);
    const size_t bytesB = countB * sizeof(float);
    const size_t bytesC = countC * sizeof(float);

    // 创建 CPU 矩阵
    std::vector<float> h_A(countA);
    std::vector<float> h_B(countB);
    std::vector<float> h_ref(countC, 0.0f);
    std::vector<float> h_out(countC);

    // 与之前 V2 测试保持相同输入
    for (size_t i = 0; i < countA; ++i)
        h_A[i] = (static_cast<int>(i % 7) - 3) * 0.25f;

    for (size_t i = 0; i < countB; ++i)
        h_B[i] = (static_cast<int>(i % 9) - 4) * 0.125f;

    // CPU Reference
    gemm_cpu(h_A, h_B, h_ref, M, K, N);

    // 申请 GPU 显存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_A), bytesA),
        "malloc A"
    );

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_B), bytesB),
        "malloc B"
    );

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_C), bytesC),
        "malloc C"
    );

    check_cuda(
        cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice),
        "copy A"
    );

    check_cuda(
        cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice),
        "copy B"
    );

    // 创建 cuBLAS 上下文
    cublasHandle_t handle;

    check_cublas(cublasCreate(&handle), "cublasCreate");

    check_cublas(
        cublasSetStream(handle, 0),
        "cublasSetStream"
    );

    check_cublas(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
        "cublasSetPointerMode"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ========================================
    // 9. 配置不同 Kernel 的线程与 Grid
    // ========================================

    // V2.1 Naive
    const int naive_threads = 128;

    int naive_blocks =
        static_cast<int>((mn - 1) / naive_threads + 1);

    // V2.2 Tiled16
    dim3 tiled_block(16, 16);

    dim3 tiled_grid(
        (N - 1) / 16 + 1,
        (M - 1) / 16 + 1
    );

    // V2.4 Register2 TILE=16
    dim3 reg16_block(16, 16);

    dim3 reg16_grid(
        (N - 1) / 32 + 1,
        (M - 1) / 16 + 1
    );

    // V2.4 Register2 TILE=32
    dim3 reg32_block(32, 32);

    dim3 reg32_grid(
        (N - 1) / 64 + 1,
        (M - 1) / 32 + 1
    );

    // ========================================
    // 10. 定义不同 Kernel 的启动函数
    // ========================================

    auto run_naive = [&]()
    {
        gemm_naive<<<naive_blocks, naive_threads>>>(
            d_A, d_B, d_C, M, K, N
        );
    };

    auto run_tiled = [&]()
    {
        gemm_tiled_kernel<16>
            <<<tiled_grid, tiled_block>>>(
                d_A, d_B, d_C, M, K, N
            );
    };

    auto run_reg16 = [&]()
    {
        gemm_register2_kernel<16>
            <<<reg16_grid, reg16_block>>>(
                d_A, d_B, d_C, M, K, N
            );
    };

    auto run_reg32 = [&]()
    {
        gemm_register2_kernel<32>
            <<<reg32_grid, reg32_block>>>(
                d_A, d_B, d_C, M, K, N
            );
    };

    auto run_cublas = [&]()
    {
        // Row-major C=AB
        // 按 Column-major 视角计算 C^T=B^T*A^T

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
    };

    // ========================================
    // 11. 创建 CUDA Event
    // ========================================

    cudaEvent_t start, stop;

    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    constexpr int warmup = 20;
    constexpr int repeat = 1000;
    constexpr int trials = 3;

    const double flops = 2.0 * M * N * K;

    double cublas_ms = 0.0;

    double best_custom_ms =
        std::numeric_limits<double>::infinity();

    const char* best_custom_name = "";

    bool all_pass = true;

    std::printf(
        "Shape: M=%d K=%d N=%d | warmup=%d repeat=%d trials=%d\n",
        M, K, N, warmup, repeat, trials
    );

    std::printf(
        "All cases: default stream, shared inputs, same CPU reference.\n"
    );

    // ========================================
    // 12. 统一正确性与性能测试
    // ========================================

    auto run_case = [&](const char* name,
                        auto launch,
                        bool is_cublas)
    {
        // 用 NaN 初始化输出，检测遗漏写回
        std::fill(
            h_out.begin(),
            h_out.end(),
            std::numeric_limits<float>::quiet_NaN()
        );

        check_cuda(
            cudaMemcpy(
                d_C,
                h_out.data(),
                bytesC,
                cudaMemcpyHostToDevice
            ),
            "initialize C sentinel"
        );

        // 执行一次用于正确性检查
        launch();

        check_cuda(
            cudaGetLastError(),
            "correctness launch"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "correctness sync"
        );

        check_cuda(
            cudaMemcpy(
                h_out.data(),
                d_C,
                bytesC,
                cudaMemcpyDeviceToHost
            ),
            "copy result"
        );

        bool pass = verify(name, h_ref, h_out);

        all_pass = all_pass && pass;

        if (!pass)
            return;

        // 使用相同的 Benchmark 逻辑
        double ms = benchmark_events(
            launch,
            start,
            stop,
            warmup,
            repeat,
            trials
        );

        double gflops = flops / (ms * 1.0e6);

        std::printf(
            "    %-16s latency=%.6f ms  GFLOPS=%.3f\n",
            name, ms, gflops
        );

        if (is_cublas)
        {
            cublas_ms = ms;
        }
        else if (ms < best_custom_ms)
        {
            best_custom_ms = ms;
            best_custom_name = name;
        }
    };

    // ========================================
    // 13. 顺序执行五种 GEMM
    // ========================================

    std::printf("\nCorrectness and Benchmark:\n");

    run_case("Naive",        run_naive,  false);
    run_case("Tiled16",      run_tiled,  false);
    run_case("Register2_16", run_reg16,  false);
    run_case("Register2_32", run_reg32,  false);
    run_case("cuBLAS_SGEMM", run_cublas, true);

    // ========================================
    // 14. 汇总性能结果
    // ========================================

    if (all_pass && cublas_ms > 0.0)
    {
        std::printf(
            "\nBest custom: %s (%.6f ms)\n",
            best_custom_name,
            best_custom_ms
        );

        std::printf(
            "cuBLAS: %.6f ms\n",
            cublas_ms
        );

        std::printf(
            "cuBLAS speedup vs best custom: %.3fx\n",
            best_custom_ms / cublas_ms
        );
    }

    // ========================================
    // 15. 释放资源
    // ========================================

    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");

    check_cublas(cublasDestroy(handle), "cublasDestroy");

    check_cuda(cudaFree(d_A), "free A");
    check_cuda(cudaFree(d_B), "free B");
    check_cuda(cudaFree(d_C), "free C");

    return all_pass ? 0 : 1;
}
