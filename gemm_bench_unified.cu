
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

// ============================================================
// 1. CUDA / cuBLAS 错误检查与命令行参数
// ============================================================

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

// ============================================================
// 2. V2.1 Naive GEMM：GPU 使用 FP32，不能改成 double
// ============================================================

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

// ============================================================
// 3. V2.2 Shared Memory Tiled GEMM
// ============================================================

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

    for (int t = 0; t < (K - 1) / TILE + 1; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] = (row < M && aCol < K)
                   ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N)
                   ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================
// 4. V2.4 Register Blocking 1x2：TILE=16 / TILE=32
// ============================================================

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

    for (int t = 0; t < (K - 1) / TILE + 1; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] = (row < M && aCol < K)
                   ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col0 < N)
                   ? B[bRow * N + col0] : 0.0f;
        Bs[ty][tx + TILE] = (bRow < K && col1 < N)
                          ? B[bRow * N + col1] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
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

// ============================================================
// 5. CPU Reference：输入 FP32、累加 FP64、最终存回 FP32
// ============================================================

void gemm_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = static_cast<double>(
                    A[static_cast<size_t>(row) * K + k]);
                double b = static_cast<double>(
                    B[static_cast<size_t>(k) * N + col]);
                sum += a * b;
            }
            C[static_cast<size_t>(row) * N + col] =
                static_cast<float>(sum);
        }
    }
}

// ============================================================
// 6. 统一正确性验证：使用 FP64 Reference 的 FP32 输出
// ============================================================

bool verify(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& output)
{
    // 本次 FP32 随机输入 [-1,1]、K<=512 的验收阈值。
    // 此阈值不是所有矩阵规模/数据范围的通用数值标准。
    constexpr float atol = 2e-5f;
    constexpr float rtol = 1e-4f;

    float max_diff = 0.0f;
    size_t bad = 0;
    size_t first_bad = 0;

    for (size_t i = 0; i < reference.size(); ++i) {
        float ref = reference[i];
        float got = output[i];
        float diff = std::fabs(ref - got);

        if (std::isfinite(diff) && diff > max_diff)
            max_diff = diff;

        if (!std::isfinite(ref) || !std::isfinite(got) ||
            diff > atol + rtol * std::fabs(ref)) {
            if (bad == 0)
                first_bad = i;
            ++bad;
        }
    }

    bool ok = (bad == 0);
    std::printf("%-16s %s  max_diff=%.8g  bad=%zu\n",
                name, ok ? "PASS" : "FAIL", max_diff, bad);

    if (!ok) {
        std::printf("    first error at [%zu]: CPU=%.8g, GPU=%.8g\n",
                    first_bad, reference[first_bad], output[first_bad]);
    }
    return ok;
}

// ============================================================
// 7. CUDA Graph Benchmark：64 GEMM / Graph，回放 16 次
// ============================================================

template<class Launch>
double benchmark_graph(
    Launch launch,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop,
    int warmup,
    int graph_nodes,
    int graph_replays,
    int graph_warmup,
    int trials)
{
    // 先正常调用预热，避免库首次调用发生在捕获期间。
    for (int i = 0; i < warmup; ++i)
        launch();
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin graph capture");
    for (int i = 0; i < graph_nodes; ++i)
        launch();
    check_cuda(cudaStreamEndCapture(stream, &graph),
               "end graph capture");

    check_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
               "instantiate graph");
    check_cuda(cudaGraphDestroy(graph), "destroy graph");

    // Graph 预热不计时。
    for (int i = 0; i < graph_warmup; ++i)
        check_cuda(cudaGraphLaunch(graph_exec, stream), "graph warmup");
    check_cuda(cudaStreamSynchronize(stream), "graph warmup sync");

    std::vector<double> samples;
    samples.reserve(trials);

    for (int trial = 0; trial < trials; ++trial) {
        check_cuda(cudaEventRecord(start, stream), "event start");
        for (int i = 0; i < graph_replays; ++i)
            check_cuda(cudaGraphLaunch(graph_exec, stream), "graph replay");
        check_cuda(cudaEventRecord(stop, stream), "event stop");
        check_cuda(cudaEventSynchronize(stop), "event sync");

        float total_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&total_ms, start, stop),
                   "elapsed time");

        double gemm_count = static_cast<double>(graph_nodes) * graph_replays;
        samples.push_back(static_cast<double>(total_ms) / gemm_count);
    }

    check_cuda(cudaGraphExecDestroy(graph_exec), "destroy graph exec");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// ============================================================
// 8. main：相同输入、五种 GEMM、同一 Stream、统一计时
// ============================================================

int main(int argc, char* argv[])
{
    int M = 512, K = 512, N = 512;
    if (argc == 4) {
        if (!read_positive_int(argv[1], M) ||
            !read_positive_int(argv[2], K) ||
            !read_positive_int(argv[3], N)) {
            std::fprintf(stderr, "M, K, N must be positive integers.\n");
            return 1;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [M K N]\n", argv[0]);
        return 1;
    }

    // 自定义 Kernel 使用 int 索引，限制元素数量避免溢出。
    long long mk = 1LL * M * K;
    long long kn = 1LL * K * N;
    long long mn = 1LL * M * N;
    if (mk > INT_MAX || kn > INT_MAX || mn > INT_MAX) {
        std::fprintf(stderr, "Matrix too large for int indexing.\n");
        return 1;
    }

    size_t countA = static_cast<size_t>(mk);
    size_t countB = static_cast<size_t>(kn);
    size_t countC = static_cast<size_t>(mn);
    size_t bytesA = countA * sizeof(float);
    size_t bytesB = countB * sizeof(float);
    size_t bytesC = countC * sizeof(float);

    std::vector<float> h_A(countA);
    std::vector<float> h_B(countB);
    std::vector<float> h_ref(countC, 0.0f);
    std::vector<float> h_out(countC);

    // 固定种子随机 FP32 测试。
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < countA; ++i)
        h_A[i] = dist(rng);
    for (size_t i = 0; i < countB; ++i)
        h_B[i] = dist(rng);

    std::printf("Input: Random FP32, seed=2026\n");
    gemm_cpu(h_A, h_B, h_ref, M, K, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_A), bytesA), "malloc A");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_B), bytesB), "malloc B");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_C), bytesC), "malloc C");
    check_cuda(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice),
               "copy A");
    check_cuda(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice),
               "copy B");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create stream");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode");

    // 显式工作空间：SetStream 必须在 SetWorkspace 之前。
    constexpr size_t workspace_bytes = 4 * 1024 * 1024;
    void* workspace = nullptr;
    check_cuda(cudaMalloc(&workspace, workspace_bytes), "malloc workspace");
    check_cublas(cublasSetWorkspace(handle, workspace, workspace_bytes),
                 "cublasSetWorkspace");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int naive_threads = 128;
    int naive_blocks = static_cast<int>((mn - 1) / naive_threads + 1);
    dim3 tiled_block(16, 16);
    dim3 tiled_grid((N - 1) / 16 + 1, (M - 1) / 16 + 1);
    dim3 reg16_block(16, 16);
    dim3 reg16_grid((N - 1) / 32 + 1, (M - 1) / 16 + 1);
    dim3 reg32_block(32, 32);
    dim3 reg32_grid((N - 1) / 64 + 1, (M - 1) / 32 + 1);

    auto run_naive = [&]() {
        gemm_naive<<<naive_blocks, naive_threads, 0, stream>>>(
            d_A, d_B, d_C, M, K, N);
    };
    auto run_tiled = [&]() {
        gemm_tiled_kernel<16><<<tiled_grid, tiled_block, 0, stream>>>(
            d_A, d_B, d_C, M, K, N);
    };
    auto run_reg16 = [&]() {
        gemm_register2_kernel<16><<<reg16_grid, reg16_block, 0, stream>>>(
            d_A, d_B, d_C, M, K, N);
    };
    auto run_reg32 = [&]() {
        gemm_register2_kernel<32><<<reg32_grid, reg32_block, 0, stream>>>(
            d_A, d_B, d_C, M, K, N);
    };
    auto run_cublas = [&]() {
        // Row-major C=AB；cuBLAS column-major 视角：C^T=B^T A^T。
        check_cublas(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N),
            "cublasSgemm");
    };

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    constexpr int warmup = 20;
    constexpr int graph_nodes = 64;
    constexpr int graph_replays = 16;
    constexpr int graph_warmup = 3;
    constexpr int trials = 5;

    const double flops = 2.0 * M * N * K;
    double cublas_ms = 0.0;
    double best_custom_ms = std::numeric_limits<double>::infinity();
    const char* best_custom_name = "";
    bool all_pass = true;

    std::printf("Shape: M=%d K=%d N=%d\n", M, K, N);
    std::printf("CUDA Graph: %d GEMMs/graph x %d replays x %d trials\n",
                graph_nodes, graph_replays, trials);
    std::printf("All cases: same CUDA Stream, inputs, and CPU Reference.\n");
    std::printf("Correctness: atol=2e-5, rtol=1e-4\n");
    std::printf("\nCorrectness and Graph Benchmark:\n");

    auto run_case = [&](const char* name, auto launch, bool is_cublas) {
        // 先填 NaN，用于检测输出元素是否遗漏写入。
        std::fill(h_out.begin(), h_out.end(),
                  std::numeric_limits<float>::quiet_NaN());
        check_cuda(cudaMemcpyAsync(d_C, h_out.data(), bytesC,
                                   cudaMemcpyHostToDevice, stream),
                   "initialize C sentinel");
        check_cuda(cudaStreamSynchronize(stream), "sentinel sync");

        launch();
        check_cuda(cudaGetLastError(), "correctness launch");
        check_cuda(cudaStreamSynchronize(stream), "correctness sync");
        check_cuda(cudaMemcpy(h_out.data(), d_C, bytesC,
                              cudaMemcpyDeviceToHost), "copy result");

        bool pass = verify(name, h_ref, h_out);
        all_pass = all_pass && pass;
        if (!pass)
            return;

        double ms = benchmark_graph(launch, stream, start, stop,
                                    warmup, graph_nodes, graph_replays,
                                    graph_warmup, trials);
        double gflops = flops / (ms * 1.0e6);
        std::printf("    %-16s latency=%.6f ms  GFLOPS=%.3f\n",
                    name, ms, gflops);

        if (is_cublas) {
            cublas_ms = ms;
        } else if (ms < best_custom_ms) {
            best_custom_ms = ms;
            best_custom_name = name;
        }
    };

    run_case("Naive",        run_naive,  false);
    run_case("Tiled16",      run_tiled,  false);
    run_case("Register2_16", run_reg16,  false);
    run_case("Register2_32", run_reg32,  false);
    run_case("cuBLAS_SGEMM", run_cublas, true);

    if (all_pass && cublas_ms > 0.0) {
        std::printf("\nBest custom: %s (%.6f ms)\n",
                    best_custom_name, best_custom_ms);
        std::printf("cuBLAS: %.6f ms\n", cublas_ms);
        std::printf("cuBLAS speedup vs best custom: %.3fx\n",
                    best_custom_ms / cublas_ms);
    }

    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");
    check_cuda(cudaStreamSynchronize(stream), "final stream sync");
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    check_cuda(cudaFree(workspace), "free workspace");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    check_cuda(cudaFree(d_A), "free A");
    check_cuda(cudaFree(d_B), "free B");
    check_cuda(cudaFree(d_C), "free C");

    return all_pass ? 0 : 1;
}
