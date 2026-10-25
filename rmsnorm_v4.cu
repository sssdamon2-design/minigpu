
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

constexpr int THREADS = 256;
constexpr int WARP_SIZE = 32;

// 1. Warp 内求和：复用 V4.1 的算法
__device__ __forceinline__
float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(
            0xffffffffu, value, offset);
    }
    return value;
}

// 2. Block 求和：先 Warp 内归约，再跨 Warp 归约
__device__ float block_sum(float value)
{
    __shared__ float partial[THREADS / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    value = warp_sum(value);

    if (lane == 0)
        partial[warp] = value;

    __syncthreads();

    if (warp == 0) {
        float x = (lane < THREADS / WARP_SIZE)
                  ? partial[lane] : 0.0f;

        x = warp_sum(x);

        if (lane == 0)
            partial[0] = x;
    }

    __syncthreads();

    return partial[0];
}

// 3. 核心 RMSNorm CUDA Kernel
__global__ void rmsnorm_kernel(
    const float* X,
    const float* gamma,
    float* Y,
    int D,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * D;

    // 第一步：各线程计算局部平方和
    float local_sum = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        float x = X[base + col];
        local_sum += x * x;
    }

    // 第二步：整个 Block 进行 Sum Reduction
    float sum_sq = block_sum(local_sum);

    // 第三步：计算逆均方根
    float inv_rms = rsqrtf(sum_sq / D + eps);

    // 第四步：归一化并进行 Gamma 缩放
    for (int col = tid; col < D; col += blockDim.x) {
        Y[base + col] = X[base + col] * inv_rms * gamma[col];
    }
}




void check_cuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR at %s: %s\n",
                     where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// CPU Reference：用 double 作为中间计算精度
std::vector<float> cpu_rmsnorm(
    const std::vector<float>& X,
    const std::vector<float>& gamma,
    int M, int D, float eps)
{
    std::vector<float> Y(X.size());

    for (int row = 0; row < M; ++row) {
        int base = row * D;

        double sum_sq = 0.0;

        for (int col = 0; col < D; ++col) {
            double x = X[base + col];
            sum_sq += x * x;
        }

        double inv_rms =
            1.0 / std::sqrt(sum_sq / D + eps);

        for (int col = 0; col < D; ++col) {
            Y[base + col] = static_cast<float>(
                X[base + col] * inv_rms * gamma[col]);
        }
    }

    return Y;
}

// 一组 Shape 的正确性测试
bool run_test(int M, int D)
{
    constexpr float eps = 1e-5f;
    constexpr float atol = 2e-5f;
    constexpr float rtol = 1e-4f;

    size_t count = static_cast<size_t>(M) * D;
    size_t bytes = count * sizeof(float);
    size_t gamma_bytes = static_cast<size_t>(D) * sizeof(float);

    std::vector<float> X(count);
    std::vector<float> gamma(D);

    std::mt19937 rng(2026 + D);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);

    for (float& x : X)
        x = dist(rng);

    for (float& g : gamma)
        g = scale_dist(rng);

    // 特殊边界：全零输入，检查 eps 是否正常工作
    if (D == 1)
        X[0] = 0.0f;

    auto reference = cpu_rmsnorm(X, gamma, M, D, eps);

    std::vector<float> output(
        count, std::numeric_limits<float>::quiet_NaN());

    float* d_X = nullptr;
    float* d_gamma = nullptr;
    float* d_Y = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_X), bytes),
               "malloc X");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_gamma), gamma_bytes),
               "malloc gamma");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_Y), bytes),
               "malloc Y");

    check_cuda(cudaMemcpy(d_X, X.data(), bytes,
                          cudaMemcpyHostToDevice), "copy X");
    check_cuda(cudaMemcpy(d_gamma, gamma.data(), gamma_bytes,
                          cudaMemcpyHostToDevice), "copy gamma");

    // 初始化 NaN，帮助检查是否有输出元素未被写入
    check_cuda(cudaMemcpy(d_Y, output.data(), bytes,
                          cudaMemcpyHostToDevice), "initialize Y");

    rmsnorm_kernel<<<M, THREADS>>>(
        d_X, d_gamma, d_Y, D, eps);

    check_cuda(cudaGetLastError(), "rmsnorm launch");
    check_cuda(cudaDeviceSynchronize(), "rmsnorm sync");

    check_cuda(cudaMemcpy(output.data(), d_Y, bytes,
                          cudaMemcpyDeviceToHost), "copy Y");

    check_cuda(cudaFree(d_X), "free X");
    check_cuda(cudaFree(d_gamma), "free gamma");
    check_cuda(cudaFree(d_Y), "free Y");

    bool pass = true;
    float max_diff = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float diff = std::fabs(output[i] - reference[i]);

        if (!std::isfinite(output[i]) ||
            diff > atol + rtol * std::fabs(reference[i])) {
            pass = false;
        }

        if (std::isfinite(diff))
            max_diff = std::max(max_diff, diff);
    }

    std::printf("M=%d D=%d  %s  max_diff=%.8g\n",
                M, D, pass ? "PASS" : "FAIL", max_diff);

    return pass;
}

int main()
{
    bool all_pass = true;

    all_pass = run_test(4, 128) && all_pass;
    all_pass = run_test(3, 257) && all_pass;
    all_pass = run_test(2, 768) && all_pass;
    all_pass = run_test(1, 1) && all_pass;

    std::printf("\nV4.2 RMSNorm: %s\n",
                all_pass ? "ALL PASS" : "FAILED");

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
