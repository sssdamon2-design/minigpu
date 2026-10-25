#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>
constexpr int THREADS = 256;
constexpr int WARP_SIZE = 32; // constexpr 编译期就定死的常量，运行的时候不能改，GPU CUDA 代码里很常见。


// 1. Warp 内最大值归约
__device__ __forceinline__
float warp_max(float value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value = fmaxf(value,__shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}


// 2. Warp 内求和归约
__device__ __forceinline__
float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(
            0xffffffffu, value, offset
        );
    }
    return value;
}

// 3. 整个 Block 求最大值
__device__ float block_max(float value)
{
    __shared__ float partial[THREADS / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    value = warp_max(value);

    if (lane == 0)
        partial[warp] = value;

    __syncthreads();

    if (warp == 0)
    {
        float x = (lane < THREADS / WARP_SIZE) ? partial[lane]: -CUDART_INF_F;

        x = warp_max(x);

        if (lane == 0)
            partial[0] = x;
    }

    __syncthreads();
    return partial[0];
}

// 4. 整个 Block 求和
__device__ float block_sum(float value)
{
    __shared__ float partial[THREADS / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    value = warp_sum(value);

    if (lane == 0)
        partial[warp] = value;

    __syncthreads();

    if (warp == 0)
    {
        float x = (lane < THREADS / WARP_SIZE)
                  ? partial[lane]
                  : 0.0f;

        x = warp_sum(x);

        if (lane == 0)
            partial[0] = x;
    }

    __syncthreads();
    return partial[0];
}

// 5. Row-wise Softmax
__global__ void softmax_kernel(
    const float* X,
    float* Y,
    int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * N;

    // 第一阶段：每个线程求局部最大值
    float local_max = -CUDART_INF_F;

    for (int col = tid; col < N; col += blockDim.x)
    {
        local_max = fmaxf(
            local_max, X[base + col]
        );
    }

    // 第二阶段：整个 Block 求最大值
    float max_value = block_max(local_max);

    // 第三阶段：计算指数与局部和
    float local_sum = 0.0f;

    for (int col = tid; col < N; col += blockDim.x)
    {
        float e = expf(X[base + col] - max_value);
        Y[base + col] = e;
        local_sum += e;
    }

    // 第四阶段：整个 Block 求指数和
    float sum_value = block_sum(local_sum);

    // 第五阶段：归一化
    for (int col = tid; col < N; col += blockDim.x)
    {
        Y[base + col] /= sum_value;
    }
}


 // V4.4: Fused Scale + Softmax
__global__ void scaled_softmax_kernel(
    const float* X,
    float* Y,
    int N,
    float scale)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * N;

    // 1. 求整行最大值
    float local_max = -CUDART_INF_F;

    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(
            local_max,
            X[base + col]
        );
    }

    float max_value = block_max(local_max);

    // 2. Scale + Exp + 局部求和
    float local_sum = 0.0f;

    for (int col = tid; col < N; col += blockDim.x) {

        float value = X[base + col];

        float e = expf(
            (value - max_value) * scale
        );

        Y[base + col] = e;
        local_sum += e;
    }

    // 3. 整行 Sum Reduction
    float sum_value = block_sum(local_sum);

    // 4. 归一化
    for (int col = tid; col < N; col += blockDim.x) {
        Y[base + col] /= sum_value;
    }
}










#ifndef MINIGPU_SOFTMAX_NO_MAIN


void check_cuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR at %s: %s\n",
                     where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// CPU Reference：double 中间计算，提高参考值可靠性
std::vector<float> cpu_softmax(
    const std::vector<float>& X, int M, int N)
{
    std::vector<float> Y(X.size());

    for (int row = 0; row < M; ++row) {
        int base = row * N;

        double max_value = -INFINITY;
        for (int col = 0; col < N; ++col) {
            max_value = std::max(
                max_value, static_cast<double>(X[base + col]));
        }

        std::vector<double> exp_values(N);
        double sum = 0.0;

        for (int col = 0; col < N; ++col) {
            double e = std::exp(
                static_cast<double>(X[base + col]) - max_value);
            exp_values[col] = e;
            sum += e;
        }

        for (int col = 0; col < N; ++col) {
            Y[base + col] = static_cast<float>(
                exp_values[col] / sum);
        }
    }

    return Y;
}

// 执行一组 Shape 的正确性测试
bool run_test(int M, int N)
{
    size_t count = static_cast<size_t>(M) * N;
    size_t bytes = count * sizeof(float);

    std::vector<float> X(count);
    std::mt19937 rng(2026 + N);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    for (int row = 0; row < M; ++row) {
        float offset = 0.0f;

        // 大数测试：正数与负数交替
        if (N == 513)
            offset = (row % 2 == 0) ? 1000.0f : -1000.0f;

        for (int col = 0; col < N; ++col)
            X[row * N + col] = offset + dist(rng);
    }

    std::vector<float> reference = cpu_softmax(X, M, N);
    std::vector<float> output(count);

    float* d_X = nullptr;
    float* d_Y = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_X), bytes),
               "cudaMalloc X");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_Y), bytes),
               "cudaMalloc Y");

    check_cuda(cudaMemcpy(d_X, X.data(), bytes,
                          cudaMemcpyHostToDevice),
               "H2D");

    softmax_kernel<<<M, THREADS>>>(d_X, d_Y, N);

    check_cuda(cudaGetLastError(), "softmax launch");
    check_cuda(cudaDeviceSynchronize(), "softmax synchronize");

    check_cuda(cudaMemcpy(output.data(), d_Y, bytes,
                          cudaMemcpyDeviceToHost),
               "D2H");

    check_cuda(cudaFree(d_X), "cudaFree X");
    check_cuda(cudaFree(d_Y), "cudaFree Y");

    constexpr float atol = 2e-5f;
    constexpr float rtol = 1e-4f;

    bool pass = true;
    float max_diff = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float diff = std::fabs(output[i] - reference[i]);

        if (!std::isfinite(output[i]) ||
            diff > atol + rtol * std::fabs(reference[i]))
            pass = false;

        if (std::isfinite(diff))
            max_diff = std::max(max_diff, diff);
    }

    // Softmax 每一行概率之和应接近 1
    for (int row = 0; row < M; ++row) {
        double row_sum = 0.0;

        for (int col = 0; col < N; ++col)
            row_sum += output[row * N + col];

        if (!std::isfinite(row_sum) ||
            std::fabs(row_sum - 1.0) > 1e-4)
            pass = false;
    }

    std::printf("M=%d N=%d  %s  max_diff=%.8g\n",
                M, N, pass ? "PASS" : "FAIL", max_diff);

    return pass;
}

int main()
{
    bool all_pass = true;

    all_pass = run_test(4, 128) && all_pass;
    all_pass = run_test(3, 257) && all_pass;
    all_pass = run_test(2, 513) && all_pass;
    all_pass = run_test(1, 1) && all_pass;

    std::printf("\nV4.1 Softmax: %s\n",
                all_pass ? "ALL PASS" : "FAILED");

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}


#endif  // MINIGPU_SOFTMAX_NO_MAIN