
#include <cuda_runtime.h>
#include "runtime/device_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 复用 V4.1 的 Softmax Kernel
__global__ void softmax_kernel(
    const float* X, float* Y, int N);

void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(where) + ": " +
            cudaGetErrorString(err)
        );
    }
}

int main()
{
    try {
        constexpr int M = 3;
        constexpr int N = 257;
        constexpr int THREADS = 256;

        const std::size_t count =
            static_cast<std::size_t>(M) * N;

        std::vector<float> input(count);
        std::vector<float> output(count);

        // 构造确定性输入
        for (std::size_t i = 0; i < count; ++i) {
            input[i] =
                (static_cast<int>(i % 41) - 20) * 0.25f;
        }

        // 关键：GPU 显存由 C++ 对象管理
        DeviceBuffer<float> d_X(count);
        DeviceBuffer<float> d_Y(count);

        cuda_check(
            cudaMemcpy(d_X.data(), input.data(),
                       d_X.bytes(), cudaMemcpyHostToDevice),
            "H2D"
        );

        // 执行已有 V4 Softmax
        softmax_kernel<<<M, THREADS>>>(
            d_X.data(), d_Y.data(), N
        );

        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "kernel sync");

        cuda_check(
            cudaMemcpy(output.data(), d_Y.data(),
                       d_Y.bytes(), cudaMemcpyDeviceToHost),
            "D2H"
        );

        // CPU Reference：稳定 Softmax
        bool pass = true;
        double max_diff = 0.0;

        for (int row = 0; row < M; ++row) {
            int base = row * N;

            double max_value =
                -std::numeric_limits<double>::infinity();

            for (int col = 0; col < N; ++col) {
                max_value = std::max(
                    max_value,
                    static_cast<double>(input[base + col])
                );
            }

            double sum = 0.0;

            for (int col = 0; col < N; ++col) {
                sum += std::exp(
                    static_cast<double>(input[base + col])
                    - max_value
                );
            }

            double row_sum = 0.0;

            for (int col = 0; col < N; ++col) {
                int idx = base + col;

                double ref = std::exp(
                    static_cast<double>(input[idx]) - max_value
                ) / sum;

                double diff = std::fabs(output[idx] - ref);
                max_diff = std::max(max_diff, diff);
                row_sum += output[idx];

                if (!std::isfinite(output[idx]) ||
                    diff > 2e-5 + 1e-4 * std::fabs(ref)) {
                    pass = false;
                }
            }

            if (std::fabs(row_sum - 1.0) > 1e-4)
                pass = false;
        }

        std::printf(
            "V5.1 DeviceBuffer + Softmax: %s\n",
            pass ? "PASS" : "FAIL"
        );

        std::printf("max_diff=%.8g\n", max_diff);

        // 无需手动 cudaFree
        // d_Y、d_X 离开作用域自动析构
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
