
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

// 来自 softmax_v4.cu 的已有 Kernel
__global__ void softmax_kernel(const float* X, float* Y, int N);
__global__ void scaled_softmax_kernel(
    const float* X,
    float* Y,
    int N,
    float scale
);
constexpr int THREADS = 256;

void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n",
                     where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s: %d\n",
                     where, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

// V4.3: 独立 Scale Kernel
// V4.4 将尝试与 Softmax 融合
__global__ void scale_kernel(float* scores, int count, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        scores[idx] *= scale;
}




void attention_forward(
    cublasHandle_t handle,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_weights,
    float* d_out,
    int S,
    int D,
bool fused,
cudaStream_t stream = nullptr)
{
    const float one = 1.0f;
    const float zero = 0.0f;

    // 1. GEMM: Scores = Q @ K^T
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, S, D,
            &one,
            d_K, D,
            d_Q, D,
            &zero,
            d_scores, S),
        "QK^T"
    );

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    if (fused) {
        // V4.4: 融合路径
        scaled_softmax_kernel<<<S, THREADS, 0, stream>>>(
            d_scores, d_weights, S, scale
        );
        check_cuda(cudaGetLastError(), "fused softmax launch");
    } else {
        // V4.3: 原始 Baseline
        scale_kernel<<<
    (S * S + THREADS - 1) / THREADS,
    THREADS, 0, stream>>>(d_scores, S * S, scale);
        check_cuda(cudaGetLastError(), "scale launch");

        softmax_kernel<<<S, THREADS, 0, stream>>>(
    d_scores, d_weights, S
);
        check_cuda(cudaGetLastError(), "softmax launch");
    }

    // 3. GEMM: Output = Weights @ V
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, S, S,
            &one,
            d_V, D,
            d_weights, S,
            &zero,
            d_out, D),
        "Weights@V"
    );
}




double benchmark_attention(
    cublasHandle_t handle,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_weights,
    float* d_out,
    int S,
    int D,
    bool fused)
{
    constexpr int WARMUP = 10;
    constexpr int REPEAT = 50;
    constexpr int TRIALS = 5;

    // 预热：不计入正式时间
    for (int i = 0; i < WARMUP; ++i) {
        attention_forward(
            handle, d_Q, d_K, d_V,
            d_scores, d_weights, d_out,
            S, D, fused
        );
    }

    check_cuda(cudaDeviceSynchronize(), "benchmark warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    std::vector<double> samples;

    for (int trial = 0; trial < TRIALS; ++trial) {
        check_cuda(cudaEventRecord(start), "record start");

        for (int i = 0; i < REPEAT; ++i) {
            attention_forward(
                handle, d_Q, d_K, d_V,
                d_scores, d_weights, d_out,
                S, D, fused
            );
        }

        check_cuda(cudaEventRecord(stop), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");

        float elapsed_ms = 0.0f;

        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "elapsed time"
        );

        double latency_us =
            static_cast<double>(elapsed_ms) * 1000.0 / REPEAT;

        samples.push_back(latency_us);
    }

    check_cuda(cudaEventDestroy(start), "destroy start");
    check_cuda(cudaEventDestroy(stop), "destroy stop");

    std::sort(samples.begin(), samples.end());

    return samples[TRIALS / 2];
}



// CPU Reference: 使用 double 计算中间结果
std::vector<float> attention_cpu(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int S, int D)
{
    std::vector<float> out(static_cast<size_t>(S) * D);
    const double inv_scale = 1.0 / std::sqrt(static_cast<double>(D));

    std::vector<double> scores(S);
    std::vector<double> weights(S);

    for (int i = 0; i < S; ++i) {
        double max_score = -std::numeric_limits<double>::infinity();

        // 1. QK^T / sqrt(D)
        for (int j = 0; j < S; ++j) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += static_cast<double>(Q[i * D + d]) * K[j * D + d];

            scores[j] = dot * inv_scale;
            max_score = std::max(max_score, scores[j]);
        }

        // 2. Stable Softmax
        double sum = 0.0;
        for (int j = 0; j < S; ++j) {
            weights[j] = std::exp(scores[j] - max_score);
            sum += weights[j];
        }

        // 3. Attention Weights @ V
        for (int d = 0; d < D; ++d) {
            double v = 0.0;
            for (int j = 0; j < S; ++j)
                v += (weights[j] / sum) * V[j * D + d];

            out[i * D + d] = static_cast<float>(v);
        }
    }
    return out;
}

bool run_test(int S, int D) {
    const size_t data_count = static_cast<size_t>(S) * D;
    const size_t score_count = static_cast<size_t>(S) * S;
    const size_t data_bytes = data_count * sizeof(float);
    const size_t score_bytes = score_count * sizeof(float);

    std::mt19937 rng(2026 + S + D);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q(data_count), K(data_count), V(data_count);

    for (size_t i = 0; i < data_count; ++i) {
        Q[i] = dist(rng);
        K[i] = dist(rng);
        V[i] = dist(rng);
    }

    auto reference = attention_cpu(Q, K, V, S, D);
    std::vector<float> output(data_count);

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_scores = nullptr, *d_weights = nullptr, *d_out = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_Q), data_bytes), "malloc Q");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_K), data_bytes), "malloc K");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_V), data_bytes), "malloc V");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scores), score_bytes), "malloc scores");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), score_bytes), "malloc weights");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), data_bytes), "malloc out");

    check_cuda(cudaMemcpy(d_Q, Q.data(), data_bytes,
                          cudaMemcpyHostToDevice), "copy Q");
    check_cuda(cudaMemcpy(d_K, K.data(), data_bytes,
                          cudaMemcpyHostToDevice), "copy K");
    check_cuda(cudaMemcpy(d_V, V.data(), data_bytes,
                          cudaMemcpyHostToDevice), "copy V");

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "create handle");

    // Baseline: 原始 4 阶段
    attention_forward(
        handle, d_Q, d_K, d_V,
        d_scores, d_weights, d_out,
        S, D, false
    );

    check_cuda(cudaDeviceSynchronize(), "baseline sync");

    std::vector<float> baseline(data_count);
    check_cuda(
        cudaMemcpy(baseline.data(), d_out, data_bytes,
                   cudaMemcpyDeviceToHost),
        "copy baseline"
    );

    // Fusion: 融合后的 3 阶段
    attention_forward(
        handle, d_Q, d_K, d_V,
        d_scores, d_weights, d_out,
        S, D, true
    );

    check_cuda(cudaDeviceSynchronize(), "fused sync");

    check_cuda(
        cudaMemcpy(output.data(), d_out, data_bytes,
                   cudaMemcpyDeviceToHost),
        "copy fused output"
    );


    constexpr float atol = 2e-5f;
    constexpr float rtol = 1e-4f;

    bool pass = true;
    float baseline_diff = 0.0f;
    float fused_diff = 0.0f;

    for (size_t i = 0; i < data_count; ++i) {
        float diff1 = std::fabs(baseline[i] - reference[i]);
        float diff2 = std::fabs(output[i] - reference[i]);

        float tolerance =
            atol + rtol * std::fabs(reference[i]);

        if (!std::isfinite(baseline[i]) ||
            !std::isfinite(output[i]) ||
            diff1 > tolerance ||
            diff2 > tolerance) {
            pass = false;
        }

        if (std::isfinite(diff1))
            baseline_diff = std::max(baseline_diff, diff1);

        if (std::isfinite(diff2))
            fused_diff = std::max(fused_diff, diff2);
    }

    std::printf(
        "S=%d D=%d  %s  baseline_diff=%.8g fused_diff=%.8g\n",
        S, D, pass ? "PASS" : "FAIL",
        baseline_diff, fused_diff
    );

    // 只对较大的三组 Shape 进行性能测试
    if (pass && S >= 64) {
        double baseline_us = benchmark_attention(
            handle, d_Q, d_K, d_V,
            d_scores, d_weights, d_out,
            S, D, false
        );

        double fused_us = benchmark_attention(
            handle, d_Q, d_K, d_V,
            d_scores, d_weights, d_out,
            S, D, true
        );

        std::printf(
            "    baseline=%.3f us  fused=%.3f us  speedup=%.3fx\n",
            baseline_us, fused_us, baseline_us / fused_us
        );
    }


    check_cublas(cublasDestroy(handle), "destroy handle");

    check_cuda(cudaFree(d_Q), "free Q");
    check_cuda(cudaFree(d_K), "free K");
    check_cuda(cudaFree(d_V), "free V");
    check_cuda(cudaFree(d_scores), "free scores");
    check_cuda(cudaFree(d_weights), "free weights");
    check_cuda(cudaFree(d_out), "free out");

    return pass;
}
#ifndef MINIGPU_ATTENTION_NO_MAIN
int main() {
    bool pass = true;

    pass = run_test(1, 8) && pass;
    pass = run_test(4, 16) && pass;
    pass = run_test(17, 32) && pass;

    pass = run_test(64, 64) && pass;
    pass = run_test(128, 64) && pass;
    pass = run_test(256, 64) && pass;

    std::printf("\nV4.4 Attention Fusion: %s\n",
                pass ? "ALL PASS" : "FAILED");

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif