#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "runtime/device_buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// 已有 V4 Attention / CPU Reference，链接 attention_v4.cu。
void attention_forward(cublasHandle_t handle,
                       const float* d_Q, const float* d_K, const float* d_V,
                       float* d_scores, float* d_weights, float* d_out,
                       int S, int D, bool fused, cudaStream_t stream);
std::vector<float> attention_cpu(const std::vector<float>& Q,
                                 const std::vector<float>& K,
                                 const std::vector<float>& V,
                                 int S, int D);

void v5_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
}
void v5_check_blas(cublasStatus_t err, const char* where) {
    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(where) + ": cuBLAS error " +
                                 std::to_string(static_cast<int>(err)));
}

// 这些资源同样使用 RAII，防止异常时遗漏清理。
struct CudaStream {
    cudaStream_t value = nullptr;
    CudaStream() { v5_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "stream create"); }
    ~CudaStream() noexcept {
        if (value) {
            // 异常退出时也先等提交任务结束，再释放 Host/Device Buffer。
            (void)cudaStreamSynchronize(value);
            (void)cudaStreamDestroy(value);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};
struct CudaEvent {
    cudaEvent_t value = nullptr;
    CudaEvent() { v5_check(cudaEventCreate(&value), "event create"); }
    ~CudaEvent() noexcept { if (value) (void)cudaEventDestroy(value); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};
struct BlasHandle {
    cublasHandle_t value = nullptr;
    BlasHandle() { v5_check_blas(cublasCreate(&value), "cublasCreate"); }
    ~BlasHandle() noexcept { if (value) (void)cublasDestroy(value); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
};
// Pinned Host Memory 作为 cudaMemcpyAsync 的稳定异步传输源/目标。
template<class T>
struct PinnedBuffer {
    T* value = nullptr;
    explicit PinnedBuffer(std::size_t count) {
        v5_check(cudaMallocHost(reinterpret_cast<void**>(&value), count * sizeof(T)),
                 "cudaMallocHost");
    }
    ~PinnedBuffer() noexcept { if (value) (void)cudaFreeHost(value); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};


// V5.3: 仅为了公平对比，组合 6 个已有 RAII DeviceBuffer。
// 无内存池、无复杂 allocator：fresh 每轮构造，reuse 循环外构造。
struct AttentionWorkspace {
    DeviceBuffer<float> q, k, v, scores, weights, out;

    AttentionWorkspace(std::size_t data_count, std::size_t score_count)
        : q(data_count), k(data_count), v(data_count),
          scores(score_count), weights(score_count), out(data_count) {}
};

// 两种策略使用完全相同的数据传输与 fused Attention 计算路径。
// 只有显存申请和释放策略不同。
void execute_attention(
    AttentionWorkspace& w,
    BlasHandle& blas, CudaStream& stream,
    PinnedBuffer<float>& h_Q, PinnedBuffer<float>& h_K,
    PinnedBuffer<float>& h_V, PinnedBuffer<float>& h_out,
    int S, int D)
{
    const std::size_t bytes = static_cast<std::size_t>(S) * D * sizeof(float);
    v5_check(cudaMemcpyAsync(w.q.data(), h_Q.value, bytes,
                             cudaMemcpyHostToDevice, stream.value), "bench H2D Q");
    v5_check(cudaMemcpyAsync(w.k.data(), h_K.value, bytes,
                             cudaMemcpyHostToDevice, stream.value), "bench H2D K");
    v5_check(cudaMemcpyAsync(w.v.data(), h_V.value, bytes,
                             cudaMemcpyHostToDevice, stream.value), "bench H2D V");

    attention_forward(blas.value,
                      w.q.data(), w.k.data(), w.v.data(),
                      w.scores.data(), w.weights.data(), w.out.data(),
                      S, D, true, stream.value);

    v5_check(cudaMemcpyAsync(h_out.value, w.out.data(), bytes,
                             cudaMemcpyDeviceToHost, stream.value), "bench D2H");
    // 每轮都等待本次工作完成，Host Buffer 在此之前不被改写或读取。
    v5_check(cudaStreamSynchronize(stream.value), "bench stream sync");
}

bool verify_attention_output(const float* actual,
                             const std::vector<float>& reference,
                             float& max_diff)
{
    bool pass = true;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        if (!std::isfinite(actual[i]) ||
            diff > 2e-5f + 1e-4f * std::fabs(reference[i])) {
            pass = false;
        }
        if (std::isfinite(diff)) max_diff = std::max(max_diff, diff);
    }
    return pass;
}

// CPU wall time: Fresh 包括 6 次 cudaMalloc + 计算/传输/同步 + 6 次 cudaFree。
// Reuse 只包括相同的计算/传输/同步；预先分配的 Workspace 不计入每轮。
double measure_one(bool fresh, AttentionWorkspace& reused,
                   BlasHandle& blas, CudaStream& stream,
                   PinnedBuffer<float>& h_Q, PinnedBuffer<float>& h_K,
                   PinnedBuffer<float>& h_V, PinnedBuffer<float>& h_out,
                   int S, int D)
{
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();

    if (fresh) {
        // 括号确保析构/cudaFree 也落在计时区间内。
        AttentionWorkspace temporary(static_cast<std::size_t>(S) * D,
                                     static_cast<std::size_t>(S) * S);
        execute_attention(temporary, blas, stream, h_Q, h_K, h_V, h_out, S, D);
    } else {
        execute_attention(reused, blas, stream, h_Q, h_K, h_V, h_out, S, D);
    }

    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double median_us(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

bool benchmark_buffer_reuse(
    BlasHandle& blas, CudaStream& stream,
    PinnedBuffer<float>& h_Q, PinnedBuffer<float>& h_K,
    PinnedBuffer<float>& h_V, PinnedBuffer<float>& h_out,
    const std::vector<float>& reference, int S, int D)
{
    constexpr int WARMUP = 5;
    constexpr int PAIRS = 30;

    AttentionWorkspace reused(static_cast<std::size_t>(S) * D,
                              static_cast<std::size_t>(S) * S);

    // 预热两条路径。输入与参考结果来自 V5.2 最后一轮，保持不变。
    for (int i = 0; i < WARMUP; ++i) {
        (void)measure_one(true, reused, blas, stream, h_Q, h_K, h_V, h_out, S, D);
        (void)measure_one(false, reused, blas, stream, h_Q, h_K, h_V, h_out, S, D);
    }

    std::vector<double> fresh_us, reuse_us;
    fresh_us.reserve(PAIRS);
    reuse_us.reserve(PAIRS);
    bool pass = true;
    float max_diff = 0.0f;

    // 交替测试先后顺序，降低固定运行顺序带来的偏差。
    for (int i = 0; i < PAIRS; ++i) {
        for (int j = 0; j < 2; ++j) {
            const bool fresh = ((i + j) % 2 == 0);
            const double us = measure_one(
                fresh, reused, blas, stream, h_Q, h_K, h_V, h_out, S, D);
            (fresh ? fresh_us : reuse_us).push_back(us);
            pass = verify_attention_output(h_out.value, reference, max_diff) && pass;
        }
    }

    const double fresh_med = median_us(fresh_us);
    const double reuse_med = median_us(reuse_us);
    const double speedup = fresh_med / reuse_med;
    std::printf("V5.3 benchmark: S=%d D=%d FP32, warmup=%d, pairs=%d\n",
                S, D, WARMUP, PAIRS);
    std::printf("fresh_alloc_each: median=%.3f us (min=%.3f, max=%.3f)\n",
                fresh_med, *std::min_element(fresh_us.begin(), fresh_us.end()),
                *std::max_element(fresh_us.begin(), fresh_us.end()));
    std::printf("reuse_buffers:    median=%.3f us (min=%.3f, max=%.3f)\n",
                reuse_med, *std::min_element(reuse_us.begin(), reuse_us.end()),
                *std::max_element(reuse_us.begin(), reuse_us.end()));
    std::printf("speedup (ratio of medians)=%.3fx; max_diff=%.8g\n",
                speedup, max_diff);
    std::printf("V5.3 Buffer Reuse: %s\n", pass ? "ALL PASS" : "FAILED");
    return pass;
}

int main() {
    try {
        constexpr int S = 64, D = 64;
        constexpr std::size_t data_count = S * D;
        constexpr std::size_t score_count = S * S;
        constexpr std::size_t data_bytes = data_count * sizeof(float);

        // GPU Buffer 只申请一次，在三轮 Attention 中持续复用。
        DeviceBuffer<float> d_Q(data_count), d_K(data_count), d_V(data_count);
        DeviceBuffer<float> d_scores(score_count), d_weights(score_count);
        DeviceBuffer<float> d_out(data_count);

        PinnedBuffer<float> h_Q(data_count), h_K(data_count);
        PinnedBuffer<float> h_V(data_count), h_out(data_count);
        CudaStream stream;
        CudaEvent start, done;
        BlasHandle blas;
        v5_check_blas(cublasSetStream(blas.value, stream.value), "cublasSetStream");

        bool all_pass = true;
        std::vector<float> Q(data_count), K(data_count), V(data_count);
        std::printf("GPU buffers allocated once; S=%d D=%d, rounds=3\n", S, D);

        for (int run = 0; run < 3; ++run) {
            // 三轮输入不同；写入 Pinned Host Buffer。
            for (std::size_t i = 0; i < data_count; ++i) {
                Q[i] = (static_cast<int>((i + run) % 31) - 15) * 0.05f;
                K[i] = (static_cast<int>((i * 3 + run) % 29) - 14) * 0.04f;
                V[i] = (static_cast<int>((i * 7 + run) % 37) - 18) * 0.03f;
                h_Q.value[i] = Q[i];
                h_K.value[i] = K[i];
                h_V.value[i] = V[i];
            }

            auto reference = attention_cpu(Q, K, V, S, D);

            // 同一个 Stream：H2D -> Attention -> D2H -> Event。
            v5_check(cudaEventRecord(start.value, stream.value), "record start");
            v5_check(cudaMemcpyAsync(d_Q.data(), h_Q.value, data_bytes,
                                     cudaMemcpyHostToDevice, stream.value), "H2D Q");
            v5_check(cudaMemcpyAsync(d_K.data(), h_K.value, data_bytes,
                                     cudaMemcpyHostToDevice, stream.value), "H2D K");
            v5_check(cudaMemcpyAsync(d_V.data(), h_V.value, data_bytes,
                                     cudaMemcpyHostToDevice, stream.value), "H2D V");

            attention_forward(blas.value,
                              d_Q.data(), d_K.data(), d_V.data(),
                              d_scores.data(), d_weights.data(), d_out.data(),
                              S, D, true, stream.value);

            v5_check(cudaMemcpyAsync(h_out.value, d_out.data(), data_bytes,
                                     cudaMemcpyDeviceToHost, stream.value), "D2H output");
            v5_check(cudaEventRecord(done.value, stream.value), "record done");
            v5_check(cudaEventSynchronize(done.value), "wait done");

            float elapsed_ms = 0.0f;
            v5_check(cudaEventElapsedTime(&elapsed_ms, start.value, done.value),
                     "elapsed");

            float max_diff = 0.0f;
            bool pass = true;
            for (std::size_t i = 0; i < data_count; ++i) {
                float actual = h_out.value[i];
                float diff = std::fabs(actual - reference[i]);
                if (!std::isfinite(actual) ||
                    diff > 2e-5f + 1e-4f * std::fabs(reference[i]))
                    pass = false;
                if (std::isfinite(diff)) max_diff = std::max(max_diff, diff);
            }
            all_pass = all_pass && pass;
            std::printf("round=%d %s max_diff=%.8g stream_elapsed=%.4f ms\n",
                        run + 1, pass ? "PASS" : "FAIL", max_diff, elapsed_ms);
        }
        std::printf("V5.2 Attention Runtime: %s\n", all_pass ? "ALL PASS" : "FAILED");

        if (all_pass) {
            // 最后一轮 Q/K/V 仍在 Pinned Host Buffer 中；CPU Reference 不计时。
            auto reference = attention_cpu(Q, K, V, S, D);
            all_pass = benchmark_buffer_reuse(
                blas, stream, h_Q, h_K, h_V, h_out, reference, S, D);
        }
        return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "V5.2 error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
