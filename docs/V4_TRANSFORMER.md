# MiniGPU V4 — Transformer CUDA 算子、Attention Forward 与 Kernel Fusion

> **定位**：MiniGPU V1～V6 的第四阶段；用于后续自己复习、AI Systems / CUDA 面试、跨 AI 工程交接。  
> **数据说明**：本报告中的 GPU 正确性与性能数字均来自本次会话中的用户真实运行日志；代码行为以已提供的 `attention_v4.cu`、V4.1/V4.2 实现与命令为依据。未实测项目明确标记。  
> **记录状态**：V4.1～V4.4 的核心实现和正确性验收已完成；Fusion 性能已经历 4 轮测量。**截至记录时，V4 文档尚未由用户复制到 WSL 仓库，尚无 `v4.0` 的 Git Commit/Tag/Push 成功记录。**不要把技术实验完成误写成 GitHub 发布成功。  
> **主机**：NVIDIA RTX 2060 Laptop（6GB，SM 7.5）／WSL2 Ubuntu 24.04.5／CUDA Toolkit 12.8.93；Windows 驱动 591.74。  
> **仓库**：<https://github.com/sssdamon2-design/minigpu>；工程目录 `/home/damon/projects/minigpu`。

## 0. 三十秒回顾

1. **V4.1** 实现 FP32 Row-wise Stable Softmax：Warp Shuffle + 跨 Warp Shared Memory Block Reduction；`4/4 PASS`。
2. **V4.2** 实现 FP32 RMSNorm：平方和归约 + `rsqrtf` + Gamma 缩放；`4/4 PASS`。
3. **V4.3** 使用 cuBLAS SGEMM 和自定义 Scale/Softmax，完成单 Head、不带 Mask 的 FP32 `QKᵀ → Scale → Softmax → PV`；`4/4 PASS`。
4. **V4.4** 新增 `scaled_softmax_kernel()`，保留 Baseline 和 Fusion 双路径；`6/6 PASS`。三组 Shape 各测 4 轮，配对加速比中位数约 **1.279×（S64）、1.209×（S128）、1.248×（S256）**。
5. **重要限制**：有显著性能离群波动，不能断言所有环境下稳定获得这些加速。当前没有做 V4 指令级 Nsight Profile，无法将任何异常波动归因于 Bank Conflict、Occupancy、DRAM 或特定硬件事件。
6. **明确不做**：FlashAttention、Backward、KV Cache、工业级 Tensor 框架、重复 GEMM 优化、全量性能参数搜索。下一阶段 **V5 轻量 GPU Runtime**。

---

## 1. 工程与版本关系

```text
MiniGPU（同一仓库持续演进）
V1：CUDA 基础与 MLP                   已完成并发布 v1.0
V2：FP32 GEMM、Tiling、Register Blocking 已完成并发布 v2.0
V3：Nsight Profiling、瓶颈实验         已完成并发布 v3.0，Commit f248bdb
V4：Softmax / RMSNorm / Attention / Fusion 代码测试完成，待文档提交与 v4.0 发布
V5：Buffer RAII、Stream、Event、资源复用     未开始
V6：真实小型 Transformer 的端到端优化     未开始
```

### 1.1 文件与构建目标

| 文件 | 作用 | 本阶段验证 |
|---|---|---|
| `softmax_v4.cu` | `warp_max/sum`、`block_max/sum`、`softmax_kernel`、`scaled_softmax_kernel`；独立测试主函数通过宏保护 | 原始 Softmax 正确性通过；Fusion Kernel 经 Attention 端到端验证 |
| `rmsnorm_v4.cu` | RMSNorm Kernel、CPU Reference、四组测试 | 正确性通过 |
| `attention_v4.cu` | cuBLAS GEMM、独立 Scale、两种 Attention 路径、CPU Reference、Event Benchmark | 正确性和性能对照通过 |
| `CMakeLists.txt` | 增加三项 V4 可执行目标 | `build-v4` 已成功编译相应目标 |
| `docs/V4_TRANSFORMER.md` | 本报告 | 尚需复制至本地仓库并 Git 提交 |

CMake 关键结构（已有目标不要重复添加）：

```cmake
add_executable(softmax_v4 softmax_v4.cu)
add_executable(rmsnorm_v4 rmsnorm_v4.cu)

add_executable(attention_v4 attention_v4.cu softmax_v4.cu)
target_compile_definitions(attention_v4 PRIVATE MINIGPU_SOFTMAX_NO_MAIN)
target_link_libraries(attention_v4 PRIVATE CUDA::cublas CUDA::cudart)
```

`softmax_v4.cu` 在测试主函数与 CPU 测试代码之前使用 `#ifndef MINIGPU_SOFTMAX_NO_MAIN`，末尾用 `#endif` 收束；这样 Attention 目标只链接 GPU Kernel，不会因为两个 `main()` 冲突而失败。

### 1.2 实机和测试边界

- GPU：RTX 2060 Laptop 6GB，Compute Capability 7.5。
- WSL2 Ubuntu 24.04.5；Toolkit/NVCC 12.8.93。
- 计算数据类型：**FP32**；CPU Reference 在关键中间环节使用 **double**。
- Tensor 布局：代码按 **row-major** 扁平存储；cuBLAS 接口按 **column-major** 解释，需通过矩阵转置恒等式正确使用。
- Attention：**单 Head、无 Mask、无投影层、无反向传播、无 KV Cache**；不要将本项目称作完整 Transformer 或 FlashAttention。
- Benchmark：当前统计的是重复提交的 Attention **GPU Stream 区间耗时**；不包括显存分配与 H2D/D2H，可能包含 GPU 空闲间隔与 CPU 提交节奏影响。

---

## 2. V4.1：Reduction 与 Stable Softmax

### 2.1 数学公式与张量 Shape

对于 `X [M,N]` 的每一行：

\[
 m_i = \max_{0\le j<N}X_{i,j},\qquad
 Y_{i,j}=\frac{\exp(X_{i,j}-m_i)}{\sum_{k=0}^{N-1}\exp(X_{i,k}-m_i)}.
\]

为何减最大值：Softmax 在实数运算中对整体平移不变，但直接计算 `exp(1000)` 会让 FP32 溢出。先减最大值把最大指数输入变成 0，避免这类正向溢出。

注意：输入应具有正长度、值应满足当前实现的有限值假设；未测试包含 NaN、`+Inf`、全 `-Inf` 等输入的行为，不要宣传为通用鲁棒 Softmax。

### 2.2 CUDA 线程映射

```cpp
int row = blockIdx.x;
int tid = threadIdx.x;
int base = row * N;

for (int col = tid; col < N; col += blockDim.x) {
    // 当前线程负责该行的 col, col+256, ...
}
```

- 采用 **每行一个 Block**、每块 **256 个线程**。
- `blockIdx.x` 指定行；`threadIdx.x` 指定行内线程编号。
- 假设 `N=513`：线程 0 负责列 `0,256,512`；线程 1 负责 `1,257`；越界由 `col < N` 保护。
- 连续线程访问连续列，便于连续全局内存访问；该固定线程映射以正确性和可解释性为先，不是所有行宽下的最优算法。

### 2.3 Warp 内归约

```cpp
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}
```

- 32 个 Lane 组成一个 Warp；`offset` 依次取 `16,8,4,2,1`。
- 对求最大值，把加法换成 `fmaxf(value, other)`。
- **使用完整参与掩码时，32 个 Lane 需要按约定参与此指令**；本项目固定 256 线程并给空元素设置单位元，符合这一前提。
- `warp_sum` 的完整 Warp 归约结果只从 **Lane 0** 取用，不能认为其他所有 Lane 都拥有相同结果。

### 2.4 跨 Warp 的 Block Reduction

256 线程 = 8 Warp；在每个 Warp 内先求部分结果：

1. 各线程计算局部值（Max 使用 `-INFINITY`，Sum 使用 `0` 作单位元）。
2. Warp 内执行 Shuffle 归约。
3. 每个 Warp 的 Lane 0 写 `partial[warp]` 至 Shared Memory。
4. `__syncthreads()`：确保所有 Warp 的部分结果都已可见。
5. Warp 0 读取 8 个有效部分结果；其余 Lane 用相应单位元补齐，再次执行 Shuffle。
6. Warp 0 Lane 0 将最终结果写入 `partial[0]`。
7. 第二次 `__syncthreads()`：确保其他 Warp 可以安全读取最终结果。

**同步的面试要点**：第一次 Barrier 保护跨 Warp 部分结果的写→读，第二次保护最终结果的写→读；`__shfl_down_sync` 仅用于 Warp 内通信，不跨 Warp。

### 2.5 Softmax Kernel 数据流

```cpp
local_max = -INFINITY;
// 各线程扫描自己负责的列，得到 local_max
max_value = block_max(local_max);

local_sum = 0;
// e=expf(x-max_value); Y[col]=e; local_sum+=e
sum_value = block_sum(local_sum);

// Y[col] /= sum_value
```

`block_max` 和 `block_sum` 分别同步；线程写入各自不同的 `Y[base+col]`，避免输出写入竞争。GPU FP32 并行求和与 CPU double 的舍入顺序不必完全相同。

### 2.6 V4.1 实测正确性

| Shape [M,N] | 结果 | 最大绝对误差 |
|---|---|---:|
| `[4,128]` | PASS | `3.7252903e-09` |
| `[3,257]` | PASS | `1.8626451e-09` |
| `[2,513]` | PASS | `9.3132257e-10` |
| `[1,1]` | PASS | `0` |

`V4.1 Softmax: ALL PASS`。测试包括非 32/256 整倍数的行宽、±1000 偏移输入、单元素边界；对照 CPU Stable Softmax，逐元素容差 `atol=2e-5, rtol=1e-4`，并检查每行概率和与 1 的误差 ≤ `1e-4`。

---

## 3. V4.2：RMSNorm CUDA

### 3.1 公式与 Shape

\[
 y_i = x_i\,\gamma_i\cdot \operatorname{rsqrt}
 \left(\frac{1}{D}\sum_{j=0}^{D-1}x_j^2+\epsilon\right),
 \qquad \epsilon=10^{-5}.
\]

- `X [M,D]`，`gamma [D]`，`Y [M,D]`；对每行的最后一维求平方均值。
- `gamma[col]` 是所有行共享的特征缩放系数，不应写成 `gamma[row*D+col]`。
- `eps` 加在平方根内部，确保零输入时分母不为 0。
- 与 LayerNorm 不同，本次 RMSNorm **不减均值**；与 Softmax 不同，不需要 Max Reduction 或指数计算。

### 3.2 核心 Kernel

```cpp
__global__ void rmsnorm_kernel(
    const float* X, const float* gamma, float* Y,
    int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * D;
    float local_sum = 0.0f;

    for (int col = tid; col < D; col += blockDim.x) {
        float x = X[base + col];
        local_sum += x * x;
    }

    float sum_sq = block_sum(local_sum);
    float inv_rms = rsqrtf(sum_sq / D + eps);

    for (int col = tid; col < D; col += blockDim.x)
        Y[base + col] = X[base + col] * inv_rms * gamma[col];
}
```

与 V4.1 相同的 Warp/Block 求和算法在新文件中再次使用，是当前简单工程阶段的可接受代码复用方式；没有引入通用 Tensor Runtime。以后 V5 根据实际需要再抽公共模块。

### 3.3 V4.2 实测正确性

| Shape [M,D] | 结果 | 最大绝对误差 |
|---|---|---:|
| `[4,128]` | PASS | `2.3841858e-07` |
| `[3,257]` | PASS | `2.3841858e-07` |
| `[2,768]` | PASS | `4.7683716e-07` |
| `[1,1]` | PASS | `0` |

`V4.2 RMSNorm: ALL PASS`。包含随机输入 Gamma（非全 1）、非整倍数 Shape、大于 Block 宽度的特征维度，以及 `[1,1]` 零输入。CPU Reference 采用 double 中间计算；容差 `atol=2e-5, rtol=1e-4`。**本阶段仅通过正确性，未单独测 RMSNorm 性能。**

---

## 4. V4.3：FP32 Self-Attention Forward

### 4.1 计算公式

\[
 \operatorname{Attention}(Q,K,V)
 =\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V.
\]

- 单个 Head 的输入：`Q,K,V [S,D]`。
- `Scores = Q Kᵀ`：`[S,D] × [D,S] = [S,S]`。
- `Scale`：乘 `1/sqrt(D)`；`Softmax` 对 Scores **每一行**归一化；得到 `Weights [S,S]`。
- `Output = Weights × V`：`[S,S] × [S,D] = [S,D]`。
- 扩展到多 Head 时可理解为 `Q/K/V [B,H,S,D]`，每个 Head 独立计算；**当前源码没有实现多 Head 调度**。

### 4.2 Attention 的四个计算阶段

```text
Device: Q[S,D] / K[S,D] / V[S,D]
        │
        ├─ cuBLAS SGEMM #1  Scores = Q × Kᵀ       → [S,S]
        ├─ scale_kernel  Scores *= 1/sqrt(D)     → [S,S]
        ├─ softmax_kernel Stable Softmax(rows)   → [S,S]
        └─ cuBLAS SGEMM #2  Output = Weights × V  → [S,D]
```

各操作提交到默认 Stream（本测试中 cuBLAS Handle 也默认使用该 Stream），同一 Stream 的顺序保证依赖，不需要在每一步之后调用 `cudaDeviceSynchronize()`；最终进行同步和结果复制。

### 4.3 最容易写错的地方：行主序数据与 cuBLAS 列主序 API

Host/CUDA 数组使用 `X[row * cols + col]` 行主序，而 `cublasSgemm()` 依据列主序、leading dimension 参数解释内存。

本项目并未显式转置 Q、K、V，而是使用转置恒等式和参数交换获得相同的线性内存排列。

**计算 `Scores_row = Q_row K_rowᵀ`：**

```cpp
cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    S, S, D,
    &one,
    d_K, D,
    d_Q, D,
    &zero,
    d_scores, S);
```

**计算 `Output_row = Weights_row V_row`：**

```cpp
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    D, S, S,
    &one,
    d_V, D,
    d_weights, S,
    &zero,
    d_out, D);
```

可通过小矩阵手工计算和与 CPU Reference 对照验证映射，避免只靠记忆参数顺序。`leading dimension` 是列主序视角下列之间的元素跨度，不能忽略。

### 4.4 V4.3 实测正确性

| Shape (S,D) | 结果 | 最大绝对误差 |
|---|---|---:|
| `(1,8)` | PASS | `0` |
| `(4,16)` | PASS | `5.9604645e-08` |
| `(17,32)` | PASS | `5.9604645e-08` |
| `(64,64)` | PASS | `4.4703484e-08` |

`V4.3 Attention: ALL PASS`。单 Head、无 Mask、随机 FP32 `[-1,1]`，CPU FP64 中间结果，容差 `atol=2e-5, rtol=1e-4`。**此时只完成正确性，还未进行正式 Attention 性能对照。**

---

## 5. V4.4：Scale + Softmax Kernel Fusion

### 5.1 假设与目标

原始执行路径有两次 GEMM 加两个逐元素/行归一化 Kernel：

```text
BASELINE：QKᵀ GEMM → Scale Kernel → Softmax Kernel → PV GEMM
FUSION：  QKᵀ GEMM → Scaled Softmax Kernel         → PV GEMM
```

融合预期省去一次 GPU Kernel Launch，以及对 Scores 矩阵额外的一次读取和写回；减少中间数据流，但不能事先保证整体延迟一定降低。

令 `a=1/sqrt(D)>0`。数学上：

\[
 \operatorname{Softmax}(aX)_i
 = \frac{\exp((X_i-\max(X))a)}
         {\sum_j\exp((X_j-\max(X))a)}.
\]

FP32 下两条执行路径的舍入顺序可能有微小不同，所以验证 Baseline 和 Fusion **各自**与 CPU Reference 的误差，而非要求两个 GPU 版本逐位相等。

### 5.2 关键改动（真实实验代码的核心思想）

Baseline Softmax 内部是：

```cpp
float e = expf(X[base + col] - max_value);
```

融合版增加正的 `scale` 参数并改成：

```cpp
float e = expf((X[base + col] - max_value) * scale);
```

完整的 `scaled_softmax_kernel(X,Y,N,scale)` 仍沿用 `block_max`、`block_sum`、两个阶段的 Warp/Shared Memory 归约。**独立 `scale_kernel` 和原始 `softmax_kernel` 保留，作为公平的基线。**

注意：Baseline 会原地缩放 `d_scores`，Fusion 不缩放 `d_scores`。但每次 `attention_forward()` 先调用 `cublasSgemm(..., beta=0, d_scores)` 完整重写 Scores，因此两次路径运行不会因为上一次的原地缩放而污染结果。

### 5.3 `attention_forward(..., bool fused)`

```cpp
// 1. 两条路径共有：Scores = QKᵀ，cuBLAS SGEMM

float scale = 1.0f / std::sqrt(static_cast<float>(D));
if (fused) {
    scaled_softmax_kernel<<<S, 256>>>(d_scores, d_weights, S, scale);
    check_cuda(cudaGetLastError(), "fused softmax launch");
} else {
    scale_kernel<<<(S*S+255)/256, 256>>>(d_scores, S*S, scale);
    check_cuda(cudaGetLastError(), "scale launch");
    softmax_kernel<<<S, 256>>>(d_scores, d_weights, S);
    check_cuda(cudaGetLastError(), "softmax launch");
}

// 3. 两条路径共有：Output = Weights×V，cuBLAS SGEMM
```

真实函数保留原有完整的 cuBLAS 参数、设备指针、CUDA API 错误检查；不需要引入新的独立 Tensor 框架。

### 5.4 测量方法（一定要分清范围）

源码 `benchmark_attention()` 参数：

| 参数 | 值 |
|---|---:|
| Warmup | 10 次完整 Attention 调用 |
| REPEAT | 每个 Trial 连续调用 50 次 |
| TRIALS | 5 |
| 计时 | CUDA Event，在重复提交前后记录 |
| 统计 | 五次 Trial 的 **median** |
| 单位 | `elapsed_ms * 1000 / 50` = 每次平均 μs |
| 性能对象 | **整条 Attention 执行链路**，不是单独 Softmax Kernel |
| 不包含 | CPU Reference、cudaMalloc/Free、H2D/D2H、cuBLAS Handle 创建 |

**已知局限**：

1. Baseline 和 Fusion 总是按固定顺序分别测试，未交替对照或锁定机器负载。
2. 代码反复从 CPU 提交多个小 GPU 工作，而非 CUDA Graph；CUDA Event 度量其所在 GPU Stream 的耗时区间，可能包含 CPU 提交不够及时导致的设备空闲间隔。
3. RTX 2060 Laptop 存在动态功耗/频率管理以及操作系统/WSL2 干扰的可能；这只是可能来源，**本实验没有逐项测量其归因**。
4. 每次程序内的 5 Trial median 仍可能出现整轮偏离。下一节完整保留离群值，不挑选最好看的结果。
5. 没有独立测量融合前后 Softmax Kernel 的指令数、寄存器、Warp Stall，不可声称特定硬件瓶颈已被改善。

### 5.5 正确性结果

最终 V4.4 验证了两条路径（Baseline / Fusion）都在相同输入下通过 CPU Reference：

| Shape (S,D) | Baseline 最大误差 | Fusion 最大误差 | 结果 |
|---|---:|---:|---|
| `(1,8)` | 0 | 0 | PASS |
| `(4,16)` | `5.9604645e-08` | `5.9604645e-08` | PASS |
| `(17,32)` | `5.9604645e-08` | `5.9604645e-08` | PASS |
| `(64,64)` | `4.4703484e-08` | `4.4703484e-08` | PASS |
| `(128,64)` | `3.7252903e-08` | `3.7252903e-08` | PASS |
| `(256,64)` | `3.7252903e-08` | `3.7252903e-08` | PASS |

最终行：`V4.4 Attention Fusion: ALL PASS`。

### 5.6 **真实原始性能数据：全部四轮，不删除异常值**

单位：Latency=μs；Speedup=Baseline/Fusion。Run 0 是首次实验；Run 1～3 是后来追加的三次程序复测。每一组数字本身是该轮程序中 5 Trial 的中位数。

| 轮次 | S | D | Baseline μs | Fusion μs | Speedup |
|---|---:|---:|---:|---:|---:|
| Run 0 | 64 | 64 | 50.131 | 37.888 | 1.323× |
| Run 0 | 128 | 64 | 53.831 | 43.950 | 1.225× |
| Run 0 | 256 | 64 | 64.685 | 94.095 | **0.687×** |
| Run 1 | 64 | 64 | 45.147 | 36.532 | 1.236× |
| Run 1 | 128 | 64 | 48.852 | 40.960 | 1.193× |
| Run 1 | 256 | 64 | 71.274 | 51.720 | 1.378× |
| Run 2 | 64 | 64 | 45.907 | 82.133 | **0.559×** |
| Run 2 | 128 | 64 | 46.319 | 39.979 | 1.159× |
| Run 2 | 256 | 64 | 62.192 | 54.517 | 1.141× |
| Run 3 | 64 | 64 | 47.985 | 33.348 | 1.439× |
| Run 3 | 128 | 64 | 103.549 | 80.696 | 1.283× |
| Run 3 | 256 | 64 | 72.023 | 53.166 | 1.355× |

### 5.7 跨轮汇总（正确统计口径）

对于每种 Shape，先在每个程序运行中计算配对比值 `baseline / fused`，再对四个比值求中位数。不要把四轮任意挑选的最小 Baseline 和 Fusion 配成一对，也不要只保留胜出的数据。

| Shape | 4 轮 Fusion 获胜次数 | 四轮配对 Speedup 中位数 | Baseline μs 的跨轮中位数 | Fusion μs 的跨轮中位数 |
|---|---:|---:|---:|---:|
| `(64,64)` | 3/4 | **1.279×** | 46.946 | 37.210 |
| `(128,64)` | 4/4 | **1.209×** | 51.342 | 42.455 |
| `(256,64)` | 3/4 | **1.248×** | 67.980 | 53.842 |

注意：**配对加速比的中位数**与**两个延迟中位数之比**在数学上不必相同。本报告以配对 Speedup 中位数作为跨轮速度总结，完整原始数字保留在上表。

### 5.8 可以说什么、不能说什么

**已证实：**

- Fusion 不破坏当前六组测试的正确性。
- 本次实现减少一项独立 Scale Kernel Launch，省去一次完整的 Scores 全局读写（按算法数据流分析）；但并不保证所有访存事务及缓存行为一定等比例减少。
- 在该特定 RTX 2060 / WSL2 / 当前重复提交 Benchmark 中，大多数运行有加速趋势，三个 Shape 的中位配对加速比约 1.21～1.28×。
- 四轮测试中出现显著异常，分别为 `(S=256, Run0)` 及 `(S=64, Run2)`；`(S=128, Run3)` 的两个绝对延迟也同时升高。

**未证实：**

- 不能声称某一 Shape 天然更适合/不适合 Fusion；首次 S256 退化在后续三次并未重复。
- 不能声称已经将 GPU Kernel 本身优化了 20～30%，因为测量对象是整条 Attention 链路，并含调度因素。
- 不能确定异常来自 GPU 频率、WSL、CPU 负载、寄存器或 Cache；没有针对这些因素取证。
- 不能将当前单 Head FP32 小矩阵结论外推到大序列、多 Head、多 Batch、FP16/BF16 或 FlashAttention。

**工程决定**：两条路径都保留，Fusion 作为 V4 的功能性优化实验，并记录条件性性能收益与方法限制；不再为 V4 追加大量 Shape、Nsight 或参数调优。未来在 V6 真实模型需要时再用更严格的 Benchmark 和端到端数据进一步确认。

---

## 6. CUDA 基础与性能知识速查（复习版）

| 概念 | 必须真正理解的点 |
|---|---|
| `__global__` | Host 启动的 GPU Kernel 入口；`<<<grid,block>>>` 指定执行布局 |
| `__device__` | 供 GPU 侧代码调用的函数 |
| `blockIdx.x` | 负责哪一行；`threadIdx.x` 是 Block 内的线程编号 |
| Warp / Lane | 一个 Warp 32 线程；Lane 是 Warp 内编号；Warp Shuffle 不跨 Warp |
| `__shfl_down_sync` | 从指定偏移 Lane 取得 Warp 内寄存器值；注意参与掩码和有效 Lane |
| Shared Memory | Block 内跨 Warp 共享部分归约结果 |
| `__syncthreads()` | Block 级同步、保障相关 Shared Memory 写入可见，不能在不一致分支中错误使用 |
| Reduction 单位元 | Sum 用 0，Max 用 -∞，保证空处理线程不污染结果 |
| Stable Softmax | 先求最大值再减去，避免较大指数输入溢出 |
| `rsqrtf` | 计算 `1/sqrt(x)`，用于 RMSNorm；浮点舍入允许与 double Reference 有微小误差 |
| Gamma | 对最后一维特征缩放；大小 `[D]`，各行共享 |
| GEMM + leading dimension | cuBLAS 默认列主序，行主序计算通过转置/交换输入关系映射 |
| CUDA Stream 顺序 | 同一 Stream 中按顺序排队，保证前序 Kernel 数据写入后续可见 |
| CUDA Event | 记录某一 GPU Stream 时间区间；不要混同于 host-only 提交或完整 H2D/D2H 端到端耗时 |
| Median | 比单次值更抗离群干扰，但有限轮数的 median 也可能明显波动 |
| Fusion | 减少 Launch 和中间读写，可能改善小工作负载；也可能受其他成本限制 |
| `atol + rtol*abs(ref)` | FP32 并行计算的标准误差判据之一；也需检查 NaN/Inf |

### 6.1 计算题：Attention 中间显存

`S=256,D=64`，FP32：

- Q、K、V 各 `S*D=16,384` 元素、各 65,536 bytes（64 KiB）。
- Scores `[S,S]`：65,536 元素 × 4 = **262,144 bytes = 256 KiB**。
- Weights `[S,S]`：**256 KiB**。
- Output `[S,D]`：**64 KiB**。
- 仅此 6 个设备 Tensor 的有效元素总大小（Q/K/V/Scores/Weights/Output）为 `3×64 + 2×256 + 64 = 768 KiB`，不包含内存分配器额外开销、cuBLAS 内部 Workspace 或其他资源。
- 两次 GEMM 的理论 FLOPs 约为 `2×(2 S² D)=4S²D`，这里约 **16,777,216 FLOPs**，不包含 Scale/Softmax 的指数等额外操作。

### 6.2 面试：为什么 Fusion 可能更快但也可能更慢？

完整答法：Fusion 减少一个 Kernel Launch 和一轮 Scores 的读写，理论上降低启动及数据移动成本；但是 GPU 时间还受到工作负载大小、计算依赖、缓存行为、资源占用、内核实际指令实现和主机提交节奏影响。必须比较正确性和受控条件下的端到端/内核耗时，当前数据还显示明显波动，不能说每次都更快。

---

## 7. 开发命令与可复现步骤

### 7.1 构建目标

```bash
cd ~/projects/minigpu
cmake -S . -B build-v4 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build-v4 --target softmax_v4 rmsnorm_v4 attention_v4 --parallel 4
```

- 第一句进入工程根目录。
- 第二句为已有 CUDA 12.8 工程配置独立构建目录，不覆盖 V2/V3 的构建缓存。
- 第三句编译三项 V4 独立测试目标。
- 正常输出包括 `[100%] Built target ...`；具体构建百分比会随增量编译变化。

### 7.2 正确性测试与 Fusion Benchmark

```bash
./build-v4/softmax_v4
./build-v4/rmsnorm_v4
./build-v4/attention_v4 | tee /tmp/minigpu_v4_fusion.log
```

注意：`tee` 只是把控制台输出保存在临时日志文件中。若希望 Shell 在管道中正确反映程序失败，可在 Bash 启用 `set -o pipefail` 后再执行。

**预期关键输出文本**：

```text
V4.1 Softmax: ALL PASS
V4.2 RMSNorm: ALL PASS
V4.4 Attention Fusion: ALL PASS
```

`attention_v4` 最新版本的 `main()` 打印 `V4.4 ...`，因为它同时验证 V4.3 Baseline 和 V4.4 Fusion；V4.3 的单独基线版本历史运行已取得 `V4.3 Attention: ALL PASS`。

### 7.3 提交与发布（只在确认源文件在本地后执行）

```bash
cd ~/projects/minigpu
mkdir -p docs
# 将本报告下载/复制为 docs/V4_TRANSFORMER.md

git status --short
git add CMakeLists.txt softmax_v4.cu rmsnorm_v4.cu attention_v4.cu docs/V4_TRANSFORMER.md
git diff --cached --stat
git commit -m "feat: add transformer CUDA ops and attention fusion benchmarks"
git tag -a v4.0 -m "MiniGPU V4: Transformer operators and attention fusion"
git -c http.version=HTTP/1.1 push origin main
git -c http.version=HTTP/1.1 push origin v4.0
git status -sb
```

这些是**待执行步骤**，不表示本报告生成时已执行。本地状态及 GitHub 标签必须根据实际命令输出核实；不要重复创建已经存在的 tag。

若希望维护历史 V3 文档一致性，可以在下一次正常提交中将 V3 报告中的“尚未推送 v3.0”更新为已发布事实，但不是 V4 的算子开发依赖，也不需要重新打 V3 标签。

---

## 8. 面试验收（先自己作答，再看提示）

### 8.1 核心源码题

**题**：逐行解释以下逻辑为什么可以计算整行 RMSNorm：

```cpp
for (int col = tid; col < D; col += blockDim.x) {
    float x = X[base + col];
    local_sum += x * x;
}
float sum_sq = block_sum(local_sum);
float inv_rms = rsqrtf(sum_sq / D + eps);
```

**参考要点**：线程步长覆盖当前行全部特征并且不重叠；局部寄存器累加降低多线程共享写入冲突；`block_sum` 汇总 256 线程贡献；取平均、加 ε、rsqrt 得到共享归一化因子。

### 8.2 Tensor Shape/计算题

**题**：`Q,K,V=[2,4,128,64]`；计算 Scores、Weights、Output Shape，以及每个 Head 的 Scores FP32 字节数。

**参考答案**：Scores/Weights `[2,4,128,128]`，Output `[2,4,128,64]`；**单个 Head** Scores `128×128×4=65,536 bytes=64 KiB`。不要混淆单 Head 与整个 Batch、多 Head 的总内存。

### 8.3 工程故障题

**题**：为什么 `cublasSgemm()` 已经返回 `SUCCESS`，不能证明最终 Attention 的 GPU 结果正确？

**参考要点**：返回值主要反映 API 调用状态，不取代异步执行同步后的 CUDA 错误检查，也不代替语义级的数值一致性校验；矩阵布局/转置参数可能导致不崩溃但算错。

### 8.4 三道递进追问

1. **Warp Shuffle 与 Shared Memory 什么时候各自使用？** 参考要点：Warp 内寄存器交换 vs Block 跨 Warp 共享；Barrier 的目的。
2. **Softmax 为什么必须先减最大值？为什么归约初值是负无穷？** 参考要点：溢出与全负输入正确性。
3. **为什么 4 轮测量有 0.559× 和 1.439×，你仍然要保留全部原始数据？** 参考要点：噪声与测量边界、统计口径、可审计、不可随意声称稳定收益。

### 8.5 V4 是否通过学习验收？

- **工程验收**：已通过（上述正确性和性能实验有实机记录）。
- **面试口述/独立源码解释验收**：本阶段给了问题和答案方向，但**用户还没有在本对话中独立作答并通过评判**。不要把“看过解析”写成“掌握已验收”。
- **Git 交付验收**：截至本报告尚未验证 `v4.0` 发布，待执行提交/推送。

---

## 9. 给下一个 AI 的交接说明：直接进入 V5

### 9.1 请不要重复做的事

- 不要再安装 CUDA、重做 V2 GEMM、重跑 V3 全套 Nsight、反复做 Softmax/RMSNorm 性能调参。
- V4.1 Softmax 与 V4.2 RMSNorm 已经通过 CPU/GPU 数值测试。
- V4.3 的单 Head、无 Mask FP32 Attention 已通过 V4.3 原版四个 Shape 的正确性测试；V4.4 双路径六个 Shape 全部通过。
- Fusion 已做四轮性能实验并保存完整数据；无需为了让某次结果“更漂亮”继续筛选。
- 保留 Baseline 和 Fusion 两条路径；`attention_v4.cu` 仍是完整的可运行测试程序。

### 9.2 当前仍需要收尾的事项

1. 将 `V4_TRANSFORMER.md` 放进 WSL 工程 `docs/`。
2. Git 检查并提交真实源文件和报告，创建并推送 `v4.0` 标签。
3. 可选择更新 V3 报告中已过期的 Git 发布状态（顺手做，不为此引入独立任务）。
4. 如果需要严格满足最初项目计划中的“独立口述验收”，抽取第 8 节源码题、计算题、工程题和递进追问进行一次真实面试练习。

### 9.3 V5 最小实施方案（只做核心能力）

1. 以现有 Attention/Softmax 或 GEMM 工作负载为用例，实现 **RAII GPU Buffer**（所有权、移动语义、自动释放）。
2. 加入**最小显存复用**，避免执行每次算子都重复 `cudaMalloc/cudaFree`；用实际使用路径验证，不做虚构 Runtime 接口。
3. 加入 CUDA **Stream + Event**，正确表达执行顺序、同步和计时；先单 Stream，只有证据支持时才扩展并发。
4. 比较真实执行链路与原版的正确性和适用的延迟/开销指标。整理 `V5_RUNTIME.md`，提交/发布 `v5.0`。

不提前做复杂调度器、多 GPU、NCCL、MLIR、KV Cache 或工业级 Tensor 库。V6 再考虑把 CUDA Kernel 接入真实小型 Transformer 模型做端到端性能优化。

### 9.4 真实性与面试边界

- 目前 V4 的优化是 **Scale + Softmax 融合**，不是 FlashAttention，也不是融合整个 Attention。
- 当前性能提升是**测得的条件性结果**，不是跨环境、跨 GPU、跨 Shape 的通用保证。
- V4 的最终代码和 `v4.0` 是否在 GitHub 上，必须以实际推送结果为准，不能以这份交接文档的存在代替发布证据。
- 符合求职价值的叙述方式：亲手实现 CUDA Reduction/Softmax/RMSNorm、串接 cuBLAS 单 Head Attention、一次数学正确的融合实验、CPU/GPU 数值核验、记录性能收益和测量局限。

---

## 10. 一句话记忆

**V4 = Warp/Block Reduction → Stable Softmax → RMSNorm → cuBLAS Attention Forward → Scale+Softmax Fusion → 实机正确性与带局限的性能对照。**

下一步：**只需文档入库 + `v4.0` 发布，之后进入 V5 轻量级 GPU Runtime。**
