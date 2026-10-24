# MiniGPU V2 — GEMM 优化原理与工程复盘

> 文档定位：解释已经完成的 **Naive GEMM → Shared Memory Tiled GEMM → Register Blocking 1×2 → TILE 调优 → cuBLAS 对照**，对应 MiniGPU V2 的真实 CUDA 源码与实测结果。  
> 测试平台：NVIDIA GeForce RTX 2060（Turing / SM75，6 GiB）、WSL2 Ubuntu 24.04.5、CUDA Toolkit 12.8.93、NVIDIA Driver 591.74。  
> 数据来源：开发者实际运行的 `gemm_bench_unified.cu` 输出。完整原始测试条件见 [`PERFORMANCE.md`](PERFORMANCE.md)。  
> 项目性质：学习与性能分析工程，**不宣称达到通用高性能 GEMM 库的水平**。

## 1. 问题定义：我们在优化什么？

目标是 FP32 矩阵乘法：

\[
C_{M\times N}=A_{M\times K}\times B_{K\times N},\qquad
C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}B_{k,j}
\]

工程中 A、B、C 都采用 **Row-major（行优先）** 连续存储：

```cpp
A[row * K + k]
B[k * N + col]
C[row * N + col]
```

其中 `M` 是输出行数、`N` 是输出列数、`K` 是归约维度。GEMM 的惯用运算量估计为 `2×M×N×K` FLOPs（一次乘法和一次加法分别计 1 FLOP）。

### 1.1 性能不仅取决于 FLOPs

GPU 处理 GEMM 时，实际延迟同时受以下因素影响：

- **Global Memory 数据流量**：相同 A/B 元素是否被重复加载。
- **Shared Memory 复用**：线程块是否把全局内存加载来的数据重复利用。
- **寄存器级复用**：一个线程能否用同一 A 元素更新多个 C 累加器。
- **线程与资源映射**：Block 线程数、寄存器使用量、Shared Memory 用量、可并驻 Block 数。
- **同步及控制开销**：`__syncthreads()`、循环、边界判断与启动开销。
- **尺寸与硬件运行状态**：小矩阵、大矩阵和非整除尺寸的最优策略可能不同。

后文区分 **理论访存复用推导** 和 **已经实测的性能**。理论量不能直接当成 GPU 硬件计数器的结果。

---

## 2. V2.1：Naive GEMM——建立正确性与性能基线

### 2.1 线程映射

当前统一 Benchmark 中的 Naive 采用一维 Grid，**一个 CUDA 线程计算一个输出元素**：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= M * N) return;

int row = idx / N;
int col = idx % N;

float sum = 0.0f;
for (int k = 0; k < K; ++k)
    sum += A[row * K + k] * B[k * N + col];

C[row * N + col] = sum;
```

在统一测试中，Naive 的 Block 使用 **128 threads**，Grid 大小为 `ceil(M*N/128)`。

**优点：** 代码直观，边界处理简单，容易建立 CPU Reference 与测试框架；小矩阵时不必承担 Shared Memory 搬运和块内同步。

**限制：** 从算法角度看，计算相邻 C 元素的不同线程会反复访问 A、B 中的相同元素。硬件缓存能缓解一部分请求，因此不能把源代码中的所有 Load 简单等同于 DRAM 访问。

### 2.2 为什么它是基线？

优化指标必须有明确参照：

\[
\text{自定义优化加速比}
=\frac{T_\text{Naive}}{T_\text{Optimized}}
\]

这项指标只衡量已实测时间变化；正确性验证和 Benchmark 口径必须一致，才能比较。

---

## 3. V2.2：Shared Memory Tiling——减少重复全局访存

### 3.1 线程块负责一个 C Tile

对于 `TILE=16`：

- Block 为 `dim3(16,16)`，即 **256 个线程**。
- 一个 Block 计算 C 的 `16×16` 输出区域。
- 每轮沿 K 维度推进 16 个元素。
- 使用 `As[16][16]` 和 `Bs[16][16]` 暂存 A/B 输入块。

计算流程：

```text
Global A、Global B
       │
       ▼
线程协作加载 16×16 的 As、Bs
       │
       ▼
__syncthreads()  等待 Tile 装载完毕
       │
       ▼
每个线程循环 k=0..15，累加自己的 C[row,col]
       │
       ▼
__syncthreads()  避免下一轮覆盖仍在使用的 Tile
       │
       └────── 沿 K 维度重复 ──────► 写回 Global C
```

核心代码（从统一工程中提取）：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

As[ty][tx] = (row < M && aCol < K)
    ? A[row * K + aCol] : 0.0f;
Bs[ty][tx] = (bRow < K && col < N)
    ? B[bRow * N + col] : 0.0f;

__syncthreads();
for (int k = 0; k < TILE; ++k)
    sum += As[ty][k] * Bs[k][tx];
__syncthreads();
```

### 3.2 两次同步分别保护什么？

1. **第一次同步（装载后）**：确保 Block 内其他线程所负责的 As/Bs 元素已经写入，当前线程才能读取整个 Tile。
2. **第二次同步（计算后）**：确保所有线程完成本轮 Tile 读取，下一轮才能安全覆盖 Shared Memory。

**重要边界规则：** 使用 `__syncthreads()` 的 Kernel 不应让部分线程因 `row>=M` 或 `col>=N` 而在同步前提前退出，否则可能造成线程块中的同步行为不一致。本实现让边界线程参与同步，通过越界读置零、越界写跳过来保持正确性。

### 3.3 共享内存占用与理论复用

每个 Block 的静态 Shared Memory：

\[
(16^2+16^2)\times 4\text{ B}=2048\text{ B}=2\text{ KiB}
\]

把 `T×T` 的一个 K 分块作为理想化比较：

- Naive 不考虑缓存时，需要约 `2T³` 次标量 A/B 读取才能计算该 C Tile。
- Tiled 协作加载仅需 `2T²` 个标量到 Shared Memory，再从 Shared Memory 复用。
- 因此理论上**该分块的全局加载次数**可减少约 `T` 倍（T=16 时约 16 倍）。

这里的“约 16 倍”**不是**整个 GEMM 的实际 DRAM 带宽降低 16 倍，也不是性能提升 16 倍：缓存命中、访存合并、同步、指令数和算力都影响实际结果。

对于忽略 C 读写及边界的单个 K Tile，理想算术强度：

\[
AI_\text{Tiled}=\frac{2T^3}{(2T^2)\times4\text{ B}}=\frac{T}{4}\ \text{FLOP/B}
\]

当 `T=16`，得 `4 FLOP/B`。这是简化分析，不是 Nsight 实测值。

---

## 4. V2.4：Register Blocking 1×2——每线程计算两个 C

### 4.1 与 Tiled 的本质区别

Tiled16 中，每个线程负责：

```cpp
float sum = 0.0f;
sum += As[ty][k] * Bs[k][tx];
```

Register2 中，每个线程负责同一行、不同列的 **两个输出**：

```cpp
float sum0 = 0.0f;
float sum1 = 0.0f;

for (int k = 0; k < TILE; ++k) {
    float a = As[ty][k];
    sum0 += a * Bs[k][tx];
    sum1 += a * Bs[k][tx + TILE];
}
```

相同的 `a` 被两个乘加复用；两个 `sum` 是线程私有的 FP32 累加器，通常保存在寄存器中。**这是 Register Blocking 的直接收益：提高每次加载 A 数据后所完成的计算量。**

### 4.2 输出坐标为什么要改？

```cpp
int row  = blockIdx.y * TILE + ty;
int col0 = blockIdx.x * (2 * TILE) + tx;
int col1 = col0 + TILE;
```

因此一个 Register2 Block 的输出区域是 **`TILE × (2*TILE)`**：

- `TILE=16`：输出 `16×32`，Block `16×16=256` 线程，总计最多 512 个输出。
- `TILE=32`：输出 `32×64`，Block `32×32=1024` 线程，总计最多 2048 个输出。

注意：**一个线程产生 2 个输出，不等于一个 Block 的线程数翻倍。**

### 4.3 为什么 B 的 Shared Memory 需要两倍宽度？

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][2 * TILE];
```

因为一个线程需要分别读取两个列位置的 B 值：

- `Bs[k][tx]` 用于更新 `sum0`。
- `Bs[k][tx + TILE]` 用于更新 `sum1`。

静态 Shared Memory 占用：

| 配置 | As | Bs | 合计 |
|---|---:|---:|---:|
| Tiled16 | 1024 B | 1024 B | **2048 B (2 KiB)** |
| Register2 TILE16 | 1024 B | 2048 B | **3072 B (3 KiB)** |
| Register2 TILE32 | 4096 B | 8192 B | **12288 B (12 KiB)** |

### 4.4 为什么 Register2 可能进一步提高数据复用？

单个 `T×2T` 输出 Tile、单个长度 T 的 K 分块，理论上：

- 乘加运算量：`2 × (T×2T×T)=4T³ FLOPs`。
- 需要协作装入 A 的 `T²` 个 float 和 B 的 `2T²` 个 float。
- 输入全局访问体积约 `(3T²)×4=12T² B`。

理想化算术强度为：

\[
AI_\text{Register2}=\frac{4T^3}{12T^2}=\frac{T}{3}\ \text{FLOP/B}
\]

它比相同 T 下 `T/4` 的简单 Tiled 设计提高约 **4/3 倍**。该估算忽略输出写回、边界、缓存和实际机器指令，**只能解释复用方向，不能充当性能测量结论**。

---

## 5. 资源权衡：TILE16 与 TILE32 为什么没有绝对赢家？

### 5.1 已实现配置对照

| Kernel | Block 线程 | 单 Block 输出区域 | 静态 Shared Memory |
|---|---:|---|---:|
| Tiled16 | 256 | 16×16 | 2 KiB |
| Register2 TILE16 | 256 | 16×32 | 3 KiB |
| Register2 TILE32 | 1024 | 32×64 | 12 KiB |

之前单独使用 `nvcc -O3 -arch=sm_75 -Xptxas=-v` 做过资源统计：Tiled16 约 39 registers/thread、Reg2 TILE16 约 52、Reg2 TILE32 约 54，均未报告 Register Spill。**这些是当时特定编译方式的记录，不是当前统一 CMake 目标重新 Profiling 的结果**；正式分析应对最终二进制重新核对编译选项和实际寄存器资源。

### 5.2 Occupancy 不等于性能

Occupancy 指一个 SM 上活跃 Warp 数与架构允许最大活跃 Warp 数的比值，受线程数、寄存器数、Shared Memory、Block 数量等资源约束。

- TILE32 一块有 1024 线程，在 SM75 上已经接近/达到单 SM 并驻线程数上限，因此一个大 Block 就可能占满线程名额。
- 即使理论 Occupancy 很高，运行速度仍取决于访存、指令依赖、Block 数量和并行任务规模。
- 大 TILE 可能有更好的输入复用，但也意味着更大的 Block、更高资源需求以及小矩阵时更多无效线程。

不能仅凭 `registers/thread` 或理论 Occupancy 就预测最终速度；需要结合实际延迟和 Nsight 指标。

### 5.3 为什么 K、M、N 非整除也能 PASS？

各个 Kernel 对边界进行条件加载或写回：越界输入以 0 代替，越界输出不写；Grid 使用上取整。比如 `M=130, K=65, N=70`：

- Tiled16：输出网格 `ceil(70/16) × ceil(130/16)`。
- Register2 TILE16：`ceil(70/32) × ceil(130/16)`。
- Register2 TILE32：`ceil(70/64) × ceil(130/32)`。

部分 Block 计算了边界之外的线程位置，但不应越界读取或写入。真实测试中，这组非整除矩阵的五种 GEMM 全部通过。

---

## 6. 五组 Shape 的性能验证（随机 FP32）

**测试口径：**固定种子 `2026`，CPU FP64 累加作为 Reference 并转回 FP32；`atol=2e-5`、`rtol=1e-4`；每个 Graph 64 GEMM，正式计时每轮回放 16 Graph，5 轮取中位数。所有五种实现共用输入和 Stream。详细条件见 [`PERFORMANCE.md`](PERFORMANCE.md)。

### 6.1 延迟总表（μs，越低越好）

| Shape（M×K×N） | Naive | Tiled16 | Register2 16 | Register2 32 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 64³ | 5.092 | **4.986** | 6.041 | 13.155 | 6.955 |
| 128³ | **9.306** | 9.599 | 10.229 | 12.672 | 5.084 |
| 256³ | 60.234 | 42.888 | **39.179** | 50.580 | 15.838 |
| 512³ | 502.394 | 320.928 | 274.126 | **265.238** | 70.168 |
| 130×65×70 | 4.031 | **3.997** | 4.064 | 9.906 | 4.455 |

**粗体**只标记本次最好的 *自定义 Kernel*。表中的每行是单次独立进程的 Benchmark 结果；小 Shape 的绝对延迟存在运行间波动。  
**正确性：** 5 Shape × 5 实现 = **25/25 PASS**，只证明所测范围与上述容差下通过。

### 6.2 大矩阵 512³ 的重点计算

`2×512³ = 268435456 FLOPs`。

| 版本 | Latency | GFLOPS | 对 Naive 加速 |
|---|---:|---:|---:|
| Naive | 502.394 μs | 534.312 | 1.00× |
| Tiled16 | 320.928 μs | 836.435 | ≈1.57× |
| Register2 TILE16 | 274.126 μs | 979.241 | ≈1.83× |
| Register2 TILE32 | 265.238 μs | **1012.055** | **≈1.89×** |
| cuBLAS | 70.168 μs | 3825.628 | 对 Reg2 TILE32 ≈**3.78×** |

公式：

\[
\text{Speedup}_{\text{Reg32 vs Naive}}=\frac{502.394}{265.238}\approx1.89\times
\]

\[
\text{Speedup}_{\text{cuBLAS vs Reg32}}=\frac{265.238}{70.168}\approx3.78\times
\]

**结论：** 在这次 512³ 测试中，自己实现的 Register Blocking 明显优于 Naive，但仍显著慢于 cuBLAS。与库的差距是下一阶段 Profiling 的研究对象，而非已经得到根因的性能诊断。

### 6.3 小矩阵与非整除矩阵的结论

- `64³`：Tiled16 的当次延迟较低。三轮独立复测中，相对 cuBLAS 的比值较稳定，但运行间的绝对延迟曾接近相差一倍，**不能推广为所有小矩阵中自定义 Kernel 都更快**。
- `128³`：Naive 是当次最快自定义实现；Tiled 和 Register Blocking 的同步/调度等成本可能尚未摊薄。
- `256³`：Register2 TILE16 比 TILE32 更快，说明更大的 Tile 不是线性收益。
- `130×65×70`：Naive、Tiled16、Register2 TILE16 接近，多次运行的最快自定义实现在三者之间变化，**不宣称存在稳定单一冠军**。

这也是实际推理 Runtime 为何会根据 `M/N/K` 选择不同 Kernel 的一个直观动机；本项目**还没有实现自动 Kernel Dispatch**。

---

## 7. 与 cuBLAS 的比较：不仅是算法，也包括计时口径

### 7.1 Row-major 输入如何交给 cuBLAS？

cuBLAS 传统 `cublasSgemm()` 按 Column-major 理解矩阵。项目使用：

\[
C=AB\quad\Longrightarrow\quad C^T=B^TA^T
\]

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
```

Column-major 的 `Cᵀ[N,M]` 与 Row-major 的 `C[M,N]` 有相同的连续内存布局，因此**不需要额外申请转置缓冲区，也不需要启动转置 Kernel**。

### 7.2 为什么采用 CUDA Graph？

普通循环：CPU 连续发起 1000 次 Kernel 调用；对微秒级短 Kernel，CPU 提交间隔可能使 GPU 出现空档。CUDA Graph 先捕获固定的 GPU 操作序列，然后反复回放，降低 Host 逐次提交的影响。

本项目采用：

```text
20 次普通预热
  → Capture 64 次 GEMM
  → Instantiate GraphExec
  → 3 次 Graph 预热
  → Event Start
  → Replay 16 次 Graph（合计 1024 GEMM）
  → Event Stop / Synchronize
  → 总耗时除以 1024
  → 重复 5 轮，取中位数
```

这里测得的是 **CUDA Graph 重复回放场景的平均单次 GEMM 时间**，不是 Host 端调用一次 API 的耗时、不是含 H2D/D2H 的端到端延迟，也不能声称已经完全消除调度影响。

### 7.3 为什么不直接声称 cuBLAS 使用了 Tensor Core？

cuBLAS 会针对硬件和 Shape 选择优化实现，但目前**未用 Nsight Compute / Nsight Systems 抓取 cuBLAS 实际 Kernel 名称、指令和 Tensor Core 利用率**。RTX 2060 为 Turing / SM75；不能仅根据峰值 GFLOPS 推定 cuBLAS 采用了何种底层算法或指令路径。

---

## 8. 正确性测试中遇到的真实工程问题

### 8.1 规律输入的局限

早期输入使用 `(i%7-3)*0.25f` 等二进制可精确表示的规律数值，多个 Kernel 可出现 `max_diff=0`。之后改为固定种子随机 FP32，才能更好地检验一般舍入误差和潜在索引问题。

### 8.2 CPU FP32 并非高精度 Reference

采用 `float sum` 的 CPU Reference 时，cuBLAS 某元素略超出原始容差。对异常元素改用 `double` 单独计算后发现 CPU FP32、GPU FP32 均与 FP64 有小误差。之后把完整 CPU Reference 改为 **FP64 乘加，输出仍为 FP32**。

最终判定使用：

```cpp
fabs(gpu - ref) <= 2e-5f + 1e-4f * fabs(ref)
```

这里 `atol` 应对近零元素，`rtol` 应对较大元素；两者均为**当前实验范围的选定阈值**，不应无条件套用到任何 GEMM。可进一步记录 ULP、平均误差和归一化最大误差以提升验收标准。

### 8.3 不小心把 Naive 改成 FP64 会怎样？

调试时曾将 GPU `gemm_naive` 的累加变量误改为 `double`，512³ 延迟由约 0.5 ms 上升至约 3.96 ms。后来恢复为 `float sum`，性能恢复到约 0.50 ms。这里展示了 **数值精度与硬件吞吐能力的工程权衡**，也说明修改 Reference 时必须区分 CPU 与 GPU Kernel 的职责。

### 8.4 编译失败不等于旧可执行文件消失

曾发生编译失败后仍运行了 `build-v2/gemm_bench_unified`：Linux 保留上次成功编译的可执行文件。为避免误把旧结果当新结果，推荐：

```bash
cmake --build build-v2 --target gemm_bench_unified --parallel 4 \
  && ./build-v2/gemm_bench_unified 512 512 512
```

---

## 9. 尚未完成的性能诊断与下一阶段方向

当前 V2 实际完成：多种 CUDA GEMM、cuBLAS 对照、五 Shape 验证、固定随机数 FP32、CPU FP64 参考值、Graph Benchmark。

以下是**后续研究计划**，并非 V2 已实现成果：

- 使用 **Nsight Compute** 检查 DRAM/L2 读写、Memory Throughput、Warp Stall、Achieved Occupancy 和寄存器使用。
- 使用 **Nsight Systems** 观察 CPU 提交、Stream 时间线、Graph Replay 与 GPU 空闲间隔。
- 对最终 CMake 编译产物复核 ptxas 报告，必要时比较不同编译选项。
- 在 Profiling 支持下考虑每线程 `2×2` / `4×4` 输出块、Vectorized Load、双缓冲/软件流水线，而不是先猜测某一优化必然有效。
- 更丰富的数值测试：不同随机种子、K、输入量级、正负抵消、特殊值；将 FP64 Reference 以 double 保存以做更精细统计。
- 将基于 Shape 的 Kernel Dispatch 作为后续可能的 Runtime 练习，而不是把当前 `main()` 中的固定顺序测试当作自动调度器。

## 10. 复现与验收清单

```bash
cd ~/projects/minigpu
cmake -S . -B build-v2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build-v2 --target gemm_bench_unified --parallel 4

for shape in '64 64 64' '128 128 128' '256 256 256' \
             '512 512 512' '130 65 70'; do
  read -r M K N <<< "$shape"
  ./build-v2/gemm_bench_unified "$M" "$K" "$N" || exit 1
done
```

验收检查：

- `gemm_naive()` 为 `float sum`，`gemm_cpu()` 为 `double sum`；**不要把二者混淆**。
- 运行时打印 `Random FP32, seed=2026`。
- 每组 Shape 的五个 Kernel 均输出 `PASS`。
- Graph 参数为 64 GEMM/Graph、16 Replay、5 Trial。
- 记录 `nvidia-smi` 的驱动版本与 `nvcc --version` 的 Toolkit 版本；WSL 可直接使用 `/usr/lib/wsl/lib/nvidia-smi`。
- 正式 Git 提交前审查 `git status`，不要把 `build-v2/`、临时压缩包和不必要的重复实验备份一起提交。

---

## 11. 面试表达：把项目讲成一条技术因果链

> 我在 RTX 2060 的 CUDA 12.8 环境中实现了 FP32 GEMM 基线，并针对重复访存引入 16×16 Shared Memory Tiling；之后让每个线程计算两个输出，引入双累加器的 Register Blocking，并比较 TILE16/TILE32 在不同 Shape 下的资源与性能权衡。我用固定随机 FP32、CPU FP64 参考值验证五种实现，并通过 CUDA Graph 的统一计时机制与 cuBLAS 对照。在 512³ 下，最快自定义 Kernel 的延迟由 Naive 的 502.394 μs 降到 265.238 μs，约 1.89 倍加速；但 cuBLAS 仍约快 3.78 倍。我能解释优化的数据复用原理，也明确当前尚未用 Nsight 定位更深层瓶颈。

以上表述只覆盖已经实现并验证的工作；不能将后续 Runtime、Attention、Nsight Profiling 或更高级 GEMM 优化说成 V2 已完成。
