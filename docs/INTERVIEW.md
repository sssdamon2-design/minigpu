# MiniGPU V2 · CUDA / AI Systems 面试复盘手册

> 项目：MiniGPU（CUDA GEMM 从 Naive、Shared Memory Tiling 到 Register Blocking，再到 cuBLAS / CUDA Graph Benchmark）  
> 面向：CUDA / GPU Kernel / AI Systems / 推理性能优化实习  
> 版本：V2 结项复盘，2026-10-25  
> 使用方法：**先遮住参考答案口头回答，再看答案；能够画出索引与数据流，比背诵术语重要。**

## 0. 项目事实卡（面试之前先背准确）

| 项目 | 已完成、可以直接陈述的事实 |
|---|---|
| 硬件 | NVIDIA GeForce RTX 2060 Laptop，6 GiB 显存，Turing / SM75 |
| 系统 | Windows 主机、WSL2 Ubuntu 24.04.5 LTS |
| 软件 | NVIDIA Windows 驱动 591.74，CUDA Toolkit / nvcc 12.8.93，CMake |
| 数据类型 | A、B、C 为 FP32；CPU Reference 使用 FP64 乘法与累加，再转成 FP32 输出 |
| GPU 实现 | Naive；Shared Memory Tiled16；Register2 TILE16；Register2 TILE32 |
| 对照实现 | `cublasSgemm`，按 Row-major → Column-major 转置视角映射 |
| 正确性 | 固定种子 2026、随机输入 [-1,1]、五组形状 × 五实现 = **25/25 PASS**；`atol=2e-5, rtol=1e-4` |
| Graph 计时 | 一个 Graph 中捕获 64 次 GEMM；每轮回放 16 次；测量 5 轮取中位数；预热排除于计时之外 |
| 512³ 结果 | Naive **502.394 μs**；Register2 TILE32 **265.238 μs**；cuBLAS **70.168 μs** |
| 512³ 核心指标 | 自定义优化 **1.89×**（Naive / Reg2 TILE32），最快自定义约 **1.012 TFLOPS**；cuBLAS 比其快 **3.78×** |

**注意三条边界**：

1. `nvidia-smi` 显示的 CUDA 13.1 是驱动支持版本；本项目真正的 CUDA **编译工具链版本为 12.8.93**。
2. 25/25 PASS 是对所测五组形状、输入分布与容差的**实测验收**，不是任意输入的形式化正确性证明。
3. 小矩阵延迟存在跨独立进程波动，不要将一次运行中的不足 1 μs 的领先当作普适性能优势。

### 五组 Shape 的代表性数据（随机 FP32 / CUDA Graph）

下表取每个 Shape 首次完成随机 FP32 验收时同一次运行中的结果；**不是统一多个独立进程取出的稳定中位数**。单位为 μs。

| Shape（M×K×N） | 最快自定义版本 | 自定义延迟 | cuBLAS 延迟 | cuBLAS / 自定义（延迟比） |
|---|---|---:|---:|---:|
| 64×64×64 | Tiled16 | 4.986 | 6.955 | 0.717× |
| 128×128×128 | Naive | 9.306 | 5.084 | 1.830× |
| 256×256×256 | Register2 TILE16 | 39.179 | 15.838 | 2.474× |
| 512×512×512 | Register2 TILE32 | 265.238 | 70.168 | 3.780× |
| 130×65×70 | Tiled16 | 3.997 | 4.455 | 0.897× |

对于 64³，后续三次独立进程复测的 Tiled16 延迟为 **5.026 / 2.657 / 2.669 μs**，cuBLAS 为 **7.002 / 3.722 / 3.693 μs**，说明**绝对延迟波动显著**，尽管在这三次实验中相对排序一致。130×65×70 中 Naive、Tiled16 和 Register2 TILE16 的排序则会变化。

---

# 第一部分：面试必答 25 题（先答题，再看参考答案）

## A. 基础 CUDA 与索引（Q1—Q5）

### Q1. 讲清楚 Thread、Block、Grid、Warp 和 SM 的关系。

**参考回答**：Kernel 启动生成 Grid；Grid 包含多个 Block；Block 包含线程。GPU 按硬件资源将 Block 分配到 SM 上执行，同一 Block 的线程可通过 Shared Memory 交换数据，能使用 `__syncthreads()` 做 Block 级同步。NVIDIA Warp 通常是 32 个线程，Warp 是 SIMT 执行/调度的重要单位。**Block 不等于 SM，一个 SM 在资源允许时可同时驻留多个 Block。**

**连续追问**：① `blockDim` 与 `gridDim` 谁在 CPU 启动时指定？② Warp 跨两个 Block 吗？③ 能否用 `__syncthreads()` 同步两个不同 Block？

### Q2. `idx = blockIdx.x * blockDim.x + threadIdx.x` 有什么作用？为什么必须有边界判断？

**参考回答**：用 Block 编号、Block 尺寸和线程在 Block 内编号，生成一维全局索引 `idx`。Grid 往往按向上取整配置，尾部线程可能超出 `M*N`，所以应检查 `idx < M*N`，避免非法读写。

**计算练习**：`M=130,N=70`，`threads=128`，需要 `ceil(9100/128)=72` 个 Block，共 9216 个线程，尾部无效线程 **116** 个。

### Q3. Row-major 的 A、B、C 下标为什么分别是 `row*K+k`、`k*N+col`、`row*N+col`？

**参考回答**：Row-major 二维矩阵 `[rows, cols]` 的元素 `(r,c)` 地址偏移是 `r*cols+c`。所以 A 是 `[M,K]`，B 是 `[K,N]`，C 是 `[M,N]`，每一行跨度分别为 K、N、N。

**追问**：如果 B 下标错写成 `k*K+col`，在何种情形可能暂时未暴露？答：例如 K=N 时错误可能被形状巧合掩盖；因此要测试非方阵与不能整除 TILE 的尺寸。

### Q4. 你的 Naive GEMM 为什么通常不如 Tiled GEMM？

**参考回答**：Naive 让每个线程计算一个 C 元素，每次循环直接从 Global Memory 读 A、B；相邻线程可能重复请求许多相同数据。Cache 可能缓解重复读取，所以不能武断地说所有访问都打到 DRAM。Tiled 将两个输入子块搬到 Shared Memory，让 Block 内线程复用，减少对 Global Memory 的重复请求。

**追问**：Naive 是“完全不合并访问”吗？答：不是，是否合并取决于同一条访存指令中 Warp 的线程访问地址，不能仅凭“Naive”标签判断。

### Q5. `<<<grid, block, 0, stream>>>` 四个参数分别是什么？

**参考回答**：Grid 配置、Block 配置、每 Block 动态 Shared Memory 字节数、Stream。第三项的 0 **不表示没有 Shared Memory**：源码中 `__shared__ float As[...]` 是**静态 Shared Memory**，不由第三个参数指定。

**追问**：`<<<grid,block>>>` 与显式传 `stream` 对 CUDA Graph Capture 有何影响？答：本项目需要把 Kernel 提交到正在捕获的同一个非阻塞 Stream 中。

## B. Shared Memory / Register Blocking（Q6—Q11）

### Q6. Tiled16 的一次 Tile 循环到底做了什么？

**参考回答**：每个线程加载一项 A 和一项 B 到 `As[ty][tx]`、`Bs[ty][tx]`；首次 `__syncthreads()` 保证同 Block 的数据加载完成；对 K 方向 16 个元素进行乘加；第二次同步避免部分线程开始覆盖共享数组，而其他线程仍在读上一轮 Tile。

**代码题**：如果 K=65，TILE=16，循环次数是 **5**，最后一轮越界位置用 0 补齐。

### Q7. 为什么不能把两个 `__syncthreads()` 随意删除？

**参考回答**：第一个确保读取的数据已写入 Shared Memory；第二个确保上一轮的数据消费结束，防止下一轮写入覆盖尚未读完的数据。它们不是“等所有 GPU 的 Block”：同步范围只在**当前 Block**。

**追问**：如果某些线程在 `__syncthreads()` 前直接 `return`，有何风险？答：可能导致 Block 内线程同步不一致；需要避免在需要同步的循环前让部分线程跳过同步。

### Q8. Tiled16 与 Register2 的一个 Block 分别输出多少元素？

**参考回答**：Tiled16 Block 16×16=256 线程，每线程输出 1 个结果，对应最多 **16×16**；Register2 TILE16 同样 256 线程，每线程输出 2 个结果，对应最多 **16×32**；Reg2 TILE32 是 1024 线程，输出最多 **32×64**。边界 Block 的有效输出可能更少。

### Q9. 为什么 Register2 每线程计算两列可能加速？

**参考回答**：同一行的一个 A 值 `a=As[ty][k]`，可以同时用于 `sum0+=a*Bs[k][tx]` 和 `sum1+=a*Bs[k][tx+TILE]`，提升 A 数据在寄存器内的复用。代价是更多寄存器、Shared Memory B 子块，以及资源/并行性变化。

**追问**：每线程两个输出就一定比一个输出快吗？答：不是，小矩阵、较大 Block、寄存器压力可能使其更慢。

### Q10. 手算三个 Kernel 的静态 Shared Memory 占用。

**参考计算**：

- Tiled16：`(16×16 + 16×16)×4 = 2048 B = 2 KiB`。
- Register2 TILE16：`(16×16 + 16×32)×4 = 3072 B = 3 KiB`。
- Register2 TILE32：`(32×32 + 32×64)×4 = 12288 B = 12 KiB`。

**追问**：为什么 Shared Memory 用得少也不保证更快？答：决定速度的还有线程数、寄存器占用、并行度、访存方式、同步和指令效率。

### Q11. 为什么 TILE32 在 512³ 最快，但在 64³ 很慢？

**参考回答**：更大 TILE 增加数据复用，也增加一个 Block 的线程数（Reg2 TILE32 达到 1024）、Shared Memory 和寄存器等资源需求；对小矩阵可能使 Block 数量少、有效线程比例低、固定开销更显著。512³ 计算量更大，更有机会摊薄开销并发挥复用。**这是与测量结果一致的解释候选；需要 Nsight Profiling 才能确认主瓶颈。**

## C. GPU 资源与性能分析（Q12—Q16）

### Q12. Occupancy 是什么？为什么不是越高越好？

**参考回答**：一般指每 SM 活跃 Warp 数与硬件最大允许活跃 Warp 数之比。受线程/Block、寄存器、Shared Memory、Block 上限等因素约束。Occupancy 是并行资源驻留指标，不等于 SM 利用率或吞吐量；较低 Occupancy 如果能带来更好的数据复用/指令级并行，也可能更快。

**追问**：`52 registers/thread` 可以直接推断实际 Occupancy 吗？答：不能，还需 GPU/SM 资源限制、Block 大小、分配粒度和其他资源约束。

### Q13. 你的 `nvcc -Xptxas=-v` 能看什么？

**参考回答**：可以查看每线程寄存器数量、Shared Memory、spill 等编译资源信息；`-arch=sm_75` 指向 RTX 2060 的架构。但编译参数变化可能改变资源结果；这不等于运行时 Profiling，不足以单独解释性能变化。

### Q14. Memory Coalescing（访存合并）是什么意思？

**参考回答**：同一 Warp 的线程执行某条 Global Memory 访存指令时，硬件根据访问地址把这些请求组合为内存事务。访问连续、合理对齐的数据通常更有利于合并。需要按 **Warp 内每一条指令的地址** 分析，而不能只看二维数组定义。

**追问**：为什么 `B[bRow*N+col]` 随 `threadIdx.x` 改变通常更连续？答：同一行里 col 连续变化，对 Row-major B 对应连续元素。

### Q15. GEMM 的 FLOPs 和 GFLOPS 如何算？

**参考回答**：约 `2*M*N*K` FLOPs（1 次乘法 + 1 次加法算 2 次浮点操作；FMA 按 2 FLOPs 计）。`GFLOPS = 2MNK / (latency_s * 1e9)`。时间单位是毫秒时可写作 `2MNK / (latency_ms * 1e6)`。

**计算题**：512³ 的操作量是 **268,435,456 FLOPs**。Register2 TILE32 延迟 0.265238 ms，约 **1012.055 GFLOPS**；cuBLAS 延迟 0.070168 ms，约 **3825.628 GFLOPS**（基于当前日志）。

### Q16. 512³ 的 1.89× 和 3.78× 具体指什么？

**参考回答**：

- 自定义优化 **1.89×**：`502.394 / 265.238 ≈ 1.89`，即 Naive 延迟除以 Reg2 TILE32 延迟。
- cuBLAS 相对最快自定义 **3.78×**：`265.238 / 70.168 ≈ 3.78`。

**最重要的追问**：可以说“我的 Kernel 比 cuBLAS 快 3.78 倍”吗？**不可以，方向相反**；这里 cuBLAS 更快。

## D. 测试方法、CUDA Graph、cuBLAS（Q17—Q21）

### Q17. 为什么要在同一 Benchmark 中测五种 Kernel？

**参考回答**：共享输入、CPU Reference、GPU 显存、Stream、Warmup、Event 和重复策略，降低不同程序/测试条件带来的偏差。仍需考虑测试顺序、功耗/温度、GPU 时钟和不同进程状态；同一程序不等于完全公平。

### Q18. CUDA Event 测的是纯 Kernel 执行时间吗？

**参考回答**：Event 记录 GPU 时间线上的两个点。本项目在相同 Stream 上 Record start、提交操作、Record stop，再同步并读取 elapsed time；这个时间包含它们之间的 GPU 工作以及可能的等待/调度间隙。**不能不加限定地称为完全纯净的 Kernel 执行时间**。不包括计时区间之外的 CPU Reference、H2D/D2H、malloc、Graph Instantiate。

### Q19. CUDA Graph Capture、Instantiate、Replay 的区别？

**参考回答**：`cudaStreamBeginCapture` 开始把提交到 Stream 的操作记录进图；`cudaStreamEndCapture` 得到 `cudaGraph_t`；`cudaGraphInstantiate` 生成 `cudaGraphExec_t`；`cudaGraphLaunch` 将其提交执行。捕获阶段 **不实际执行被捕获的那些操作**。图复用的主要收益是减少 Host 多次提交的开销，尤其在小操作上明显。

**代码解释**：64 个 GEMM/Graph × 16 次 Graph 回放 = **1024 次 GEMM/轮**，总 Event 毫秒数应除以 1024。

**追问**：Graph 能自动提升 Shared Memory 命中率、改变寄存器分块算法吗？答：**不能**，没有优化 Kernel 内部算法。

### Q20. 为什么 CUDA Graph 中要给 cuBLAS 绑定 Stream 和 Workspace？

**参考回答**：让 cuBLAS 操作与自定义 Kernel 在同一被捕获 Stream 中；预先配置 Workspace 可减少 cuBLAS 捕获过程中内部内存管理带来的复杂性。本项目先 `cublasSetStream(handle, stream)` 再 `cublasSetWorkspace`；后续若重新设置 Stream，应检查工作空间配置。

### Q21. cuBLAS 为什么把 `d_B` 放在 `d_A` 前面，并传 `N,M,K`？

**参考回答**：我们的矩阵以 Row-major 存放；cuBLAS 传统 SGEMM 接口按 Column-major 解释。基于 `(AB)^T = B^T A^T`，把 Row-major `B[K,N]` 内存视为 Column-major 的 `B^T[N,K]`，把 A 视为 `A^T[K,M]`，得到 `C^T[N,M]` 的 Column-major 内存，恰好等于需要的 `C[M,N]` Row-major 内存，不必进行真实转置或搬运。

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,  // B^T：列优先的 N × K，leading dimension = N
            d_A, K,  // A^T：列优先的 K × M，leading dimension = K
            &beta,
            d_C, N); // C^T：列优先的 N × M，leading dimension = N
```

**计算题**：A[2,3] × B[3,4]，cuBLAS 视角计算 Bᵀ[4,3] × Aᵀ[3,2] = Cᵀ[4,2]。`lda/ldb/ldc` 是对应列优先矩阵的 Leading Dimension，**不是元素总数**。

## E. 正确性、数值精度与项目真实性（Q22—Q25）

### Q22. 为什么使用固定种子的随机 FP32 输入？

**参考回答**：原先输入为周期性、二进制可精确表示的值，出现 `max_diff=0` 不代表一般输入都完全一致。随机输入覆盖更多非整齐数值，固定种子便于复现失败元素。**但仅一个种子、一个取值区间仍不能覆盖所有数值病态情况。**

### Q23. CPU Reference 为什么采用 FP64 累加，又为什么还会有 FP32 误差？

**参考回答**：输入 A/B 是 FP32，先转换成 double 再做乘加可显著降低参考结果的累加误差；最后结果存回 FP32，因此仍有最终转换舍入。参考值是较高精度基准，而不是完全无误差的数学证明。GPU 各实现可能使用 FMA、不同累加顺序；浮点加法不满足严格结合律。

**本项目踩坑**：我误将 Naive Kernel 改成了 `double`，512³ 从约 0.50 ms 变成约 3.96 ms；之后恢复为 `float`。**这是“数据类型会直接影响硬件路径与性能”的实测教训。**

### Q24. 你的容差怎么定？为什么 `max_diff > atol` 仍可能 PASS？

**参考回答**：本次采用 `abs(gpu-ref) <= 2e-5 + 1e-4*abs(ref)`，容差包含绝对项与相对项。对于幅值较大的 ref，允许误差大于固定 atol，因此 `max_diff` 可能超过 atol 但每个元素仍通过。当前阈值是针对 K≤512、随机 [-1,1] 的经验验收阈值，不能推广为万能阈值。更严格的工程验证可增加多种随机种子、极端输入以及依据点积误差界的容差设计。

**追问**：为什么接近 0 的输出尤其容易因为纯相对误差 FAIL？答：接近 0 时相对误差的分母很小，故需要合适的绝对容差。

### Q25. 如果面试官问“工具链都成熟了，你实际做了什么”，该怎么回答？

**参考回答（建议口头控制在 60–90 秒）**：

> 我在 RTX 2060 上实现了可运行的 CUDA FP32 GEMM，从最初的 Naive 一线程一输出开始，增加 Shared Memory Tiling，又进一步做了每线程两列输出的 Register Blocking，并比较了 TILE16 / TILE32 在不同形状下的效果。我独立搭建了 CPU Reference、数值正确性验证和 CUDA Event / CUDA Graph Benchmark；还解决过 Row-major 与 cuBLAS Column-major 接口适配的问题。
>
> 在 512³、随机 FP32、CUDA Graph 的测试中，最快自定义版本相比 Naive 加速约 1.89×，吞吐量约 1.01 TFLOPS；cuBLAS 仍比它快约 3.78×。我没有把 cuBLAS 的内部算法当作自己实现的功能，也不会把一次小矩阵的低延迟称为普遍领先。下一步如果继续优化，我会先用 Nsight 收集访存、寄存器和 Warp 执行证据，再决定做更大的寄存器计算块或流水化。

**连续追问**：① 哪些 Kernel 是你自己写并调过的？② 哪些指标是实测，哪些是推测？③ 如何证明你的加速不是因为把 FP32 改成 FP64 / 换了计时方式？

---

# 第二部分：真实代码解读（必须能脱稿解释）

## 代码题 A：Global Index

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= M * N) return;
int row = idx / N;
int col = idx % N;
```

你必须当场说清：全局索引、边界、行列转换，以及为什么不能误写 `row = idx / K`。

## 代码题 B：Shared Memory 两次同步

```cpp
As[ty][tx] = (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;
Bs[ty][tx] = (bRow < K && col < N) ? B[bRow*N + col] : 0.0f;
__syncthreads();
for (int k = 0; k < TILE; ++k)
    sum += As[ty][k] * Bs[k][tx];
__syncthreads();
```

你必须能指出两个同步点分别防止什么读写竞态，以及为什么越界补零。

## 代码题 C：一个 A 复用两次

```cpp
float a = As[ty][k];
sum0 += a * Bs[k][tx];
sum1 += a * Bs[k][tx + TILE];
```

你必须能解释两个累加器分别对应 `col0` / `col1`，以及两列共享相同 A 数据的原因。

## 代码题 D：cuBLAS 参数视角

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
```

你必须写出 `(AB)^T = B^T A^T`，并说明 `N/M/K` 和 Leading Dimension 的含义。

## 代码题 E：CUDA Graph 与 Event

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < 64; ++i) launch();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

cudaEventRecord(start, stream);
for (int i = 0; i < 16; ++i)
    cudaGraphLaunch(graph_exec, stream);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
```

你必须说明 Capture 与 Replay 的区别，并指出平均 GEMM 时间的除数 **1024**；同时说清楚 Event 测量范围的局限。

---

# 第三部分：三轮模拟面试（先独立完成）

## 第一轮：基础（建议 15 分钟）

1. [源码] 解释 `gemm_naive()` 中线程映射和 A/B/C 索引。
2. [计算] `M=130,K=65,N=70`，Tiled16 Grid 需要 `(ceil(70/16),ceil(130/16))=(5,9)`，共有 **45 Blocks**；K 方向需要 **5 次 Tile**。
3. [连续追问] Shared Memory 为什么有两次同步？为什么补零？能否删掉第二次同步？

**通过标准**：能画出 Row-major 访问、正确给出 Block 和 Tile 数、准确说明同步范围。

## 第二轮：优化（建议 15 分钟）

1. [源码] 解释 Register2 `sum0/sum1` 的数据复用。
2. [计算] Register2 TILE32：**1024 threads/Block**、**12 KiB 静态 Shared Memory/Block**、理论输出 Tile **32×64**。
3. [连续追问] 大 TILE 何时有利？寄存器增多为什么可能有害？Occupancy 为什么不是越高越好？

**通过标准**：说出收益和代价，不能用“因为 GPU 并行度高”敷衍所有问题。

## 第三轮：工程与性能（建议 20 分钟）

1. [源码] 解释 cuBLAS 的 A/B 交换和 CUDA Graph Capture → Instantiate → Replay。
2. [计算] 512³ 有多少 FLOPs？由 **502.394 μs → 265.238 μs** 得到多少加速比？cuBLAS 的 **70.168 μs** 代表谁比谁快？
3. [连续追问] 为什么要固定随机种子？为什么 FP64 Reference 与 FP32 Kernel 会不完全相同？为什么不能宣布“小矩阵我的 CUDA 一定比 cuBLAS 快”？

**通过标准**：区分真实测量与假设；准确解释误差、计时口径和实验局限。

### 自测打分（每轮 100 分）

| 维度 | 分数 | 评分依据 |
|---|---:|---|
| 概念与公式准确 | 30 | Shape、索引、FLOPs、线程 / Shared Memory 的计算无错 |
| 能解释关键源码 | 30 | 能逐行说明输入/输出/线程映射/资源与同步 |
| 工程证据与边界 | 25 | 引用真实日志，不杜撰内部 cuBLAS 实现或因果结论 |
| 连续追问与表达 | 15 | 观点完整，能够回答“为什么/代价/怎么验证” |

建议各轮达到 **≥80 分** 且没有关键概念错误，再视为通过。

---

# 第四部分：面试容易踩的坑（自查清单）

- [ ] 没有把 `nvidia-smi` 的 CUDA 13.1 当作 nvcc 版本；项目用 CUDA Toolkit 12.8.93。
- [ ] 没有说“Shared Memory Tiling 把所有 Global Memory 访问变成 0”。
- [ ] 没有说“Occupancy=100% 所以性能一定最佳”。
- [ ] 没有混淆理论 FLOPs / GFLOPS 与 CUDA Event 实测延迟。
- [ ] 没有说“CUDA Graph 会自动优化 GEMM 的访存和寄存器用量”。
- [ ] 没有把 cuBLAS 的库实现算作自己写的 CUDA Kernel。
- [ ] 没有把输出 `PASS` 当成任意 Shape、任意数据都已证明正确。
- [ ] 说清 FP64 Reference 是“FP64 累加后写回 FP32”，并说明浮点容差来源。
- [ ] 不再使用被误改成 FP64 的 Naive 性能数据（3.959 ms）。
- [ ] 对 64³ / 130×65×70 的微秒级差异说明复测和环境波动。
- [ ] 说清 512³：**1.89× 是自定义优化，3.78× 是 cuBLAS 相比最快自定义的领先**。

## 第五部分：还没有做过的事情（面试不得冒充完成）

目前的项目记录**不能支持**以下说法：已用 Nsight Compute 定位 Memory Throughput / Stall Breakdown；已实现 WMMA/Tensor Core GEMM；已完成 Tensor Core 与普通 CUDA Core 的同精度公平对比；已经实现流水线 `cp.async`；已实现高阶、多尺寸 Auto-tuning 调度器；或已测得 CUDA Graph 的严格纯 GPU Kernel 执行时间（与所有调度间隔完全隔离）。

后续如果扩展 V3，可把 Nsight Compute、Warp 内存事务、寄存器溢出、指令效率、较大每线程寄存器 Tile、性能稳定性控制等作为下一阶段任务，**不要倒写成 V2 已完成的工作**。

---

## 参考资料与可核查来源

- 项目源文件：`gemm_bench_unified.cu`、`gemm_tiled.cu`、`gemm_register_2.cu`、`gemm_register_2_tile32.cu`、`gemm_compare.cu`；对应本次会话中的用户实测日志。
- 项目性能记录：`docs/PERFORMANCE.md`、`docs/OPTIMIZATION.md`。
- NVIDIA CUDA Programming Guide · CUDA Graphs：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA CUDA C++ Programming Guide · Global Memory / Occupancy：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA cuBLAS Documentation · Column-major / GEMM：https://docs.nvidia.com/cuda/cublas/index.html
