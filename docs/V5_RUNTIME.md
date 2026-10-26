# MiniGPU V5 — 轻量级 GPU Runtime：资源管理、执行顺序与 Buffer 复用

> **版本范围**：V5.1–V5.3；预备发布 `v5.0`，**GitHub 标签须以实际推送结果为准**。  
> **记录日期**：2026-10-26。  
> **定位**：MiniGPU V1–V6 的第五阶段；用于源码复习、工程交接和 AI Systems / GPU Runtime 面试。  
> **真实性说明**：数值直接取自用户在 RTX 2060/WSL2 实机运行的终端日志；本文没有把未做过的 Profiling、显存泄漏检测或速度提升归入已完成工作。

## 0. 三十秒回顾

- **V5.1**：`runtime/device_buffer.h` 实现最小 `DeviceBuffer<T>` RAII；以已有 V4 Softmax 实际运算验证资源封装，`PASS`，最大绝对误差 `2.7299506e-09`。
- **V5.2**：`runtime_attention_v5.cu` 复用 V4 Attention；一次分配六个 Device Buffer、三轮不同输入；一个非阻塞 CUDA Stream 负责 `H2D → Attention → D2H`；CUDA Event 记录完成与区间计时，`3/3 PASS`，旧 V4 Attention 回归 `ALL PASS`。
- **V5.3**：相同 `S=64,D=64` FP32 Fused Attention，对照每轮新建六个 Device Buffer 和在循环外复用六个 Buffer；每轮做 CPU/GPU 正确性比较。5 次预热、30 对交替测量，CPU `std::chrono::steady_clock` 计包含资源操作的墙钟耗时。新建方案中位数 **175.673 μs**，复用方案 **159.477 μs**，比值 **1.102×**；两路径正确性 `ALL PASS`。
- **项目边界**：V5 是功能明确的最小执行管理实验；**不是**通用 Tensor 框架、内存池、调度器或生产级推理引擎。未实现多 Stream 调度、CUDA Graph Runtime、复杂 allocator 或任意 Shape 缓存。

## 1. 为什么在 V4 后需要 V5

V4 关注单独的 CUDA Kernel 与 Attention 计算正确性：`QKᵀ → (Scale+)Softmax → PV`。其常规调用代码直接使用 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 和同步。V5 的新问题是：**谁拥有 GPU 资源、何时能够复用资源、任务如何有序提交与完成、如何比较管理开销**。

本版只做必要的工程升级，保留 V4 Kernel 数学实现和已有测试。这样 V5 数据流是对同一计算工作负载进行资源/执行管理，避免重复写算子导致对照混杂。

## 2. 环境、代码和构建入口

| 项目 | 已知配置 |
|---|---|
| GPU | RTX 2060 Laptop，6 GB，Turing / SM 7.5 |
| OS | WSL2 Ubuntu 24.04.5 LTS |
| Windows 显卡驱动 | 591.74（先前实测环境配置，V5 日志未重新查询驱动） |
| NVCC | CUDA Toolkit 12.8.93 |
| C++ / 构建 | GCC 13.3、C++17 / CUDA17、CMake |
| 工作目录 | `/home/damon/projects/minigpu` |
| 构建目录 | `build-v5` |
| 测试任务 | `runtime_v5`、`runtime_attention_v5`、`attention_v4` 回归 |

本阶段文件及用途：

```text
minigpu/
├── runtime/device_buffer.h        # V5.1 DeviceBuffer<T> RAII
├── runtime_v5.cu                  # V5.1 DeviceBuffer + V4 Softmax
├── runtime_attention_v5.cu        # V5.2/5.3：Attention 执行与复用 Benchmark
├── attention_v4.cu                # V4 Attention：新增可选 Stream 参数及禁用测试 main 的宏
├── softmax_v4.cu                  # V4 Softmax：直接复用
├── CMakeLists.txt                 # 新增两个 V5 target
└── docs/V5_RUNTIME.md            # 当前报告
```

构建与回归命令：

```bash
cd ~/projects/minigpu
cmake -S . -B build-v5 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build-v5 --target runtime_v5 runtime_attention_v5 attention_v4 --parallel 4
./build-v5/runtime_v5
./build-v5/runtime_attention_v5 | tee /tmp/minigpu_v5_benchmark.log
./build-v5/attention_v4 | tail -1
```

**防止假验收**：将 `cmake --build` 和实际运行用 `&&` 连接，能避免编译失败时误运行旧版可执行文件；已贴出的日志包含成功编译和执行记录。想要完整保存错误，应使用 `2>&1 | tee ...` 并启用 `set -o pipefail`。

## 3. V5.1：DeviceBuffer<T> 与 RAII

### 3.1 本节真实需求

V4 代码中，原始指针由调用者显式申请与释放：

```cpp
float* d_x = nullptr;
cudaMalloc(reinterpret_cast<void**>(&d_x), bytes);
// ... GPU 操作 ...
cudaFree(d_x);
```

只要存在早退、C++ 异常或调用层级变化，释放逻辑就容易漏写。`DeviceBuffer<T>` 通过对象生命周期为资源建立单一所有权。

```cpp
DeviceBuffer<float> d_x(count);
softmax_kernel<<<M, 256>>>(d_x.data(), d_y.data(), N);
// 对象离开作用域时自动调用析构函数释放显存
```

### 3.2 必须掌握的源码

```cpp
template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count);
    ~DeviceBuffer() noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() noexcept;
    std::size_t bytes() const noexcept;
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};
```

要点：

- 构造时分配 `count*sizeof(T)` 字节 Device Memory；失败通过 C++ 异常上报。
- 析构时 `cudaFree`，不向外抛出异常；正式项目应考虑如何诊断释放错误。
- 禁止默认复制以防同一 GPU 指针被两个对象分别释放（Double Free）。
- 当前版本没有实现移动构造/赋值，因此**不要把它当成可随意在容器间转移所有权的通用 Buffer**。
- `DeviceBuffer` 的作用是**资源安全**而不是**自动性能加速**：每次构造仍调用 `cudaMalloc`。
- 异步任务仍在访问 Device Buffer 时，不可让对象提前析构；必须保证异步操作与 Buffer 生命周期匹配。

### 3.3 实机验收

```text
[100%] Built target runtime_v5
V5.1 DeviceBuffer + Softmax: PASS
max_diff=2.7299506e-09
```

这证明封装下的 GPU 运算及该输入的参考结果正确；**不等于已经做过独立的显存泄漏检测**。

## 4. V5.2：Buffer 复用、Stream、Event

### 4.1 复用工作区

每次 Attention `S=64,D=64` 需要：

- `Q`, `K`, `V`, `Output` 各为 `[64,64]`，每块 4,096 个 FP32 = 16 KiB。
- `Scores`, `Weights` 各为 `[64,64]`，每块同样 16 KiB。
- 六块 Device Buffer 总计 **96 KiB**（不含 cuBLAS 内部空间、Pinned Host Buffer、驱动开销）。
- 一次推理 H2D 传 Q/K/V 三块，D2H 传 Output 一块：**逻辑拷贝量 64 KiB**，并不代表总显存/缓存访问量。

实际三轮推理把 Buffer 申请放在循环外，对不同的 Q/K/V 输入反复执行相同链路，输出分别与 CPU Reference 对齐。

### 4.2 为什么必须同时设置 cuBLAS 与 Kernel 的 Stream

`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` 创建非阻塞 Stream。需完成两件事：

```cpp
cublasSetStream(blas.value, stream.value);
scaled_softmax_kernel<<<S, THREADS, 0, stream.value>>>(...);
```

这样 GEMM 和自定义 Kernel 才属于**同一条 GPU 任务序列**。提交顺序为：

```text
cudaMemcpyAsync(Q/K/V H2D)
  → cuBLAS QKᵀ
  → fused Scale+Softmax Kernel
  → cuBLAS PV
  → cudaMemcpyAsync(Output D2H)
  → cudaEventRecord(done)
```

同一 CUDA Stream 内按序执行，因此**不需要在每一步后使用 `cudaDeviceSynchronize()`**。最后使用 `cudaEventSynchronize(done)` 等待结果就绪。

### 4.3 异步拷贝与 Pinned Memory

CPU 输入输出采用 `cudaMallocHost` 建立 Pinned Buffer，为 `cudaMemcpyAsync` 提供页锁定 Host 内存。它有助于实现可靠的异步数据传输；**不等于本阶段已证明 H2D 与 Kernel 计算重叠**：这里只使用单个 Stream，传输、计算依序执行。

Host 输入只有在关联拷贝不再访问它以后才能覆盖；输出只有在 D2H 完成后才能被 CPU 正确读取。Event 等待还保护设备资源在 GPU 任务仍访问它们时不被提前释放。

### 4.4 实机结果

最终 V5.3 运行前置的 V5.2 验证日志：

```text
GPU buffers allocated once; S=64 D=64, rounds=3
round=1 PASS max_diff=1.0244548e-08 stream_elapsed=15.2960 ms
round=2 PASS max_diff=1.3038516e-08 stream_elapsed=0.2084 ms
round=3 PASS max_diff=1.1408702e-08 stream_elapsed=0.2196 ms
V5.2 Attention Runtime: ALL PASS
```

先前独立 V5.2 运行的时间分别为 `31.6325 ms`、`0.3351 ms`、`0.3435 ms`，同样 `ALL PASS`，旧 V4 回归亦通过：

```text
V4.4 Attention Fusion: ALL PASS
```

注意：第一次耗时显著高于后两次，可能涉及首次执行开销与提交间隔，但缺少独立 Profiling，**不能认定成某一个已验证的根因**。这里的 Event 时间包含 H2D、计算、D2H 所在 Stream 时间区间，与 V4 Attention 的纯计算链路计时口径不一致，不要直接做倍数比较。

## 5. V5.3：Fresh Allocation 与 Buffer Reuse

### 5.1 代码结构与公平性

当前 `runtime_attention_v5.cu` 实际包含：

```cpp
struct AttentionWorkspace {
    DeviceBuffer<float> q, k, v, scores, weights, out;
    AttentionWorkspace(std::size_t data_count, std::size_t score_count)
        : q(data_count), k(data_count), v(data_count),
          scores(score_count), weights(score_count), out(data_count) {}
};
```

统一的 `execute_attention(...)` 使用一条 Stream 做三次 H2D、一条 Fused Attention Forward、一次 D2H，末尾 `cudaStreamSynchronize`。Benchmark 只替换 Workspace 生命周期：

```cpp
auto begin = std::chrono::steady_clock::now();
if (fresh) {
    AttentionWorkspace temp(data_count, score_count); // 包含六次分配
    execute_attention(temp, ...);
} else {
    execute_attention(reused, ...);                  // 不重复分配
}
auto end = std::chrono::steady_clock::now();       // fresh 包含临时对象析构/释放
```

两路径固定为相同 FP32 输入、相同 Shape、相同 Fused Attention 算法、相同 Stream/cuBLAS Handle、相同 H2D/D2H 与同步。每次完成后验证输出与 CPU 参考值一致。

**测量口径**：CPU `steady_clock` 的**单轮墙钟耗时**，包括 CPU 任务提交、Host↔Device 拷贝以及等待。Fresh 包括 `cudaMalloc/cudaFree`，Reuse 不包含预先分配的 Workspace；cuBLAS Handle、Stream/Event 和 Pinned Host 初始化不计入该循环。它不是 GPU Kernel Duration，也不是带工作区初始化的一次性完整端到端延迟。

### 5.2 测试设置

| 项目 | 设置 |
|---|---|
| GPU | RTX 2060 Laptop（Turing SM75） |
| 任务 | 单 Head Fused Attention, `S=64,D=64` |
| 精度 | FP32 |
| Warmup | 5 次，两路径分别预热 |
| 正式测试 | 30 对（每路径 30 次），配对交替顺序 |
| 计时 | CPU `std::chrono::steady_clock`，单轮 μs |
| 汇总 | 各路径独立中位数、最小、最大；加速比=Fresh 中位数 / Reuse 中位数 |
| 正确性 | 全部输入与 FP64 中间计算的 CPU Reference 按 `2e-5 + 1e-4 * abs(ref)` 判定 |

### 5.3 用户实测原始输出

```text
V5.3 benchmark: S=64 D=64 FP32, warmup=5, pairs=30
fresh_alloc_each: median=175.673 us (min=161.428, max=194.031)
reuse_buffers:    median=159.477 us (min=137.795, max=237.606)
speedup (ratio of medians)=1.102x; max_diff=1.1408702e-08
V5.3 Buffer Reuse: ALL PASS
```

| 指标 | Fresh | Reuse |
|---|---:|---:|
| Median (μs) | 175.673 | 159.477 |
| Min (μs) | 161.428 | 137.795 |
| Max (μs) | 194.031 | 237.606 |

- **本次中位数之比**：`175.673 / 159.477 ≈ 1.102×`。
- **单轮中位数差**：`16.196 μs`，相对 Fresh 中位数的时间减少约 **9.2%**。
- **波动与限制**：两条路径的 min/max 范围重叠，Reuse 曾出现 237.606 μs 的慢轮次；只有这一轮 30 对测试，没有跨进程的统计置信区间。**不应宣称 Buffer Reuse 对所有运行都稳定加速 1.102 倍**。
- **因果边界**：两种路径的主要被控变量是 Buffer 生命周期；结果与减少重复分配/释放开销一致，但 `16.196 μs` 不能被当作六次 `cudaMalloc/cudaFree` 的精确独立耗时（包含测量波动、系统调度及其他 Runtime 作用）。
- **不等于 Kernel 加速**：两路径使用完全相同的 Fused Attention Kernel；改进来自执行管理，不是算子的计算量或指令减少。

## 6. Runtime 的所有权和同步不变量

1. `DeviceBuffer` 对 GPU 指针拥有唯一所有权，禁止复制；所有权不会静默共享。
2. `AttentionWorkspace` 为一次 Attention 提供六块工作显存，Fresh 每轮析构，Reuse 跨轮存活。
3. `cublasSetStream` 和 `<<<...,stream>>>` 把 cuBLAS 与自定义 Kernel 放在同一任务队列，避免跨 Stream 缺少显式依赖。
4. H2D 源 Pinned Buffer 不会在拷贝未完成时被 CPU 覆盖；D2H 目标在完成事件或 Stream 同步后才读。
5. 异步 GPU 任务完成前不允许销毁其访问的 Device/Host 内存；本实现 `CudaStream` 析构时还做同步保护。
6. 目前 RAII 析构函数对资源释放错误不抛异常；**尚未提供独立的泄漏检测器或生产级错误恢复策略**。

## 7. 必考源码、计算与面试题

### 7.1 源码理解

**题**：`DeviceBuffer(const DeviceBuffer&) = delete` 为什么重要？  
**答题框架**：GPU 指针代表独占拥有的资源；默认浅复制会让两个析构函数重复 `cudaFree`。当前类不能直接复制；如需转移所有权，可以另行设计移动构造/赋值并清空源指针，V5 未实现。

**题**：为什么 `cublasSetStream(handle, stream)` 和 Kernel 的 `<<<...,stream>>>` 都必需？  
**答题框架**：Stream 创建只是获得执行队列句柄，不会自动更改已存在的 GPU 调用。计算必须显式进入同一 Stream，才能依赖它的有序执行保证 `QKᵀ→Softmax→PV`。

### 7.2 计算题

**题**：一次 `S=64,D=64` FP32 单 Head Attention 的六块 Buffer 总容量？  
**解**：Q/K/V/Output 四块 `[64,64]`，Scores/Weights 两块 `[64,64]`，每块 `64×64×4=16,384 B=16 KiB`，总共 `6×16 KiB=96 KiB`。不含库内部临时显存。

**题**：假设 Fresh=300 μs，Reuse=200 μs，加速比及耗时降低比例是多少？  
**解**：`300/200=1.5×`；延迟下降 `(300−200)/300=33.3%`；两个百分数口径不同。

### 7.3 工程诊断题

**题**：为什么第一轮 Event 时间数十毫秒，后两轮只有约 0.2 ms？  
**答题框架**：先区分首次延迟初始化、CPU 提交间隙、GPU 时钟/系统波动等候选因素；Event 两点之间可以包含无 GPU 执行的间隔；现有日志不足以锁定根因。如果未来真要深挖，可用 Nsight Systems 采时间线，但**V5 不必追加**。

**题**：为什么 Buffer Reuse 快了也不能说明 Kernel 更快？  
**答题框架**：同一个 Attention Kernel，差别在是否每轮构造/析构工作区；所减少的是重复资源管理开销，计时范围也覆盖 CPU 提交和传输。

### 7.4 三道递进追问

1. 设计一个只允许独占拥有的 Device Buffer 时，构造、析构、禁止复制分别承担什么责任？
2. 如果 H2D 放在 Stream A、Attention 放在 Stream B，只用 `cudaStreamSynchronize(B)` 能否保证 A 的数据已经准备好？为什么需要跨 Stream Event 依赖？
3. 如果频繁使用不同 Shape，固定容量 Buffer 不再够用，你会如何设计容量检查/按需增长和可复用区？哪些操作必须等正在执行的 GPU 任务完成才能安全修改？

## 8. 发布检查与版本边界

核心验收已实测：

- [x] V5.1 CUDA 编译 + RAII Softmax 正确性；
- [x] V5.2 三轮复用、Stream/Event、Pinned H2D/D2H + Attention 正确性；
- [x] V4 Attention 回归 `ALL PASS`；
- [x] V5.3 两种 Buffer 生命周期，30 对 CPU 墙钟计时 + 正确性；
- [x] 原始延迟、误差、测量口径与波动已记录；
- [ ] 本文件复制到仓库 `docs/V5_RUNTIME.md`；
- [ ] README / `PROJECT_STATUS.md` 状态同步；
- [ ] Git Commit、annotated tag `v5.0`、push 实际成功确认。

**不夸大**：当前没有单独证明通用内存复用池、任意不同 Shape、GPU 并发流水、多 GPU、CUDA Graph 加速、生产级调度/异常恢复，未做独立 CUDA memory leak 检测。V6 必须基于真实模型和实际 Profiling，再选择可证明有价值的优化点。

## 9. 下一阶段 V6 交接

- 核心工程：现有 `V1/V2 GEMM`、`V4 Softmax/RMSNorm/Attention`、`V5 DeviceBuffer/Stream/Event/Reuse`。
- 目标：选择适合 RTX 2060 6 GB 的真实小型 Transformer / PyTorch 模型，记录基线、确认 Profiling 瓶颈、替换或接入一个自定义算子、验证数值与端到端收益。
- 技术取舍：V5 不是为了搭建大而全执行器；V6 如要用 PyTorch CUDA Extension，应以真实模型能否运行、能否验证具体性能增量为准。
- 最小下一任务：先选一个模型和明确输入 Shape，建立不含自定义优化的原始 PyTorch 推理基线；**不要预设 MiniGPU 的自定义 Kernel 必然胜过成熟 PyTorch/cuBLAS 实现**。
