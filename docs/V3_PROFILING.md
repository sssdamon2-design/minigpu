# MiniGPU V3 — GPU Profiling、瓶颈定位与优化实验完整记录

> **用途**：长期自我复习、技术面试准备、向其他 AI 交接项目上下文。  
> **记录时间**：2026-10-25。**数据属性**：以下数字来自开发者在 RTX 2060 + WSL2 实机上运行后的日志，不是推算或模型虚构。  
> **当前状态（最后更新）**：V3 核心分析、一次优化尝试、回退与 256³ 回归均完成；`docs/V3_PROFILING.md` 已暂存，**尚未创建/推送 `v3.0`**。此前列出的全部 5 组 Shape 的恢复后回归、阶段面试验收也尚未在本对话中确认完成。

## 0. 先看结论（30 秒回顾）

- **V3 的真正目标**：从 V2 的“哪个 GEMM 快”，进阶到“用性能计数器提出可证伪的性能瓶颈假设，修改代码并验证”。不是必须提高某个 GFLOPS 数字。
- **分析对象**：`gemm_bench_unified.cu` 的 FP32 `gemm_register2_kernel<16>` 与 `<32>`，主要 Shape 为 `M=K=N=256`，补充 `512³`。
- **现象**：256³ 下，TILE16 约 **39–40 μs**；TILE32 约 **50–51 μs**。而 512³ 下原版 TILE32 略快于 TILE16；最佳 TILE Size 随 Shape 改变。
- **实测证据**：256³ 的 TILE32 `MIO Throttle=24.06` vs TILE16 `14.73`，`Long Scoreboard=17.76` vs `2.62`（Warp State 统计的归一化指标，**不是耗时百分比**）。DRAM Throughput 两者均约 1–2%，未见 DRAM 带宽饱和证据。
- **唯一 V3 代码实验**：仅对 `TILE=32` 的 K 循环尝试 `#pragma unroll 4`。256³ 从 **50.582 → 52.729 μs**；512³ 从 **265.928 → 274.702 μs**。正确性 PASS，性能未改善，**决定撤销此修改**。
- **恢复已验证**：`git restore --source=v2.0 -- gemm_bench_unified.cu` 后，256³ 五个实现全部 PASS；TILE16 **39.216 μs**、TILE32 **50.570 μs**。
- **认识边界**：已获得访存相关瓶颈线索，**未证明** Shared Memory Bank Conflict，也**未通过前后 SASS/计数器验证**展开使性能下降的确切原因。

---

## 1. 工程定位、环境、版本状态

### 1.1 六阶段主线

| 版本 | 内容 | 本次确认状态 |
|---|---|---|
| V1 | 基础 CUDA Kernel 与双层 MLP 前向闭环 | 已完成基础验收，Tag `v1.0` |
| V2 | GEMM：Naive → Shared Memory Tiling → Register Blocking → cuBLAS/CUDA Graph | 已提交并推送，Tag `v2.0` |
| **V3** | **GPU Profiling → 瓶颈假设 → 单项修改 → 验证/回退** | **核心实验完成；版本发布未完成** |
| V4 | Reduction / Softmax / RMSNorm / 基础 Attention / Fusion | **未实现** |
| V5 | 小型 GPU Runtime（Buffer RAII、Stream/Event、复用） | **未实现** |
| V6 | 真实小型 Transformer 模型 Profiling 与算子集成优化 | **未实现** |

V3 不另建复杂框架，不重新实现 GEMM，不追求将每个 Nsight 面板都学一遍。

### 1.2 硬件与软件

| 项目 | 实测或已确认配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 Laptop，6 GiB，Turing / SM75，30 SM |
| Windows NVIDIA Driver | 591.74 |
| WSL | Ubuntu 24.04.5 LTS，WSL2 |
| CUDA Toolkit / NVCC | 12.8.93，`/usr/local/cuda-12.8/bin/nvcc` |
| Nsight Systems | `nsys` 2024.6.2，位于 `/usr/local/bin/nsys` |
| Nsight Compute | `ncu` 2025.1.1；`/usr/local/cuda-12.8/bin/ncu` 或 `/opt/nvidia/nsight-compute/2025.1.1/ncu` |
| 工程目录 | `/home/damon/projects/minigpu` |
| 远程仓库 | https://github.com/sssdamon2-design/minigpu |
| V2 Commit | `fb9a54f`，已推送到 `origin/main`，Tag `v2.0` 指向该提交 |

注意：`nvidia-smi` 的 `CUDA Version: 13.1` 表示驱动支持的 CUDA 版本，不是实际 Toolkit 版本。WSL `nvidia-smi` 通常位于 `/usr/lib/wsl/lib/nvidia-smi`。

#### WSL 常见 PATH 问题（已经解决）

新终端曾出现 `nvcc: command not found` 和 `ncu not found`，但执行文件实际存在。处理方式：

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version
ncu --version
nsys --version
```

已经通过向 `~/.bashrc` 添加 `export PATH=/usr/local/cuda-12.8/bin:$PATH` 持久化（仍以本机实际配置为准）。**不要因 PATH 问题重新安装整套 CUDA 或 WSL Linux 显卡驱动。**

---

## 2. 要分析的真实源码与数据流

### 2.1 GEMM 和输入

计算 `C[M,N] = A[M,K] × B[K,N]`，Row-major FP32。当前验证使用 `std::mt19937(2026)`，随机数据范围 `[-1,1]`；CPU 先用 FP64 乘加，再转成 FP32 参考结果。

逐元素验收公式：

```text
abs(GPU - CPU_reference) <= 2e-5 + 1e-4 * abs(CPU_reference)
```

这仅是已测试的输入范围和 K≤512 的容差，不能推广到任意规模/精度。V2 的 5 组矩阵 × 5 种实现已实测 **25/25 PASS**。

### 2.2 Register Blocking 的核心 Kernel

文件：`gemm_bench_unified.cu`。**下列是重点代码节选，不是完整可编译文件**：

```cpp
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
    float sum0 = 0.0f, sum1 = 0.0f;

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
    if (row < M && col0 < N) C[row * N + col0] = sum0;
    if (row < M && col1 < N) C[row * N + col1] = sum1;
}
```

- 每线程计算同一行、相隔一个 TILE 的 **两个输出元素**，复用一个 `a` 值；两个累加器为 `sum0/sum1`。
- 每轮 K Tile，从全局内存加载 `As`、`Bs` 到 Shared Memory；两次 `__syncthreads()` 分别保障读前完成写入、下一轮覆盖前完成消费。
- `row<M`、`col<N`、`k<K` 的掩码处理非整除边界；越界输入填 0。
- TILE16 的 Block 为 `(16,16)`，每块 256 线程、8 Warps；TILE32 是 `(32,32)`，每块 1024 线程、32 Warps。
- 每 Block 静态 Shared Memory：`(TILE² + 2×TILE²) × 4 = 12×TILE² bytes`，分别 **3072 bytes（3 KiB）**、**12288 bytes（12 KiB）**；NCU 用十进制 KB 显示为约 3.07 / 12.29 Kbyte。
- 对于 `M=N=256`，TILE16 的 Grid 为 `(8,16)` = 128 Blocks；TILE32 为 `(4,8)` = 32 Blocks。

**有用的 C++ / CUDA 面试点**：模板参数 `TILE` 为编译期常量；`__shared__` 属于线程块共享存储；`__syncthreads()` 是块内同步；Row-major 下索引是 `row*leading_dimension+col`；`__global__` 函数由 CPU 发起，GPU 执行。

---

## 3. V3.1 — Nsight Systems：验证 CPU/GPU 提交路径

### 3.1 采集命令（已运行）

```bash
cd ~/projects/minigpu
nsys profile \
  --trace=cuda \
  --cuda-graph-trace=graph \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=/tmp/minigpu_v3_nsys_256 \
  ./build-v2/gemm_bench_unified 256 256 256

nsys stats --report cuda_api_sum \
  /tmp/minigpu_v3_nsys_256.nsys-rep
```

`--trace=cuda` 记录 CUDA 活动，`--cuda-graph-trace=graph` 以 Graph 为单位追踪，减少逐节点采集开销；`--sample=none` 等排除当前不需要的 CPU 采样。

### 3.2 核心实测结果

| API | 次数或现象 | 解释 |
|---|---|---|
| `cudaGraphLaunch` | **415** 次 | `5 种实现 × (3 次 Graph 预热 + 5 trials × 16 replays)`，符合代码配置 |
| `cudaEventSynchronize` | 25 次，API 时间占 78.7% | CPU 等待 Event 所依赖 GPU 任务完成；**不表示 GPU 78.7% 空闲** |
| `cudaMalloc` | 7 次；最长一次约 210 ms | 可能含初始化等成本；纯 Graph Benchmark 计时区间不包含这些步骤 |
| `cuda_gpu_sum` | `SKIPPED` | Graph 级跟踪模式下，本次 SQLite 没有单独 Kernel/Memory 节点数据 |

**重要区分**：CUDA API 统计的是 CPU 侧 API 调用耗时；CUDA Event 的 Graph 回放计时统计 GPU 时间线上的批量执行，二者不能直接等价。Graph Benchmark 不含 Host-Device 传输、初始化、Graph 捕获/实例化，也不是完整端到端推理耗时。

---

## 4. V3.2 — Nsight Compute：指标及问题排查

### 4.1 失败过程：Kernel 过滤与权限

第一次使用 `regex:gemm_register2_kernel<16` 未匹配模板名称；实际 demangled 名称含 `gemm_register2_kernel<(int)16>`。改成下面的部分匹配正则：

```text
--kernel-name-base demangled
--kernel-name 'regex:gemm_register2_kernel.*16'
```

随后出现 `ERR_NVGPUCTRPERM`：WSL 下的 GPU 性能计数器由 Windows NVIDIA 驱动控制，需要在 Windows 主机上开启 GPU Performance Counters 访问权限。调整后采集成功。**不应为这个错误重装 CUDA/显卡驱动。**

### 4.2 另一个采集陷阱：Profiler 锁频

早期 `ncu` 报告 TILE16 的 SM Frequency 仅 **156.56 MHz**，NCU Duration 达 **460.03 μs**，明显偏离未启用 Profiler 的正常延迟。关闭 Profiler 对时钟和缓存的主动控制后，观察到约 **1.88 / 1.89 GHz**。Profiler 仍可多次重放和扰动执行；正式加速比仍使用独立、无 Profiler 的 Benchmark。

### 4.3 最终资源指标采集命令（已运行）

```bash
cd ~/projects/minigpu
for tile in 16 32; do
  ncu \
    --set basic \
    --graph-profiling node \
    --clock-control none --cache-control none \
    --kernel-name-base demangled \
    --kernel-name "regex:gemm_register2_kernel.*${tile}" \
    --launch-count 1 \
    --page details --print-details all \
    ./build-v2/gemm_bench_unified 256 256 256 \
    > "/tmp/minigpu_v3_tile${tile}.log" 2>&1
done
```

用于 **只采集一个匹配 Kernel** 的基本指标；`--graph-profiling node` 对 CUDA Graph 的节点级执行进行指标采集，避免只得到 Graph 整体；`--clock-control none` 禁止 Profiler 主动锁频，`--cache-control none` 不主动清缓存。

### 4.4 资源指标（256³，真实日志）

| 指标 | TILE16 | TILE32 |
|---|---:|---:|
| SM Frequency | 1.88 GHz | 1.89 GHz |
| **NCU Duration** | **40.19 μs** | **50.98 μs** |
| Threads / Block | 256 | 1024 |
| Grid Size（总 Blocks） | 128 | 32 |
| Registers / Thread | 52 | 54 |
| Static Shared Memory / Block | 3.07 KB（≈3 KiB） | 12.29 KB（≈12 KiB） |
| Theoretical Occupancy | 100% | 100% |
| Achieved Occupancy（原始报告） | 75.57% | **101.80%（不能按物理占用率解释）** |
| DRAM Throughput | 1.67% | 1.23% |
| L1/TEX Cache Throughput | 76.90% | 80.98% |
| Waves per SM | 1.07 | 1.07 |

- **DRAM Throughput 低**：当前指标没有显示外部显存带宽已经饱和；并不说明没有任何访存瓶颈。
- **L1/TEX 利用率高**：可能涉及片上数据访问/LSU 压力，不能由此推出 L1 Cache Miss 或 Shared Memory Bank Conflict。
- **理论 100% Occupancy**：TILE32 并非简单由于理论 Occupancy 低而变慢。
- **Achieved 101.80%**：计数器测量/归一化差异，**不是硬件能超过 100% 活跃 Warp**。
- **Waves per SM ≈ 1.07**：两者都有较短尾波风险，NCU 的估计优化幅度不应被当作实测加速保证。

### 4.5 定向 Warp Stall（已运行）

```bash
for tile in 16 32; do
  ncu \
    --section SchedulerStats --section WarpStateStats \
    --graph-profiling node \
    --clock-control none --cache-control none \
    --kernel-name-base demangled \
    --kernel-name "regex:gemm_register2_kernel.*${tile}" \
    --launch-count 1 \
    --page details --print-details all \
    ./build-v2/gemm_bench_unified 256 256 256 \
    > "/tmp/minigpu_v3_stall_${tile}.log" 2>&1
done
```

| 指标 | TILE16 | TILE32 |
|---|---:|---:|
| Eligible Warps / Scheduler | 0.64 | 1.02 |
| Warp Cycles / Issued Instruction | 27.55 | 36.65 |
| **Stall MIO Throttle** | **14.73** | **24.06** |
| **Stall Long Scoreboard** | **2.62** | **17.76** |
| Stall Barrier | 3.24 | 1.69 |

**怎么解释**：

- `MIO Throttle`：Warp 受 MIO 相关指令管线/队列能力限制的等待，可能涉及 Shared Memory 等片上操作，**不是 Bank Conflict 的直接计数值**。
- `Long Scoreboard`：由于 L1TEX 相关内存操作的依赖，Warp 等待数据可用；不等于 DRAM 一定发生瓶颈。
- `Barrier`：等待同步条件满足；在 TILE32 中相对更低，因此不是本次最突出差异。
- 指标是 Nsight Warp 状态的归一化观测值，**不能作为整段 Kernel 耗时的百分比相加**；采样和统计可能有噪声。

**得到的可验证假设**：TILE32 的访存指令吞吐/依赖等待和大 Block 执行粒度可能影响性能。没有局部源代码/指令级证据时，不要把某种具体硬件机制写成“已确认根因”。

---

## 5. 最小 Roofline 理论：知道模型的边界

GEMM 浮点操作量：

\[
\operatorname{FLOPs}\approx 2MNK
\]

以 **A/B 每元素理想只从 DRAM 读取一次，C 每元素理想写入一次** 为假设，最低数据移动量：

\[
\operatorname{Bytes}_{min}=4(MK+KN+MN),\qquad AI_{ideal}=\frac{2MNK}{4(MK+KN+MN)}
\]

| Shape | FLOPs | 理想字节数 | 理想 AI |
|---|---:|---:|---:|
| 256³ | 33,554,432 | 786,432 bytes | ≈42.67 FLOP/byte |
| 512³ | 268,435,456 | 3,145,728 bytes | ≈85.33 FLOP/byte |

Roofline 基本关系：

\[
P_{attainable}\le\min(P_{peak},\;BW\times AI)
\]

这里的 AI 是基于**理想最低流量**的上界，并不是从 NCU 实测 DRAM 字节数计算所得。真实 AI 需与同一存储层级带宽口径相匹配。**不能单靠以上理想 AI 或 DRAM Throughput 两个百分比就证明 Compute Bound / Memory Bound。**

---

## 6. V3 唯一优化实验：显式循环展开 4 次

### 6.1 实验假设与改动

- **出发点**：TILE32 的 `MIO Throttle` 和 `Long Scoreboard` 较高。
- **假设**：将内部 K 计算循环部分展开，可能帮助编译器组织指令、提高指令级并行性或减少某些等待。
- **修改位置**：`gemm_bench_unified.cu` 中的 `gemm_register2_kernel<TILE>` 的 Shared Memory 内积循环。
- **修改方式**：仅针对 `TILE == 32` 使用 `#pragma unroll 4`；TILE16 保持原有循环行为。

实验核心片段：

```cpp
if (TILE == 32) {
    #pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        float a = As[ty][k];
        sum0 += a * Bs[k][tx];
        sum1 += a * Bs[k][tx + TILE];
    }
} else {
    for (int k = 0; k < TILE; ++k) {
        float a = As[ty][k];
        sum0 += a * Bs[k][tx];
        sum1 += a * Bs[k][tx + TILE];
    }
}
```

这是一次探索性编译器指令组织实验，**并没有证据预先保证**它一定能缓解 MIO 或 Long Scoreboard；可能编译器原本已做更有效的展开。要证实机制需额外的编译器/SASS 和前后硬件指标对照（本版未做，按精简原则不额外增加）。

### 6.2 实验复现方式

原始 V2 executable 作为不可修改对照：

```bash
cp build-v2/gemm_bench_unified /tmp/minigpu_v2_baseline
cmake -S . -B build-v3 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build-v3 --target gemm_bench_unified --parallel 4

for size in 256 512; do
  echo "BASELINE ${size}"
  /tmp/minigpu_v2_baseline "$size" "$size" "$size"
  echo "EXPERIMENT ${size}"
  ./build-v3/gemm_bench_unified "$size" "$size" "$size"
done
```

**复现实验的前提**：必须先重新应用上面的实验修改才能编译出实验版；截至本文最后状态，仓库中的 `gemm_bench_unified.cu` 已恢复 V2 原版，单纯运行上述构建命令**不能**重现已撤销的修改。先前改动补丁曾保存在 `/tmp/minigpu_v3_unroll4.diff`（`/tmp` 文件不保证跨重启保留）。不要错误宣称 V3 当前源码仍带此优化。

V2 Benchmark 统一采用 CUDA Graph：64 GEMMs / Graph，16 Graph Replay / 轮，5 轮，中位数；计时不包括显存申请、H2D/D2H、Graph 构建和 CPU Reference。所有对照都以**不启用 Profiler**的正常运行输出为准。

### 6.3 正确性与性能结果（开发者真实输出）

| Shape | Kernel | 原始 V2 Baseline | V3 Unroll4 | 变化 |
|---|---|---:|---:|---:|
| 256³ | Register2_16 | 39.250 μs | 39.308 μs | +0.15% |
| **256³** | **Register2_32** | **50.582 μs** | **52.729 μs** | **延迟 +4.24%** |
| 512³ | Register2_16 | 275.952 μs | 280.692 μs | +1.72% |
| **512³** | **Register2_32** | **265.928 μs** | **274.702 μs** | **延迟 +3.30%** |

两组 Shape 的两种 Register2 实现在相应输出里都显示 `PASS`。只做了这些独立进程对照，没有充分独立重复和统计置信区间，所以不能证明 3%～4% 差异在所有运行条件下必然存在。但**本次并无证据支持采用该修改**。

### 6.4 工程结论和处理

- **结论**：手动 `#pragma unroll 4` 不是这个版本的已验证加速措施；两组 Shape 的本次测量都出现退化。
- **不应该写成事实的原因**：“寄存器变多导致退化”“指令数增加导致退化”“Bank Conflict 导致退化”等；因为未做对应的前后硬件/SASS 对照。
- **保留工程版本**：撤销尝试，保持发布过的 V2 Kernel，避免把性能退化的改动带入 V4。

---

## 7. 回退和恢复后验证（最新状态）

开发者已实际执行：

```bash
git diff -- gemm_bench_unified.cu > /tmp/minigpu_v3_unroll4.diff
git restore --source=v2.0 -- gemm_bench_unified.cu
git diff --exit-code v2.0 -- gemm_bench_unified.cu
# 输出：[OK] 已恢复 V2 Kernel

cmake --build build-v3 --target gemm_bench_unified --parallel 4
./build-v3/gemm_bench_unified 256 256 256
```

**恢复后的 256³ 实测：**

| Kernel | 正确性 | Latency |
|---|---|---:|
| Naive | PASS | 80.776 μs |
| Tiled16 | PASS | 42.487 μs |
| Register2_16 | PASS | **39.216 μs** |
| Register2_32 | PASS | 50.570 μs |
| cuBLAS SGEMM | PASS | 15.621 μs |

本次 Naive 波动较大，不能用这一次变化推断 Naive 算法发生实质退化；Register2 16/32 基本回归到既有性能区间。先前完整 V2 的 **25/25 PASS** 已完成；截至本报告，尚未在本轮回退后重新运行所有五组 Shape。

用户最新的 Git 输出（V3 尚未发布）：

```text
A  docs/V3_PROFILING.md
 docs/V3_PROFILING.md | 106 +++++++++++++++++++++++++++++++++...
```

即当前暂存的是上一份 **106 行** 的精简版报告；源码已恢复并与 `v2.0` 一致。**如果采用这份完整版，请替换 `docs/V3_PROFILING.md` 并重新 `git add`；不要误以为完整版已自动进入你的 WSL 工作目录。**

---

## 8. 知识点自测（面试准备）

### 8.1 必须真正理解的八个概念

1. **CUDA Event vs Nsight Systems CUDA API Time**：前者可测设备流时间段，后者的 `cudaEventSynchronize` 主要表现 CPU 等 GPU 完成的等待时间，不能混淆。
2. **Graph 级和 Node 级追踪**：Graph 级记录整体 Graph 执行，节点级记录内部 Kernel；采集粒度决定能回答什么问题。
3. **Occupancy**：SM 的活跃 Warp 相对上限的比例；不是计算单元利用率，更不是越高越快。
4. **Register Pressure**：每线程寄存器占用会影响 Block 驻留及指令/数据布局，但要通过资源限制或实测确认后果。
5. **MIO Throttle**：片上/特定 MIO 相关指令管线有压力的线索，不能等价于 Bank Conflict。
6. **Long Scoreboard**：L1TEX 相关存储依赖等待，不等于 DRAM 带宽已经用满。
7. **Arithmetic Intensity / Roofline**：同一存储层级的 FLOPs/Bytes 与带宽上界模型，理想 AI ≠ 实测 AI。
8. **Evidence-driven optimization**：必须有可证伪假设、单一修改变量、正确性、正常运行性能对照，并能接受退化时回退。

### 8.2 一段源码解释题

解释 `gemm_register2_kernel<32>` 中 `col0`、`col1`、`As`、`Bs`、`sum0/sum1`、两个 `__syncthreads()` 的作用；为什么每线程处理两个输出、为什么最后要检查边界。

**参考要点**：`col1=col0+TILE`；`sum0/sum1` 对应两个列方向输出，复用同一个 A 元素；共享内存中 A/B 的加载在第一次 barrier 后被所有线程正确读取，第二次 barrier 保护下一轮覆写。测试 `130×65×70` 可覆盖非 TILE 整倍数。

### 8.3 计算题

**题目**：Register2 TILE32，`M=N=K=256`。求 Block 线程数、Grid Block 数、每块静态 Shared Memory 理论容量。

**答案**：线程 `32×32=1024`；Grid `(N/(2TILE), M/TILE)=(4,8)`，总 `32 Blocks`；共享内存 `(32²+32×64)×4=12,288 bytes=12 KiB`。

### 8.4 工程分析题与三道递进追问

**问题**：TILE32 的 MIO Throttle 更高，为什么不能认定就是 Bank Conflict？

**参考回答**：MIO Throttle 衡量 Warp 因 MIO 管线资源压力而不能继续发射/执行，涵盖的原因比 Shared Memory Bank Conflict 更广。要证实 Bank Conflict，必须观察针对 Shared Memory 的冲突/请求或 Wavefront 指标，并定位到具体指令或访问模式；当前报告没有这一层证据。

- **追问 1**：DRAM Throughput 低，是否就说明 Kernel 是 Compute Bound？  
  **答**：否。可能瓶颈在 Shared Memory/L1/LSU、指令依赖或调度；需进一步看 FP32 管线利用率、存储层级吞吐、实际字节数与 Roofline。
- **追问 2**：为什么 TILE32 理论 Occupancy 为 100%，仍可能比 TILE16 慢？  
  **答**：Occupancy 只代表可驻留 Warp 资源上限，不能表示有用指令执行速率；大 Block、访存指令吞吐/依赖、有限 Grid/尾波和调度都可能影响性能。
- **追问 3**：为什么 `#pragma unroll 4` 性能下降，能否直接说寄存器压力增加？  
  **答**：不能。可以将寄存器、指令调度、已有编译器优化等列为可能解释，但本次未比较前后 SASS/寄存器/NCU Stall，故只得出“本次未加速，决定回退”的可靠结论。

---

## 9. 给下一个 AI 的交接说明（请优先读本节）

### 9.1 已完成且不要重复的工作

- V2：Naive、Tiled16、Register2 TILE16/TILE32、cuBLAS、CUDA Graph，五组 Shape × 五种实现 **25/25 PASS**，Tag `v2.0` 已推送。
- V3：Nsight Systems CUDA API/Graph 汇总；Nsight Compute GPU 基础资源；SchedulerStats + WarpStateStats；256³ 的瓶颈线索；一次 Unroll4 实验（两个 Shape）；正确性检查；决定撤销性能退化修改。
- WSL / CUDA / ncu / nsys 已可正常使用。**不要重复安装 CUDA、反复检查 PATH、重新走 ncu 权限配置。**
- `gemm_bench_unified.cu` 已恢复 `v2.0` 源码；`build-v3` 重新编译后 256³ 全部 5 种实现 PASS。

### 9.2 仍未完成（如需发布 V3）

1. 将本**完整版**覆盖 WSL 中的 `docs/V3_PROFILING.md` 并执行 `git add docs/V3_PROFILING.md`（当前 Git 已暂存的是 106 行版本）。
2. 若按照最初 V3 严格验收，可在恢复原版后补齐 **五组 Shape** 的回归；但避免其他冗余的 Profiling。`v2.0` 原始完整测试已通过。
3. 完成一次实际面试口述验收；本文件附带参考题，但**有参考答案不等于用户已经独立回答通过**。
4. 检查 `git status --short`，再提交和推送 `v3.0`。**截至本文没有 V3 Commit/Tag 的成功证据，不能宣称已发布。**

### 9.3 下一阶段 V4 既定方向

简化为 **Reduction/Softmax → RMSNorm → QKᵀ/Softmax/PV 基础 Attention → 一次 Scale+Softmax Fusion**。优先复用现有 GEMM/cuBLAS，避免重新造 GEMM 或构建工业级 Runtime；模型反向、KV Cache、FlashAttention 高级版本不属于 V4 的最小验收。学习仍需源码解释、计算题、工程分析和连续追问。

### 9.4 核心纪律

**真实数据优先；不能把 Nsight 推测当成证明；不把 Profiler 内延迟当成公平 Benchmark；源码恢复后别声称实验修改仍在；不为了微小性能数字无限调参；保持 V1～V6 同一工程持续演进。**
