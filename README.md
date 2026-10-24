# MiniGPU — 从 CUDA Kernel 到可复现的 GEMM 性能分析

MiniGPU 是一个循序渐进的 **C++ / CUDA / AI Systems** 学习型工程：从基础 GPU Kernel 与双层 MLP 前向计算出发，逐步实现 FP32 GEMM（矩阵乘法）的 Shared Memory Tiling、Register Blocking，并使用 **NVIDIA cuBLAS** 作为优化对照，通过 **CUDA Event / CUDA Graph** 记录性能。

> **当前状态（V2，2026-10-25）：** V1 的基本 CUDA / MLP 闭环和 V2 的五种 GEMM 统一正确性及 Benchmark 已在 RTX 2060 + WSL2 实机运行。V3–V6 属于后续计划，**未在本版本实现**。
>
> 本项目面向学习与性能工程实践，**不是**工业级 GEMM 库、完整 AI 推理引擎或 PyTorch 替代品。

## 1. 当前完成了什么

| 阶段 | 已完成内容 | 状态 |
|---|---|---|
| V1 | CUDA 线程索引、Vector Add、Naive GEMM、ReLU、双层 MLP Forward、CPU Reference、CMake 与基础 Benchmark | 已完成基础验收 |
| V2.1 | 独立 FP32 Naive GEMM 基线 | 已完成 |
| V2.2–V2.3 | Shared Memory Tiling、访存复用与线程布局分析 | 已完成 |
| V2.4 | 每线程两个输出（1×2 Register Blocking）、TILE=8/16/32 实验 | 已完成 |
| V2.5 | 5 组 Shape 的正确性与初步性能对比 | 已完成 |
| V2.6 | cuBLAS SGEMM 接入、统一 CUDA Event、CUDA Graph Benchmark | 已完成 |
| V2.7 | 固定随机输入、FP64 累加 CPU Reference、数值误差排查、性能报告 | 测试完成，文档与仓库整理中 |

### V2 的五种对比实现

| 实现 | 主要设计 |
|---|---|
| **Naive** | 每个线程计算一个输出元素；FP32 累加 |
| **Tiled16** | `16×16` Shared Memory Tile；块内复用输入 |
| **Register2_16** | TILE=16；每个线程使用两个累加器计算两个列方向输出 |
| **Register2_32** | TILE=32；相同的 1×2 Register Blocking 思路 |
| **cuBLAS_SGEMM** | `cublasSgemm()`；作为 NVIDIA 优化库参考 |

矩阵使用 **Row-major** 存储：`A[M,K] × B[K,N] = C[M,N]`。传统 cuBLAS SGEMM 采用 Column-major 解释矩阵，调用中通过 `Cᵀ=BᵀAᵀ` 交换 `B/A` 参数顺序和 `N/M` 维度，**不执行额外的物理转置**。

## 2. 实机测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060，6144 MiB 显存，Turing / SM75 |
| 宿主与 Linux | Windows + WSL2，Ubuntu 24.04.5 LTS |
| NVIDIA Windows 驱动 | 591.74 |
| CUDA Toolkit / NVCC | 12.8.93 |
| 构建工具 | CMake + NVIDIA CUDA Compiler，C++17 / CUDA17 |

**版本区别：**`nvidia-smi` 显示的 `CUDA Version: 13.1` 是驱动支持的 CUDA 版本，不是本工程实际使用的 CUDA Toolkit；`nvcc --version` 显示 **12.8.93**。

WSL 中如果终端找不到 `nvidia-smi`，可运行：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

无需因此额外安装 Linux NVIDIA 显卡驱动。

## 3. 编译与运行

项目使用 CUDA 12.8 路径；RTX 2060 对应的目标架构是 `sm_75`。

```bash
cd ~/projects/minigpu

# 配置独立的 V2 构建目录
cmake -S . -B build-v2 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8

# 只编译 V2 的统一正确性和 CUDA Graph 测试目标
cmake --build build-v2 --target gemm_bench_unified --parallel 4

# 大矩阵与非整除尺寸
./build-v2/gemm_bench_unified 512 512 512
./build-v2/gemm_bench_unified 130 65 70
```

命令行三个参数依次是 **`M K N`**，即 `A[M,K] × B[K,N]`。不输入时，当前程序默认使用 `512 512 512`。

批量运行五组已验证的 Shape：

```bash
for shape in '64 64 64' '128 128 128' '256 256 256' \
             '512 512 512' '130 65 70'; do
  read -r M K N <<< "$shape"
  ./build-v2/gemm_bench_unified "$M" "$K" "$N" || exit 1
done
```

如果你修改源码，务必**先确认编译成功再运行**，避免误将上次成功构建的可执行文件当成新结果：

```bash
cmake --build build-v2 --target gemm_bench_unified --parallel 4 \
  && ./build-v2/gemm_bench_unified 512 512 512
```

早期学习阶段的独立测试程序（如 `gemm_bench.cu`、`gemm_tiled.cu`、`gemm_register_2*.cu`、`gemm_compare.cu`）仍保留，用于复现逐步优化过程；日常 V2 结果以 `gemm_bench_unified` 为统一测试入口。

## 4. 正确性验证

- **输入：** FP32，`std::mt19937(2026)` 固定种子，均匀分布在 `[-1,1]`。
- **参考计算：** CPU 接收 FP32 输入，使用 FP64 完成乘法与累加，最后转换回 FP32 参考输出。
- **逐元素判定：** `abs(gpu - ref) <= 2e-5 + 1e-4 * abs(ref)`，同时检查 NaN/Inf；测试前将 GPU 输出填为 NaN，以发现遗漏写回。
- **已实测：** `64³`、`128³`、`256³`、`512³`、`130×65×70` 五种 Shape × 五种实现，**25/25 PASS**。

验证容差是为**当前数据范围和 K≤512** 的实验选取的，不代表任意矩阵规模或任意输入分布都能达到同样的数值误差。曾对近零输出发生的单点容差边界问题进行 FP64 诊断；不能要求不同 FP32 GEMM 实现与 CPU 逐位完全相同。

## 5. CUDA Graph 性能测试

所有实现共用设备输入 A/B、同一个 CUDA Stream 和 CPU Reference。统一计时设置为：

1. 每种实现普通启动预热 **20 次**。
2. 捕获 **64 次 GEMM** 为一个 CUDA Graph，实例化成可回放 GraphExec。
3. 正式计时前预热 Graph **3 次**。
4. 每轮 CUDA Event 测量 **16 次 Graph Replay**，相当于 **1024 次 GEMM**。
5. 每个实现进行 **5 轮**测量，以每轮总时间除以 1024，再取**中位数**。

`GFLOPS = 2 × M × N × K / (latency_ms × 10^6)`。

**口径：**不计 GPU 显存分配、Host↔Device 传输、CPU Reference 和 Graph 捕获/实例化；计入 GPU 时间线上 Graph 回放的执行及可能存在的调度间隔。它是重复回放下的平均 GEMM 时间，**不是**端到端 API 调用延迟，也不代表完全去除一切调度影响后的纯指令耗时。

## 6. V2 实测结果

以下是**随机 FP32 + CPU FP64 累加参考值**条件下的 CUDA Graph 结果；单位 **μs**，越低越好：

| Shape | Naive | Tiled16 | Register2_16 | Register2_32 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 64³ | 5.092 | **4.986** | 6.041 | 13.155 | 6.955 |
| 128³ | **9.306** | 9.599 | 10.229 | 12.672 | 5.084 |
| 256³ | 60.234 | 42.888 | **39.179** | 50.580 | 15.838 |
| 512³ | 502.394 | 320.928 | 274.126 | **265.238** | 70.168 |
| 130×65×70 | 4.031 | **3.997** | 4.064 | 9.906 | 4.455 |

**粗体表示该次测试中最快的自定义实现，不代表比 cuBLAS 更快。**

对于 `512×512×512`：

- Naive：**502.394 μs**，约 **534 GFLOPS**。
- Register2_32：**265.238 μs**，约 **1.01 TFLOPS**；相对 Naive 获得 **1.89× 加速**。
- cuBLAS：**70.168 μs**，约 **3.83 TFLOPS**；相对最快自定义实现快 **3.78×**。

**小矩阵注意：**64³ 的三次独立进程复测，Tiled16 与 cuBLAS 的相对比值较稳定，但绝对延迟曾有接近两倍变化；130×65×70 的 Naive、Tiled16 与 Register2_16 很接近，三轮中最快实现有所变化。因此不把小矩阵的单次领先结果推广成通用性能结论。

完整数值、复测原始记录、优化分析及限制见 [V2 性能报告](docs/PERFORMANCE.md)。

## 7. 代码与文档索引

```text
minigpu/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── mlp.cu                      # V1：双层 MLP Forward
├── mlp_shape_test.cu           # V1：多 Shape 正确性
├── gemm_bench.cu               # V2.1：Naive GEMM 基线
├── gemm_tiled.cu               # V2.2：Shared Memory Tiling
├── gemm_register_2.cu          # V2.4：Register2 TILE16
├── gemm_register_2_tile8.cu    # V2.4：Register2 TILE8
├── gemm_register_2_tile32.cu   # V2.4：Register2 TILE32
├── gemm_compare.cu             # V2.6：独立 cuBLAS 正确性/计时
├── gemm_bench_unified.cu       # V2.6–2.7：五种实现统一 Graph Benchmark
└── docs/
    ├── v1_results.md
    ├── v2_naive_gemm_baseline.md
    ├── PERFORMANCE.md
    └── ...
```

上述为 V1/V2 的**主要文件**示意，完整文件清单以仓库当前内容为准。阶段性实验代码（如 `gemm_bench_graph.cu`、`gemm_tiled_1.cu`）可按需要保留，不作为主入口。

## 8. 结论与后续方向

MiniGPU V2 展示了从 Naive GEMM 到 Shared Memory Tiling、每线程两个输出的 Register Blocking，再到 cuBLAS 对照及 CUDA Graph 统一 Benchmark 的逐步优化过程，并记录了多尺寸、随机 FP32 的实机验证结果。

下一阶段计划（**尚未实现**）：

- **V3：** 使用 Nsight Systems / Nsight Compute 分析访存、占用率、指令与执行瓶颈，再决定进一步优化方向。
- **V4：** Reduction、Softmax、RMSNorm、基础 Attention 与 Kernel Fusion。
- **V5：** Buffer RAII、Stream/Event、显存复用等简化 GPU Runtime。
- **V6：** 小型 Transformer 的实际 Profiling、自定义 CUDA Kernel 集成与端到端验证。

**工程边界：**目前尚未完成上述 V3–V6 模块，也没有证据证明自定义 Kernel 在所有 Shape 上优于 cuBLAS，更不能声称当前是通用高性能 GEMM 库。
