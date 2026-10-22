# MiniGPU：CUDA GPU 算子与轻量执行引擎

> 项目状态：**V1 基础 CUDA 推理闭环与 CMake 构建已实测通过**；V2–V6 为后续规划。  
> 当前平台：NVIDIA RTX 2060（Compute Capability 7.5 / `sm_75`）、WSL2 Ubuntu 24.04、CUDA Toolkit 12.8。  
> 本仓库的 V1 是**手写 CUDA 算子与推理执行流程**，不是训练框架、完整模型推理框架或工业级高性能 GEMM 库。

## 1. 项目目标

MiniGPU 是一个面向 **GPU Kernel / AI Systems / Inference Runtime / 性能优化** 的渐进式 C++/CUDA 工程。以 CPU Reference 和可复现的正确性测试为基础，逐步形成：

- **V1（已完成主要工程验收）**：CUDA 基础、Vector Add、Naive GEMM、ReLU、双层 MLP Forward、CUDA Event Benchmark、多 Shape FP32 正确性验证、CMake 构建。
- **V2（下一阶段）**：Shared Memory Tiled GEMM、Coalescing、Register Blocking、cuBLAS 基线比较。
- **V3**：Nsight Systems / Nsight Compute 性能分析、Roofline 思路及证据驱动优化。
- **V4**：Reduction、Softmax、RMSNorm、基础 Attention 与 Kernel Fusion。
- **V5**：小型 GPU Runtime：RAII Buffer、内存复用、Stream/Event、执行顺序管理。
- **V6**：真实 PyTorch 小型 Transformer 的 Profiling、Custom CUDA 集成及端到端验证。

**原则**：V1 只证明算子链路可正确执行和测量；不宣称 GEMM 性能已优化，也不宣称比 PyTorch/cuBLAS 更快。

## 2. V1 推理计算图

采用 FP32、Row-Major 数据布局，不含 Bias、训练或反向传播：

```text
Host（CPU）                               Device（GPU）
X [BATCH, IN]  ──── H2D ────────────────► d_X
W1 [IN, HIDDEN] ── H2D ────────────────► d_W1
W2 [HIDDEN, OUT] ─ H2D ────────────────► d_W2
                                             │
                                        GEMM 1: X × W1
                                             ▼
                                      d_H [BATCH, HIDDEN]
                                             │
                                        ReLU（原地修改）
                                             │
                                        GEMM 2: H × W2
                                             ▼
                                      d_Y [BATCH, OUT]
CPU Reference ◄───── D2H ──────────────── GPU Output
          └── atol/rtol 正确性比较
```

数学表达式：

\[
H = \operatorname{ReLU}(XW_1),\qquad Y = HW_2
\]

Kernel 使用一维线程索引：`idx = blockIdx.x * blockDim.x + threadIdx.x`。Naive GEMM 每线程计算一个 `C[row, col]`，并使用 `if (idx >= M*N) return;` 保护非对齐尺寸。

## 3. 开发与构建环境

| 项目 | 已使用的环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 6 GB，`sm_75` |
| 开发系统 | WSL2 Ubuntu 24.04.x（aarch64 **不是**本项目平台） |
| CUDA Toolkit | 12.8；NVCC 路径 `/usr/local/cuda-12.8/bin/nvcc` |
| C++ 编译器 | GCC/G++ 13.3 |
| 构建工具 | CMake 3.28.x、Ninja 1.11.x |
| 源码目录 | `~/projects/minigpu` |

需要 Ubuntu 中可执行的 NVCC、CMake 和 Ninja；GPU 通过 WSL2 使用 Windows 侧 NVIDIA 驱动。不要为 WSL2 随意额外安装 Linux NVIDIA 显卡驱动。

## 4. 工程目录

```text
minigpu/
├── CMakeLists.txt                 # C++ / CUDA 构建目标
├── README.md
├── .gitignore
├── hello.cu                       # CUDA Hello / 线程索引入门
├── vector_add.cu                  # Vector Add / H2D、D2H、cudaMalloc
├── gemm_naive.cu                  # CPU/GPU Naive GEMM
├── mlp.cu                         # 原始小尺寸双层 MLP + Benchmark
├── mlp_shape_test.cu              # vector 动态 Tensor + Shape 验证
└── docs/
    ├── v1_results.md             # 已实测数据及其解释边界
    └── V1_详细学习总结.md          # 从原理到面试的学习/AI 交接材料
```

构建目录如 `build-v1/` 不应提交 Git。**本项目暂不使用 `CMakePresets.json`**。

## 5. CMake 编译和运行

首次配置（选择 Ninja、明确 CUDA 编译器和 `sm_75`；如果现有 `CMakeLists.txt` 已设置这些变量，重复指定不会增加必要性）：

```bash
cd ~/projects/minigpu
cmake -S . -B build-v1 -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-v1 --parallel 4
```

运行 V1 的两个核心目标：

```bash
./build-v1/mlp
./build-v1/mlp_shape_test
```

两个目标都已在实机的 CMake 构建产物上运行并输出 `Test: PASS`。修改 `.cu` 后使用 `cmake --build build-v1 --parallel 4` 重新构建。

手工 NVCC 编译（排错时可用）：

```bash
/usr/local/cuda-12.8/bin/nvcc -arch=sm_75 mlp.cu -o test
./test
```

**编译器路径要写 `cuda-12.8`（连字符），不是 `cuda_12.8`（下划线）。** 若错误路径已被 CMake 缓存，建议重新配置到全新的构建目录。

## 6. 正确性测试

| Case | BATCH | IN | HIDDEN | OUT | 已检查输出 | 实机状态 |
|---|---:|---:|---:|---:|---:|---|
| 1：原始 MLP | 2 | 3 | 4 | 2 | 4 | PASS |
| 2：非对齐尺寸 | 1 | 7 | 5 | 3 | 3 | PASS |
| 3：中等尺寸 | 8 | 16 | 32 | 8 | 64 | PASS |
| 4：较大尺寸 | 32 | 64 | 128 | 32 | 1024 | PASS |

- Case 1 使用固定手写矩阵，输出 `[[0, 2], [6, 0]]`。
- Case 2–4 使用 `std::vector<float>` 和确定性公式初始化数据；所有输出元素均已检查，终端仅显示前几项。
- FP32 检查采用 `abs(cpu-gpu) <= atol + rtol * abs(cpu)`（`atol=1e-5`、`rtol=1e-4`），并检查有限值。
- **限制**：目前 `mlp_shape_test.cu` 通过修改源码中的四个维度常量、重新编译来切换 Case；尚无一个命令批量执行四个 Case 的自动化测试框架。

测试过程和计时说明详见 [V1 实验记录](docs/v1_results.md)。

## 7. 性能指标如何理解

- Warmup：20 次完整 MLP；Benchmark：1000 次完整 MLP。
- 以 `cudaEventRecord(start)` / 三个 Kernel × 1000 / `cudaEventRecord(stop)` / `cudaEventSynchronize(stop)` / `cudaEventElapsedTime` 测量 GPU Event 区间。
- 报告的平均值为 `total_ms / 1000`，包含被测 Stream 区间内可能存在的任务间隙。
- **不包含**计时区间外的 GPU 显存分配、输入 H2D 传输和输出 D2H 传输；因此**不是端到端推理耗时**，也不保证等于三种 Kernel 各自纯计算耗时之和。
- 原始单次测量 `0.431936 ms` 与预热后重复测试条件不同，**不能宣称 Kernel 得到了约十几倍性能加速**。

## 8. 已知限制与后续工作

1. Naive GEMM 没有 Shared Memory 数据复用、寄存器分块或 Tensor Core 优化；V2 才开始实现。
2. 目前测试数据规模偏小且部分输入系数可被二进制精确表示；更多随机数据、边界 Shape、零维与溢出防护属于后续完善项。
3. 当前 V1 中 C++/CUDA Buffer 生命周期使用手工 `cudaMalloc/cudaFree`；V5 再用 RAII 管理。
4. 当前没有 PyTorch 自定义算子集成、实际 Transformer 模型加速或 cuBLAS 性能对比，不能在简历中宣称已经完成。
5. V1 的主要价值在于**正确性、代码理解、系统数据流和可复现的初步 Benchmark**。

## 9. 学习及交接材料

- [V1 实测记录：`docs/v1_results.md`](docs/v1_results.md)
- [V1 详细学习总结：`docs/V1_详细学习总结.md`](docs/V1_详细学习总结.md)

如要让其他 AI 接手，先发这两份文档，并让其检查当前仓库真实源码；不要把计划中的 V2–V6 当作已完成工作。
