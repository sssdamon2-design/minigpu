# MiniGPU V1 实验记录（真实运行数据）

> 数据来源：项目开发过程中的 Ubuntu 终端实机输出，记录截至 2026-10-22。  
> 设备：NVIDIA RTX 2060（6 GB、Compute Capability 7.5 / `sm_75`），WSL2 Ubuntu 24.04.x，CUDA Toolkit 12.8。  
> **没有依据的结果不填数值；未展示的 Profiler 指标不推测为已测得。**

## 1. 功能验证概况

V1 使用 CPU Reference 与 CUDA GPU 实现比较：

```text
Y = GEMM_2(ReLU(GEMM_1(X, W1)), W2)
```

- GEMM：FP32，Row-Major；每个线程计算一个输出元素，线程索引通过 `row = idx / N`、`col = idx % N` 映射。
- ReLU：GPU Kernel 原地修改中间 Tensor `H`。
- GPU 数据流：H2D(X/W1/W2) → GEMM1 → ReLU → GEMM2 → D2H(Y)。
- 使用 `cudaGetLastError()` 检查 Launch 错误，用 CUDA Event 同步/检测被测 Stream 完成情况。
- `cudaMalloc/cudaFree` 已实现五个 Device Buffer 的生命周期管理。

## 2. 多 Shape 正确性测试结果

| Case | BATCH | IN | HIDDEN | OUT | 输出元素 | 验收 | 证据说明 |
|---|---:|---:|---:|---:|---:|---|---|
| 1 原始手写矩阵 | 2 | 3 | 4 | 2 | 4 | **PASS** | CPU/GPU 4 个值一致 |
| 2 非对齐 Shape | 1 | 7 | 5 | 3 | 3 | **PASS** | `Diff = 0.00000000`，检查 3 个输出 |
| 3 中等 Shape | 8 | 16 | 32 | 8 | 64 | **PASS** | `Checked Elements: 64`，最大绝对误差打印为 0 |
| 4 较大 Shape | 32 | 64 | 128 | 32 | 1024 | **PASS** | `Checked Elements: 1024`，最大绝对误差打印为 0 |

**Case 1 的实际输出**：

```text
CPU MLP Output:
0.0 2.0
6.0 0.0
CPU[0] = 0.0, GPU[0] = 0.0
CPU[1] = 2.0, GPU[1] = 2.0
CPU[2] = 6.0, GPU[2] = 6.0
CPU[3] = 0.0, GPU[3] = 0.0
Test: PASS
```

**Case 2 的实际输出（摘录）**：

```text
CPU[0] = -0.300781, GPU[0] = -0.300781, Diff = 0.00000000
CPU[1] = -0.218750, GPU[1] = -0.218750, Diff = 0.00000000
CPU[2] = -0.136719, GPU[2] = -0.136719, Diff = 0.00000000
Test: PASS
```

**Case 3 的实际输出（末尾）**：

```text
Checked Elements: 64
Max Absolute Difference: 0.00000000
Test: PASS
```

**Case 4 的实际输出（末尾）**：

```text
Checked Elements: 1024
Max Absolute Difference: 0.00000000
Test: PASS
```

### FP32 容差规则

```cpp
const float atol = 1e-5f;
const float rtol = 1e-4f;
float diff = std::fabs(cpu - gpu);
float tolerance = atol + rtol * std::fabs(cpu);

if (!std::isfinite(cpu) || !std::isfinite(gpu) || diff > tolerance)
    pass = false;
```

当前测试输出的 `0.00000000` 为格式化后的八位小数；在这些测试数据中 CPU/GPU 实际比较通过，但**不等同于证明任意数据下都位精确相等**。输入系数主要为二进制可精确表示的分数。

## 3. 性能计时方法

```text
cudaMalloc + H2D（区间外）
    ↓
Warmup: 20 × (GEMM1 → ReLU → GEMM2)
    ↓ cudaDeviceSynchronize()
cudaEventRecord(start)
    ↓
Benchmark: 1000 × (GEMM1 → ReLU → GEMM2)
    ↓
cudaEventRecord(stop)
    ↓ cudaEventSynchronize(stop)
cudaEventElapsedTime(total_ms, start, stop)
    ↓
avg_ms = total_ms / 1000
    ↓
D2H + CPU/GPU 对比（区间外）
```

- 同一默认 Stream 中三个 Kernel 按提交顺序执行。
- 正式 Benchmark 循环内部**不调用** `cudaDeviceSynchronize()`，避免每轮强制 CPU 等待 GPU。
- Event 时间是**GPU 时间线上的经过区间**，包含 Kernel 之间潜在空闲时间，不能直接视为纯算术计算耗时。
- 输出 `avg_ms` 是重复区间平均，而非一次独立服务请求的端到端 Latency。

## 4. 实测性能数据

### 4.1 初始单次 CUDA Event

| 测试形式 | 实测结果 |
|---|---:|
| 未采用正式 Warmup + 1000 次重复的单次 MLP Event 区间 | **0.431936 ms** |

这个数值**不应**与后续结果直接做“Kernel 加速倍数”对比。

### 4.2 原始小尺寸 MLP，四次独立运行

Shape：`BATCH=2, IN=3, HIDDEN=4, OUT=2`；每次 `Warmup=20`，`Repeat=1000`。

| 运行 | Total GPU Event Time (ms) | Avg per MLP (ms) | PASS |
|---|---:|---:|---|
| 第 1 次 | 32.570496 | 0.032570 | 是 |
| 第 2 次 | 30.836321 | 0.030836 | 是 |
| 第 3 次 | 29.691809 | 0.029692 | 是 |
| 第 4 次 | 39.361183 | 0.039361 | 是 |
| 四次平均（由原始 total/1000 再平均） | — | **约 0.033115** | — |

观察：平均区间约 **29.692–39.361 μs**，有可见波动；没有通过 Profiler 定量判断具体来源。

### 4.3 不同 Shape 的实测例子

| Case / 运行环境 | Total Time (ms) | Avg Time (ms) | 输出验证 |
|---|---:|---:|---|
| Case 2：`1,7,5,3` | 38.919647 | 0.038920 | PASS / 3 个元素 |
| Case 3：`8,16,32,8` | 30.070816 | 0.030071 | PASS / 64 个元素 |
| Case 4：`32,64,128,32` | 38.612991 | 0.038613 | PASS / 1024 个元素 |
| CMake 生成的 `build-v1/mlp`（Case 1） | 29.933855 | 0.029934 | PASS / 4 个元素 |
| CMake 生成的 `build-v1/mlp_shape_test`（Case 3） | 29.401920 | 0.029402 | PASS / 64 个元素 |

**注意**：这些结果来自不同批次的独立运行，不属于严格控制 GPU 频率、后台负载和功耗条件的 benchmark suite；不宜跨 Case 声称确切性能差异或计算缩放规律。

## 5. CMake 验收

两个实机运行命令：

```bash
./build-v1/mlp
./build-v1/mlp_shape_test
```

两者均输出 `Test: PASS`，表明当前 CMake 配置已能构建并运行两个 CUDA 程序。未进行自动化测试套件（CTest）的验证。

推荐首次配置和构建：

```bash
cmake -S . -B build-v1 -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-v1 --parallel 4
```

## 6. 已验证 / 未验证的边界

**已验证**：CUDA 编译和运行、Naive GEMM CPU/GPU 对比、ReLU、MLP 三 Kernel 顺序执行、五块 Device Buffer、四组 Shape 的 CPU/GPU 对比、CUDA Event 计时、CMake 构建两个目标。

**尚未验证或实现**：Tiled GEMM、cuBLAS 性能基线、Tensor Core、Nsight 硬件计数器、GPU Utilization / Memory Bandwidth 的实测、稳定的独立请求 p50/p95 End-to-End Latency、随机测试集、自动运行所有 Shape 的 CTest、PyTorch 自定义算子接入、真实 Transformer 模型优化。

## 7. 工程结论

V1 达到了“**CUDA GPU 算子 + 双层 MLP 推理闭环 + CPU Reference + 正确性与初步计时 + CMake**”的目标，适合作为 V2 GEMM 访存优化的正确性基线。面试或简历中应注明：V1 重点是**正确性和系统流程**，并非已完成 GPU Kernel 优化。 

下一阶段：在保留 CPU 参考、维度/容差测试和同等 benchmark 条件的基础上实现 **Shared Memory Tiling GEMM**，再与 Naive GEMM、cuBLAS 比较。
