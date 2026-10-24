# MiniGPU V2 — GEMM 性能与正确性报告

> 测试对象：FP32 CUDA GEMM（Naive、Shared Memory Tiled、Register Blocking）与 NVIDIA cuBLAS SGEMM。  
> 数据来源：开发者在 WSL2 / RTX 2060 上于 2026-10-25 运行程序后记录的终端输出。  
> 性能数据为**指定硬件、软件版本和测试条件下的观测结果**，不代表其他环境中的普遍性能。

## 1. 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 (6 GiB / 6144 MiB) |
| GPU 架构 | Turing，计算能力 7.5（SM75） |
| 宿主系统 | Windows，运行 WSL2 |
| Linux 发行版 | Ubuntu 24.04.5 LTS |
| NVIDIA 驱动 | 591.74 |
| `nvidia-smi` 工具版本 | 590.52.01 |
| `nvidia-smi` 显示 CUDA Version | 13.1（驱动所支持的 CUDA 版本，并非当前编译工具链） |
| CUDA Toolkit / nvcc | 12.8.93 |
| 编译环境 | CMake + nvcc；构建目标 `gemm_bench_unified`，GPU 架构 SM75 |

**版本解释：**`nvidia-smi` 所示的 CUDA 13.1 不代表 Toolkit 安装为 13.1；实际使用的编译器版本以 `nvcc --version` 为准，即 12.8.93。

## 2. 实现与算法

| 实现 | 核心方法 | 当前版本特征 |
|---|---|---|
| Naive | 每个线程计算一个输出元素，直接访问全局内存 | FP32 累加 |
| Tiled16 | 16×16 Shared Memory Tiling，分块复用 A/B | Block 16×16；FP32 累加 |
| Register2_16 | 每线程计算两个列方向输出，双累加器 | TILE=16；Block 16×16 |
| Register2_32 | 同样的 1×2 Register Blocking | TILE=32；Block 32×32 |
| cuBLAS_SGEMM | NVIDIA cuBLAS `cublasSgemm` | 使用 Row-major 与 Column-major 转置视角适配；输入顺序 `B,A` |

原始 Row-major 运算为 `C[M,N]=A[M,K]×B[K,N]`，调用 cuBLAS 时利用 `Cᵀ=BᵀAᵀ`，让同一输出缓冲区在 Column-major 视角下表示 `Cᵀ`，无需显式转置。

## 3. 正确性验证

### 3.1 验证条件

- 矩阵：`64×64×64`、`128×128×128`、`256×256×256`、`512×512×512`、`130×65×70`，按 `M×K×N` 表示。
- 输入类型：FP32；以 `std::mt19937(2026)` 生成 `[-1,1]` 区间的随机值。
- Reference：CPU 读取 FP32 输入，使用 FP64 乘法和累加，最终转换成 FP32 输出。
- 验证公式：`abs(gpu - ref) <= 2e-5 + 1e-4 * abs(ref)`；同时检查输出是否为有限值，并统计不合格元素数。
- 执行：五种实现分别与 CPU Reference 比较，并使用 NaN 初始化输出缓冲区以发现遗漏写回。

### 3.2 验证结果

| Shape (`M×K×N`) | Naive | Tiled16 | Register2_16 | Register2_32 | cuBLAS |
|---|---|---|---|---|---|
| 64×64×64 | PASS | PASS | PASS | PASS | PASS |
| 128×128×128 | PASS | PASS | PASS | PASS | PASS |
| 256×256×256 | PASS | PASS | PASS | PASS | PASS |
| 512×512×512 | PASS | PASS | PASS | PASS | PASS |
| 130×65×70 | PASS | PASS | PASS | PASS | PASS |

**合计 25/25 PASS。** 这说明所有实现在上述测试范围和容差下通过，不构成对任意输入和任意矩阵大小的形式化正确性证明。

### 3.3 数值误差调试结论

早期使用 CPU FP32 累加作为 Reference 时，cuBLAS 在 `512³` 的一个元素上略微超出原定容差。通过对该元素计算 FP64 参考值，确认双方都存在小量浮点舍入误差。后改为 CPU FP64 累加作为 Reference，使用当前项目验证阈值完成 25 项测试。

此处采用的 `atol=2e-5`、`rtol=1e-4` 是为**当前随机输入范围和 K≤512 的测试**选定的阈值，并非适用于任意条件的通用标准。更严格的数值测试可增加多种输入分布、不同量级、不同 K、ULP/归一化误差分析。

## 4. 性能测量方法

### 4.1 CUDA Graph 流程

1. 所有实现使用相同设备 A/B 输入和 CUDA Stream；cuBLAS 的 Stream 与自定义 Kernel 保持一致。
2. 每个 Kernel 先进行 **20 次普通预热调用**。
3. 使用 CUDA Stream Capture，记录 **64 次 GEMM** 形成一个 CUDA Graph，之后实例化为 GraphExec。
4. 正式计时前执行 **3 次 Graph 预热**。
5. 每轮以 CUDA Event 包围 **16 次 Graph Replay**，共 `64×16=1024` 次 GEMM。
6. 单次平均延迟 = Event 记录的每轮总时间 / 1024；完成 **5 轮**并取中位数。
7. 以 `GFLOPS = 2×M×N×K / (latency_ms×10^6)` 计算算术吞吐量。

计时区间**不包括** CUDA 内存申请、Host↔Device 拷贝、CPU Reference、Graph 捕获与实例化；**包含** Graph 回放及其在 GPU 时间线上体现的调度间隔，并不等同于完全剥离所有调度成本的纯指令执行时间。

cuBLAS 使用显式分配的约 4 MiB Workspace。小矩阵结果可能受 GPU 时钟、电源状态、温度、系统调度和测试顺序影响。

## 5. 完整性能结果

以下为**随机 FP32 + FP64 Reference** 最终验收运行数据。每个值是各自程序运行中 5 轮 Graph 测试的中位单次 GEMM 延迟。

### 5.1 延迟（μs；越低越好）

| Shape | Naive | Tiled16 | Register2_16 | Register2_32 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 64³ | 5.092 | **4.986** | 6.041 | 13.155 | 6.955 |
| 128³ | **9.306** | 9.599 | 10.229 | 12.672 | 5.084 |
| 256³ | 60.234 | 42.888 | **39.179** | 50.580 | 15.838 |
| 512³ | 502.394 | 320.928 | 274.126 | **265.238** | 70.168 |
| 130×65×70 | 4.031 | **3.997** | 4.064 | 9.906 | 4.455 |

**加粗**标记的是当次测试中最快的**自定义实现**（非所有实现）。

### 5.2 最快自定义实现与 cuBLAS 的比较

| Shape | 最快自定义实现 | 自定义延迟 (μs) | cuBLAS 延迟 (μs) | cuBLAS 相对自定义的速度比 |
|---|---|---:|---:|---:|
| 64³ | Tiled16 | 4.986 | 6.955 | 0.717× |
| 128³ | Naive | 9.306 | 5.084 | 1.830× |
| 256³ | Register2_16 | 39.179 | 15.838 | 2.474× |
| 512³ | Register2_32 | 265.238 | 70.168 | 3.780× |
| 130×65×70 | Tiled16 | 3.997 | 4.455 | 0.897× |

速度比定义为 `T_最快自定义 / T_cuBLAS`；**大于 1 表示 cuBLAS 更快，小于 1 表示该次测量的自定义实现更快**。

### 5.3 512³ 的吞吐量与自定义优化效果

| 实现 | 延迟 (μs) | 吞吐量 (GFLOPS) |
|---|---:|---:|
| Naive | 502.394 | 534.312 |
| Tiled16 | 320.928 | 836.435 |
| Register2_16 | 274.126 | 979.241 |
| Register2_32 | 265.238 | **1012.055** |
| cuBLAS | 70.168 | 3825.628 |

- 自定义 Register2_32 对 Naive 的加速比：`502.394 / 265.238 ≈ 1.89×`。
- cuBLAS 对最快自定义 Register2_32 的加速比：`265.238 / 70.168 ≈ 3.78×`。
- Register2_32 在这一 Shape 下约为 **1.01 TFLOPS**，cuBLAS 约为 **3.83 TFLOPS**。

## 6. 独立复测与不确定性

小矩阵进行过三次独立进程复测，下面保留关键结果（单位 μs）：

### 6.1 64³

| 实现 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Naive | 5.104 | 2.731 | 2.740 |
| Tiled16 | 5.026 | 2.657 | 2.669 |
| Register2_16 | 6.038 | 3.180 | 3.164 |
| Register2_32 | 13.246 | 6.992 | 6.923 |
| cuBLAS | 7.002 | 3.722 | 3.693 |

三次复测的 `Tiled16/cuBLAS` 延迟比约为 `0.718×`、`0.714×`、`0.723×`（使用项目程序输出的同一速度比定义，均小于 1）。在这三次测量中 Tiled16 都快于 cuBLAS，但绝对延迟有明显变化，不宜推广为所有小矩阵或其他设备上的结论。

### 6.2 130×65×70

| 实现 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Naive | 4.036 | 3.728 | 4.039 |
| Tiled16 | 4.132 | 3.963 | 4.016 |
| Register2_16 | 4.274 | 4.008 | 4.010 |
| Register2_32 | 9.770 | 9.731 | 9.765 |
| cuBLAS | 4.226 | 4.226 | 4.225 |

Naive、Tiled16 与 Register2_16 延迟接近，三轮最优实现会变化；**不将其中任一自定义 Kernel 宣称为该 Shape 的稳定最优解**。

## 7. 优化分析

1. **Naive → Tiled16：** 用 Shared Memory 分块缓存 A、B，减少同一 Tile 内的重复全局内存读取；但增加了数据搬运和块内同步。
2. **Tiled16 → Register2：** 单线程处理两个输出，通过复用同一个 A 元素更新两个累加器，增加寄存器级数据复用。
3. **TILE 的代价：** TILE=32 不一定更快；它使用更大的 Block 和 Shared Memory，线程调度、寄存器压力、实际并行度和边界开销共同影响性能。256³ 最快的是 Register2_16，512³ 最快的是 Register2_32。
4. **小矩阵与大矩阵的权衡：** 小矩阵上同步/调度的相对开销可能较大，Naive 或 Tiled16 更有竞争力；大矩阵才有足够工作量摊薄 Tiling 成本。
5. **与 cuBLAS 的差距：** 512³ 上 cuBLAS 约快 3.78 倍。现有结果证明存在优化空间，但没有直接 Profiling 证据，不能确定 cuBLAS 具体使用了哪些指令和内核策略，也不能仅凭 GFLOPS 认定单一瓶颈。

## 8. 局限性与后续工作

- 测试只覆盖 5 种 Shape、固定随机种子和有限 FP32 输入范围。
- 没有锁定 GPU 时钟、功率状态或温度，也未用 Nsight Compute / Nsight Systems 提供 Kernel 指令、访存与 Occupancy 的逐项证据。
- 小矩阵独立运行的绝对时间有明显波动；报告保留三轮复测原始结果，不把个别微小差距作为普遍结论。
- Graph Benchmark 是固定数据、固定矩阵尺寸、重复执行的吞吐/平均延迟测试，不等于含 Host 提交、H2D/D2H 拷贝的端到端应用延迟。
- 当前 FP64 CPU Reference 最终存储为 FP32；未来可以保持 FP64 原始结果用于更精细的 ULP、归一化误差和累积误差分析。
- 后续优化可在明确瓶颈后尝试更大的每线程计算块、向量化内存访问、软件流水线等，不应在无 Profiling 依据时先验宣称有效。

## 9. 复现方式

在项目根目录执行：

```bash
cmake -S . -B build-v2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build-v2 --target gemm_bench_unified --parallel 4

for shape in '64 64 64' '128 128 128' '256 256 256' '512 512 512' '130 65 70'; do
  read -r M K N <<< "$shape"
  ./build-v2/gemm_bench_unified "$M" "$K" "$N" || exit 1
done
```

运行前建议确保 GPU 没有其他繁重任务，记录当前 `nvidia-smi` 输出。WSL2 中如找不到该命令，可运行 `/usr/lib/wsl/lib/nvidia-smi`。

**版本发布说明：** 报告基于已完成的测试和当前代码行为；最终 Git commit / `v2.0` tag 应在检查仓库文件、排除临时归档、确认 CMake 构建后再创建。
