MiniGPU V2.1 — Naive GEMM 性能基线实验记录
项目：MiniGPU — CUDA Kernel & Lightweight GPU Execution Engine
阶段：V2.1 — GEMM 矩阵乘法性能优化（Naive Baseline）
记录日期：2026-10-22
实验状态：5 组 Shape 正确性 PASS，完成第一轮基础性能测量
对应源码：gemm_bench.cu；构建目标：gemm_bench

1. 实验目的
在 V1 双层 MLP 的 gemm_naive() 基础上，单独建立 FP32 GEMM 正确性与性能基线，供 V2.2 Shared Memory Tiling、V2.4 Register Blocking 和 V2.6 cuBLAS 公平对比。此阶段没有实现 GEMM 性能优化，测量的是 Naive 版本。

2. 实验环境
项目	环境
GPU	NVIDIA GeForce RTX 2060 Laptop GPU，6 GB，Turing / SM 7.5
系统	WSL2，Ubuntu 24.04.5
CUDA Toolkit / NVCC	12.8 / V12.8.93
C++ 编译器	GCC/G++ 13.3.0
CMake	3.28.3
工程路径	/home/damon/projects/minigpu
构建目录	build-v2
数据类型	FP32（float）
矩阵布局	Row-major，无转置
上述环境信息来自项目配置记录；本次终端日志确认 CUDA 12.8.93 编译器和程序可正常构建、运行。

3. GEMM 数学定义与实现
计算：

𝐴
𝑀
×
𝐾
 
𝐵
𝐾
×
𝑁
=
𝐶
𝑀
×
𝑁
,
𝐶
[
𝑖
,
𝑗
]
=
∑
𝑘
=
0
𝐾
−
1
𝐴
[
𝑖
,
𝑘
]
𝐵
[
𝑘
,
𝑗
]
.
A 
M×K
​
 B 
K×N
​
 =C 
M×N
​
 ,C[i,j]= 
k=0
∑
K−1
​
 A[i,k]B[k,j].
Naive Kernel 使用 1 个 CUDA Thread 负责 1 个输出元素，线程索引及一维 Row-major 地址计算如下：

int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= M * N) return;
int row = idx / N;
int col = idx % N;

float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
CPU Reference 用三层循环计算相同的矩阵乘法。GPU 输入通过 Host-to-Device 拷贝传入，输出通过 Device-to-Host 拷贝返回 CPU，并逐元素进行正确性验证。

4. 测试方法与指标
GPU Kernel：gemm_naive（一维 Grid/Block），每个 Block 128 个线程。
Shape 选择：4 组方阵 + 1 组非整除 Shape，修改 M/K/N 后重新编译并运行。
Warmup：20 次 Kernel Launch，预热后同步 GPU。
测量：同一默认 Stream 中先 cudaEventRecord(start)，连续提交 1000 次 GEMM，再 cudaEventRecord(stop) 并等待 stop Event。
平均延迟：avg_ms = total_ms / 1000。
理论运算量近似：FLOPs ≈ 2 × M × N × K。
有效吞吐量：GFLOPS = (2 × M × N × K) / (avg_ms × 1e6)（avg_ms 单位为毫秒）。
数值验证：使用 abs(cpu - gpu) <= atol + rtol × abs(cpu)，其中 atol=1e-5、rtol=1e-4；对非有限值判 FAIL，检查全部 M×N 个输出元素。
计时范围说明： CUDA Event 区间涵盖连续重复的 GEMM 调用，不包含区间外的 cudaMalloc、H2D/D2H 和 CPU Reference 运算。对较短 Kernel，区间可能包含 CPU 提交不及时造成的 GPU 空闲时间，因此“平均 Event 区间耗时”不能严格等同于单次 Kernel 的纯设备执行时间。

5. 实际测试结果（2026-10-22）
以下数据全部摘录自本机实际终端输出，没有使用模拟结果。

Case	Shape（M×K×N）	FLOPs（约）	Total GPU Time (ms)	Avg GEMM Time (ms)	GFLOPS	检查元素	正确性
1	64×64×64	524,288	10.911136	0.010911	48.051	4,096	PASS
2	128×128×128	4,194,304	18.791904	0.018792	223.197	16,384	PASS
3	256×256×256	33,554,432	102.391518	0.102392	327.707	65,536	PASS
4	512×512×512	268,435,456	557.118835	0.557119	481.828	262,144	PASS
5	130×65×70	1,183,000	13.199776	0.013200	89.623	9,100	PASS
所有 Case 的终端均显示 Max Absolute Difference: 0.00000000。这表示在当前打印精度下最大绝对误差为 0，不等价于证明所有 FP32 输出逐位一致，也不代表其他随机输入一定通过。

5.1 计算示例：512×512×512
FLOPs
≈
2
×
512
3
=
268435456
FLOPs≈2×512 
3
 =268435456
GFLOPS
≈
268435456
0.557119
×
10
6
≈
481.828.
GFLOPS≈ 
0.557119×10 
6
 
268435456
​
 ≈481.828.
6. 非整除 Shape 的边界验证
Case 5：M=130, K=65, N=70，因此输出元素数为 M×N=9100。

threads = 128。
blocks = (9100 + 127) / 128 = 72（整数除法）。
总线程位置数：72×128=9216。
多余线程数：9216-9100=116。
多出的 116 个线程因 if (idx >= M*N) return; 提前退出。此次测试 PASS，说明当前 Naive Kernel 在此非整除输出 Shape 下正确处理了边界。此结论不自动代表后续带 __syncthreads() 的 Tiled Kernel 可以直接对越界线程执行 return。

7. 结果分析（观察与推测分开）
实际观察： 在本次测试的 4 组方阵中，矩阵规模由 64³ 增至 512³，平均执行区间增长，同时按公式计算的有效 GFLOPS 从 48.051 增至 481.828。512³ 是本次五组 Shape 中最高的测得吞吐量。

可能解释（尚未 Profiling 证实）： 小矩阵的 GPU 并行工作量相对较少，Kernel 提交、调度等固定开销占比可能较大；随矩阵增大，硬件并行度与计算工作量增加，固定开销占比下降。缓存、访存吞吐、寄存器及 Occupancy 也可能影响结果，但本次没有 Nsight 证据，不能断言具体瓶颈。

不要混淆： 不同 Shape 下 GFLOPS 的增长，不是同一个 Shape 经过 Kernel 优化的加速比；未来计算 Speedup 时，必须比较同设备、同 Shape、同精度、同测量方法下的 Naive 与优化 Kernel。

8. 当前限制与改进方向
目前仅对每个 Shape 记录了一次 Benchmark 运行；尚未开展多轮独立运行、均值/中位数/方差统计和 GPU 频率、功耗、温度控制。
Shape 通过修改源码并重新编译切换，后续考虑支持命令行参数或统一测试驱动，降低人为误差。
目前 Event 测量对象为连续提交的 1000 次 GEMM；V3 可借助 Nsight Systems / Compute（视 WSL2 与 GPU 权限能力）进一步分析启动空隙、访存、Occupancy 等。
尚未测试 cuBLAS，尚未实现 Shared Memory Tiling / Register Blocking，尚无优化加速比。
数值测试使用了可重复的周期性输入；后续可增加随机数据、较大幅度数据和更多非整除 Shape，以提高验证覆盖度。
9. 关键源码与构建命令
本次源码：gemm_bench.cu；构建配置在 CMakeLists.txt 中包含 add_executable(gemm_bench gemm_bench.cu)。

cd ~/projects/minigpu
cmake -S . -B build-v2
cmake --build build-v2 --target gemm_bench --parallel 4
./build-v2/gemm_bench
构建和运行成功并打印 Test: PASS 后，才能认定对应 Shape 的正确性验证完成。build-v2/ 是生成目录，已经由 .gitignore 的 /build-*/ 规则忽略，不应提交 Git。

10. V2.1 阶段结论及向 V2.2 交接
已完成： 独立 Naive GEMM、CPU/GPU 正确性比较、H2D/D2H、CUDA Event 预热与计时、GFLOPS 计算、5 组 Shape 实测，其中包括非整除边界情况；5 组均 PASS。

下一阶段 V2.2： 基于此 Naive 基线实现 gemm_tiled()，使用 __shared__ 和 Block 内同步实现 Tile 数据复用；优先保证多 Shape 正确性，再使用相同的 Bench 设置比较相同 Shape 的延迟、GFLOPS 和 Speedup。

基线是“当前实测状态”，不是理论峰值或已证实的性能瓶颈。V2.2 的优化收益须等新 Kernel 实际测试后记录，不提前承诺加速倍数。
