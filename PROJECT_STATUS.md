# MiniGPU 当前进度

- **更新日期**：2026-10-26。
- **当前版本 / 当前任务**：V5.1–V5.3 实机核心验收通过；`v5.0` Git 标签和推送尚待在用户本地执行并确认。下一阶段 V6 真实模型集成与优化。
- **实际运行环境**：RTX 2060 Laptop 6 GB (SM75)；WSL2 Ubuntu 24.04.5；NVCC CUDA Toolkit 12.8.93；Windows 驱动 591.74（以先前环境记录为准）。
- **已完成源码**：原有 V1–V4 Kernel 与测试；新增 `runtime/device_buffer.h`、`runtime_v5.cu`、`runtime_attention_v5.cu`；修改 `attention_v4.cu` 支持可选 Stream 与测试入口宏；更新 `CMakeLists.txt` 中 V5 目标。
- **已通过正确性测试**：V5.1 DeviceBuffer+Softmax `PASS` (max diff `2.7299506e-09`)；V5.2 三轮不同输入 Fused Attention `3/3 PASS`，max diff 分别 `1.0244548e-08`、`1.3038516e-08`、`1.1408702e-08`；V5.3 Fresh/Reuse 30 对、两方案每次验证 CPU Reference，`ALL PASS`；原 V4 Attention 回归 `ALL PASS`。
- **实测性能**：V5.3 `S=64,D=64,FP32`、5 次预热、30 对交替测试、CPU steady_clock 单轮墙钟耗时：Fresh median `175.673 μs` (min `161.428` max `194.031`)，Reuse median `159.477 μs` (min `137.795` max `237.606`)；中位数之比 `1.102×`，中位数差 `16.196 μs` / 时间减少约 `9.2%`。该结果存在波动，不属于稳定性能承诺。
- **已知问题 / 边界**：首次 Event 测试明显慢，原因未证实；没有独立的显存泄漏检测；DeviceBuffer 不支持移动语义；Runtime 不支持任意 Shape Memory Pool、跨 Stream DAG、CUDA Graph 执行器；当前只验证示例 Attention 工作负载。
- **关键技术决定**：V5 不重写 V4 Kernel；RAII 管理 GPU 和 Stream/Event/Pinned Buffer；cuBLAS 与自定义 Kernel 统一显式 Stream；Fresh 与 Reuse 采用相同计算/传输路径；仅在 CPU 计时中计入 Fresh 的显存分配与释放。
- **已掌握的主要代码/知识**：构造/析构、独占所有权与 `=delete`、GPU Buffer 生命周期、Pinned Host `cudaMemcpyAsync`、Stream 内顺序、Event 标记与同步、CPU/GPU 计时口径区分。
- **待补强知识**：精确理解 `cudaMalloc` 时延与异步分配差异、跨 Stream Event 依赖、CUDA Graph 生命周期、真实模型 PyTorch Extension（在 V6 按需求学习）。
- **Git**：已有 GitHub `v4.0` 已核验；`v5.0` 当前不能宣称已推送。执行 `git status -sb`、`git log -1 --oneline`、`git tag -l v5.0` 核验。
- **下一步最小任务**：将 V5 报告和本状态文件复制入仓库，简要修正 README 旧 V2 状态，提交并推送 `v5.0`；之后开始 V6 真实模型基线。

> 技术报告：[`docs/V5_RUNTIME.md`](docs/V5_RUNTIME.md)。严格区分真实运行日志、源代码结构与待实现功能。
