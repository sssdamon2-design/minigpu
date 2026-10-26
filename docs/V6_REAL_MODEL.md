# MiniGPU V6 完整技术复习手册：真实 Qwen 推理、Profiling、GEMM 与 CUDA Extension

> **文档版本：V2.0（详细复习版）**  
> **整理日期：2026-10-27**  
> **项目仓库：** `https://github.com/sssdamon2-design/minigpu`  
> **运行主机：** RTX 2060 Laptop 6GB / WSL2 Ubuntu / CUDA 12.8  
> **定位：** AI Systems / GPU Kernel / 推理优化方向的**学习型、可复现实验工程**。  
> **数据边界：** 本报告的性能、误差、运行成功记录取自用户在 V6 过程中实际提供的终端日志；代码解释基于本次会话提供的 V6 源码及已有 V2/V4/V5 工程。计算推导、合理假设和未来可做的工作均明确区别于实测事实。**不把自定义 GEMM 集成成功写成性能加速成功。**

---

## 0. 建议怎样使用这份文档

本报告不是简单的进度总结，而是给未来复习、重建关键源码、模拟面试使用的资料。建议按以下顺序阅读：

1. **快速回忆（第 1～2 章）：** 5 分钟记住做了什么、模型有多大、延迟与误差是多少。
2. **原理理解（第 3～7 章）：** 学会说清 Transformer Forward、FP16、Profiler、真实 GEMM Shape、CUDA Extension。
3. **工程判断（第 8～11 章）：** 明白怎样做公平 Benchmark、怎样进行完整模型 A/B、为什么正确而更慢。
4. **自己动手复现（第 12～13 章）：** 选择必要命令，读关键源码；不用重复下载安装模型、重做 V1～V5。
5. **面试验收（第 14～16 章）：** 先独立回答问题，再看参考要点；如答不上来，需要回到对应源码。

### 0.1 V6 一句话总览

> 在 RTX 2060 上离线运行 **Qwen2.5-0.5B-Instruct**，得到 FP32/FP16 推理基线；通过 PyTorch Profiler 定位 **FP32 GEMM** 为主要 GPU 热点；从第一层 MLP 的 `gate_proj` 提取真实输入与权重；将 V2 自定义 Tiled16 GEMM 封装为 PyTorch C++/CUDA Extension，验证单算子和完整模型 Logits 正确性；最终确认**自定义 GEMM 比成熟 PyTorch/cuBLAS 路径更慢**，如实保留负结果。

### 0.2 最重要的实测结果——必须能脱口而出

| 实验 | 必记结果 | 正确结论 |
|---|---|---|
| FP32，S=32 | **40.095 ms**，**1.900 GiB** | 完整序列 Forward 基线 |
| FP32，S=128 | **46.921 ms**，**2.049 GiB** | V6 主要对照 Shape |
| FP16，S=32 | **39.793 ms**，**0.969 GiB** | 显存明显下降，延迟改善约 **0.75%** |
| FP16，S=128 | **40.539 ms**，**1.065 GiB** | 延迟改善约 **13.60%**，内存下降约 **48.02%** |
| FP16/FP32 Logits，S=32 | Top-1 一致率 **93.75%** | 32 个位置中有 **2 个位置不同** |
| FP16/FP32 Logits，S=128 | Top-1 一致率 **100%** | 仅证明该随机输入上的位置预测一致 |
| FP32 Profiler，S=128 | `volta_sgemm_128x64_tn` **29.626 ms / 86.74%** | 计算热点是 **GEMM**，统计重叠不能相加 |
| 真实单 Linear GEMM | MiniGPU **1496.290 μs** vs PyTorch Linear **286.024 μs** | 自定义 GEMM **慢 5.23 倍** |
| 完整模型 A/B | Original **39.459 ms** vs Custom **40.711 ms** | 自定义路径**变慢 3.17%**；模型集成正确 |
| 完整模型数值误差 | Max Abs Diff **4.0531e-05**、Top-1 **100%** | 当前测试下完整输出正确性检查 PASS |

**极易混淆的三句话：**

- `V6.4 Model Integration: PASS` ≠ `V6.4 性能加速 PASS`。
- `Output finite: True` ≠ `FP16 完全无精度损失`。
- `CUDA Event 时间` ≠ `纯 GPU 指令时间`，也 ≠ `用户聊天端到端耗时`。

---

## 1. V6 的目标、与 V1～V5 的关系及项目边界

### 1.1 为什么需要 V6？

V1～V5 已经实现了很多基础能力，但大量测试采用随机生成的矩阵或独立算子。V6 需要解决：**这些 CUDA 知识能否用于一个真实预训练模型？优化决策有没有真实的性能证据？**

| 阶段 | 关键工程能力 | V6 中如何复用 |
|---|---|---|
| V1 | CUDA 线程/Block、Naive GEMM、MLP | 理解 GEMM 计算与 GPU 执行 |
| V2 | FP32 GEMM、Shared Memory Tiling、Register Blocking、cuBLAS Benchmark | **复用 Tiled16**，与模型内真实 GEMM 比较 |
| V3 | Nsight 分析、分析原因与实验验证 | 先 Profiling，再决定优化对象 |
| V4 | Stable Softmax、RMSNorm、Attention、Fusion | 理解 Transformer 算子；本次**没有强行集成 RMSNorm** |
| V5 | DeviceBuffer RAII、Stream、Event、Pinned Buffer、Buffer Reuse | 理解 Extension Stream 正确性和资源生命周期 |
| V6 | 真实模型 → 真实热点 → 真实 Tensor → 自定义 Kernel → 模型 A/B | 形成数据驱动的系统级工程闭环 |

### 1.2 V6 真正验收的事项

- [x] 实机、真实预训练权重、离线模型加载。
- [x] FP32 和 FP16 的完整序列 Forward 延迟、显存基线。
- [x] 同输入 FP32/FP16 Logits 差异。
- [x] 使用 Profiler 定位算子级和底层 Kernel 级热点。
- [x] 获取真实 `gate_proj` 的输入 Shape、权重布局。
- [x] 把自定义 FP32 GEMM 编译成 PyTorch C++/CUDA Extension。
- [x] 独立 GEMM 与 PyTorch 结果做数值一致性验证。
- [x] GEMM 三路径性能比较。
- [x] 替换完整模型**一个** Linear，验证 Logits 和整体 Forward A/B。
- [ ] 自定义 GEMM 的实际模型级加速——**没有实现**，实测反而变慢。
- [ ] 针对这个 Shape 的 Nsight Compute 微架构归因——**尚未完成**。
- [ ] 独立手写 Tensor Core / WMMA GEMM——**尚未完成**。
- [ ] 真实自然语言生成质量、KV Cache Decode、长期稳定性、吞吐压测——**不在本次范围**。

### 1.3 实际实验环境

| 项目 | 确认信息 |
|---|---|
| 操作系统 | WSL2 Ubuntu 24.04.5 LTS |
| GPU | NVIDIA GeForce RTX 2060 Laptop，6GB 显存 |
| Compute Capability | `(7, 5)`，即 `sm_75` |
| CUDA Toolkit / NVCC | CUDA 12.8 / `/usr/local/cuda-12.8/bin/nvcc` |
| PyTorch | `2.9.1+cu128` |
| Transformers | `4.57.1` |
| Python | WSL 系统 Python 3.12 |
| CUDA Extension 构建工具 | Ninja，`/usr/bin/ninja` |
| 预训练模型路径 | `/home/damon/models/Qwen2.5-0.5B-Instruct` |
| 工程路径 | `/home/damon/projects/minigpu` |
| 模型下载方式 | 台式机 Windows CMD 使用 ModelScope 下载，然后 SCP 到 WSL |

**版本常识：** `nvidia-smi` 显示的 CUDA Version 指的是驱动支持能力，不代表 Python 所用 CUDA Runtime 和 `nvcc` 版本一定相同。PyTorch 自带的 `cu128` 依赖与 Toolkit 的关系，不能简单理解为“安装 PyTorch 就必须重装显卡驱动”。

---

## 2. 模型下载、离线部署、参数结构

### 2.1 为什么台式机下载、笔记本推理？

最初 Hugging Face 在 Windows 上出现 `httpx2.ConnectTimeout: [WinError 10060]`。这属于**网络访问超时**，而不是 PyTorch/CUDA 初始化错误；`tensorflow-gpu` 与 `protobuf` 的版本冲突警告也不是此次模型下载失败的直接原因。

台式机通过 ModelScope 完成下载，获得 11 个模型文件，核心权重 `model.safetensors = 988,097,824 bytes`。通过 SCP 成功传至 WSL。注意：**模型文件不是 CUDA 可执行代码**，Windows 下载的模型权重可以被 Linux 下的 Transformers 加载。

台式机 CMD 下载命令（已实测成功）：

```bat
cd /d C:\Users\53009\Downloads
py -m pip install -U modelscope
py -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='Qwen2.5-0.5B-Instruct')"
```

台式机传到笔记本 WSL（`MiniGPU-WSL` 应替换为实际可用的 SSH 别名）：

```bat
scp -r C:\Users\53009\Downloads\Qwen2.5-0.5B-Instruct MiniGPU-WSL:/home/damon/models/
```

WSL 中验证（只需必要时重查，不需重复下载）：

```bash
ls -lh ~/models/Qwen2.5-0.5B-Instruct/model.safetensors
```

应能看到约 942 MiB 的权重文件（`988,097,824 bytes`），以及 `config.json`、Tokenizer 文件。

### 2.2 Transformers 从本地加载，如何保证离线？

```python
MODEL_PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to("cuda").eval()
```

运行前额外指定：

```bash
HF_HUB_OFFLINE=1 python3 v6/baseline.py
```

- `from_pretrained(path)`：根据配置文件创建模型结构，并从本地 `safetensors` 加载训练好的权重。
- `dtype=torch.float32`：决定加载时采用的参数数据类型；原始权重文件不是必须就是 FP32。
- `.to("cuda")`：将参数转移到 RTX 2060。
- `.eval()`：进入模型推理模式；它不等价于禁止所有梯度计算。
- `torch.inference_mode()`：在执行区关闭 Autograd 相关开销；与 `.eval()` 作用不同。
- `local_files_only=True`、`HF_HUB_OFFLINE=1`：避免运行测试时再访问 Hugging Face。

### 2.3 模型架构数据

真实程序输出：

```text
Parameters:      494,032,768
Layers:          24
Hidden Size:     896
Attention Heads: 14
KV Heads:        2
Vocabulary Size: 151936（由输出 Logits 最后一维确认）
```

Qwen2.5-0.5B 使用分组查询注意力（GQA）；一个直观理解是：Query 头数大于 K/V 头数，从而减少 K/V 头的数量。本模型记录为 14 个 Q Head、2 个 KV Head。由 `896/14 = 64` 可推导 Head Dimension 为 64，且每组 KV Head 对应 7 个 Query Head。**这些是架构参数推导，不代表 V6 已测试 Decode KV Cache。**

模型由多个 Transformer Block 组成；本次重点涉及：

```text
input_ids [1,S]
     │
     ▼
Token Embedding [1,S,896]
     │
     ▼
24 × Transformer Decoder Blocks
  ├── RMSNorm
  ├── 自回归自注意力（含 Q/K/V 投影）
  ├── Residual
  ├── RMSNorm
  └── MLP（gate_proj / up_proj / down_proj）
     │
     ▼
Final Norm / LM Head
     │
     ▼
logits [1,S,151936]
```

**重要边界：** 上图是模型的概念数据流；实际层内 Forward 还有 RoPE、Mask、残差及具体算子执行分支等，不能把上述简化示意当作完整源码实现。

### 2.4 `use_cache=False` 到底测的是什么？

```python
logits = model(input_ids=input_ids, use_cache=False).logits
```

- 输入一次整个序列；模型对序列中的位置执行 Decoder Forward。
- 这里不是循环 `generate()` 生成新 Token。
- `use_cache=False` 不构建用于后续 Decode 复用的常规 KV Cache；因此**无法从这里推出每 Token Decode 延迟或 tokens/s 吞吐**。
- 测试只关注给定 Shape 的 GPU Forward，不包含文本 Tokenizer、采样策略、流式输出、H2D 输入构造及模型加载时间。
- 模型本身是自回归模型，**不能把原先 V4 不带 Mask 的单头 Attention 直接等同于 Qwen 的 Attention**。

---

## 3. V6.1：FP32 真实模型 Forward 基线——逐段读源码

对应：`v6/baseline.py`，FP16 版本：`v6/baseline_fp16.py`。

### 3.1 配置项

```python
MODEL_ID = "/home/damon/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float32
WARMUP = 5
REPEAT = 20
SEQ_LENGTHS = [32, 128]
```

固定 Batch=1；改变 Sequence Length（以下简称 `S`），保持硬件、精度和测试方法一致。`S=32/128` 是选定实验点，不代表模型只能处理这些长度。

### 3.2 随机 Token 与 Tensor Shape

```python
input_ids = torch.randint(
    100, 10000,
    (1, seq_len),
    device=DEVICE
)
```

- `torch.randint(low,high,size)`：产生整数 Tensor（上界不包含），这里代表合法的 Token ID 采样区间。
- `size=(1,seq_len)`：`[Batch, Sequence]`。
- `device="cuda"`：输入直接放在 GPU。
- **为什么不用真实句子？** 本次先验证固定 Shape 的推理性能，减少 Tokenization 的干扰。真实文本质量另做测试才有意义。

输出：

```text
S=32 : logits [1,  32, 151936]
S=128: logits [1, 128, 151936]
```

公式：

\[
\text{Logits.shape}=[B,S,V],\quad B=1,\quad V=151936.
\]

`logits` 是未经过最终采样的词表分数，不是自然语言字符串，也不是最终生成 Token 列表。`argmax(-1)` 可以得到每个位置最高分 Token 的 ID，但不等同于完整聊天生成流程。

### 3.3 为什么需要 Warmup？

```python
with torch.inference_mode():
    for _ in range(5):
        logits = model(input_ids=input_ids, use_cache=False).logits
    torch.cuda.synchronize()
```

首次执行可能包含库初始化、内存分配器行为、GPU 频率变化等非稳态因素。预热的目的在于减少这些因素对正式测量的影响，而不是“预热以后 GPU 就一定完全稳定”。`torch.cuda.synchronize()` 保证此前提交的 GPU 工作已经完成。

### 3.4 CUDA Event 计时——必须讲清同步关系

```python
samples = []
for _ in range(20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    logits = model(input_ids=input_ids, use_cache=False).logits
    end.record()
    end.synchronize()
    samples.append(start.elapsed_time(end))
```

- `start.record()`：在当前 CUDA Stream 中加入开始事件。
- GPU Forward 的 Kernel 按相应 Stream / 依赖关系执行。
- `end.record()`：在 GPU Stream 中加入结束事件。
- `end.synchronize()`：等待 GPU 完成到该事件为止的工作。
- `start.elapsed_time(end)`：获取两事件之间的设备时间线间隔，单位 ms。
- `statistics.median(samples)`：取 20 次测量的中位数，减少少数异常值影响。

**容易答错：** CUDA Kernel 调用通常是异步的；若没有合理同步就读 CPU 墙钟或输出，可能错误地认为 GPU 已运行结束。另一方面，CUDA Event 区间可能因 CPU 提交节奏而包含设备空闲，不能无条件称为“纯 Kernel 指令耗时”。

### 3.5 显存统计和两个不同的概念

```python
torch.cuda.reset_peak_memory_stats()
# Warmup + Benchmark ...
peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
```

- `memory_allocated`：PyTorch CUDA 分配器当前已分配给 Tensor 的显存。
- `max_memory_allocated`：自上次重置后出现过的最高**已分配**值。
- 不等于 `nvidia-smi` 进程总显存使用，也不包括所有非 PyTorch 外部显存分配、CUDA Context 等。
- 显存峰值可能受到模型权重、临时激活、输出 Logits、分配器行为影响。

**模型权重大小推导：**

\[
494,032,768 \times 4 \text{ bytes} \approx 1.840\ \text{GiB（FP32 参数本身）}.
\]

\[
494,032,768 \times 2 \text{ bytes} \approx 0.920\ \text{GiB（FP16 参数本身）}.
\]

这能解释为何实际 FP32/FP16 峰值显存相差较大，但**不应把峰值与理论权重大小之差直接当成完整激活占用**，因为峰值统计的具体内存组成并未分项测量。

### 3.6 FP32 实测数据

| Shape | Median | Min | Max | Peak allocated | Finite |
|---|---:|---:|---:|---:|---|
| `[1,32]` | **40.095 ms** | 39.042 ms | 61.552 ms | **1.900 GiB** | True |
| `[1,128]` | **46.921 ms** | 44.600 ms | 54.558 ms | **2.049 GiB** | True |

这些是原始基线记录，必须保留为独立实验结果。之后完整模型 A/B 的 Original 变为 39.459 ms，属于另一轮、另一测试脚本的配对对照，不代表第一次测量错误。

---

## 4. V6.3.1：FP16 推理实验与数值误差

### 4.1 FP16 为何可能更快？为何不保证？

FP32 一个浮点数占 **4 字节**；FP16 占 **2 字节**。对于相同参数量，模型权重理论存储量减半，内存带宽需求也可能减小。RTX 2060 的 Turing 架构具备 FP16 Tensor Core 路径，适当 Shape 和库实现可能得到加速。

但是实际 Forward 包含 GEMM、归约、逐元素算子、Kernel Launch、调度以及各种数据变换。即使某些 GEMM 更快，**整个模型不一定同比例变快**；本实验也没有直接验证 cuBLAS 是否调用了 Tensor Core Kernel。

### 4.2 控制变量

`baseline_fp16.py` 是 FP32 基线拷贝，唯一有意更改的核心配置：

```python
DTYPE = torch.float16
```

其他保持 `B=1`、`S=32/128`、`WARMUP=5`、`REPEAT=20`、同一模型结构和相同测试口径。

### 4.3 FP32/FP16 结果逐项比较

| 量 | FP32 S32 | FP16 S32 | FP32 S128 | FP16 S128 |
|---|---:|---:|---:|---:|
| Median Latency | 40.095 ms | 39.793 ms | 46.921 ms | 40.539 ms |
| Min Latency | 39.042 ms | 38.611 ms | 44.600 ms | 38.988 ms |
| Max Latency | 61.552 ms | 42.923 ms | 54.558 ms | 43.412 ms |
| Peak allocated | 1.900 GiB | 0.969 GiB | 2.049 GiB | 1.065 GiB |
| Output finite | True | True | True | True |

**S=128 延迟降低计算：**

\[
\frac{46.921-40.539}{46.921}\times100\% \approx \boxed{13.60\%};\qquad
\text{Speedup}=\frac{46.921}{40.539}\approx\boxed{1.157\times}.
\]

**S=128 峰值已分配显存降低计算：**

\[
\frac{2.049-1.065}{2.049}\times100\%\approx\boxed{48.02\%}.
\]

**S=32 延迟降低：**

\[
\frac{40.095-39.793}{40.095}\times100\%\approx\boxed{0.75\%}.
\]

**正确表述：** 在本次 RTX 2060、FP32/FP16 独立运行的测试中，FP16 显著降低峰值已分配显存；S=128 延迟有约 13.6% 改善，而 S=32 改善很小。不同进程数据不是严格同进程配对 A/B，不能推出所有 Shape 或所有设备都稳定加速。

### 4.4 为什么还要比较 Logits？

`torch.isfinite(logits).all()` 只说明没有 NaN / ±Inf，不能证明 FP16 仍保持与 FP32 接近的预测排序。

`check_fp16.py` 先加载 FP32 模型、针对固定随机 Token IDs 生成参考 Logits，再释放模型并加载 FP16，用**相同的 Token IDs**执行推理。输出统一转为 `float32` 后计算：

```python
diff = (output - reference).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
top1_match = (
    output.argmax(dim=-1) == reference.argmax(dim=-1)
).float().mean().item()
```

三个指标含义不同：

- `max_diff`：全体 Logits 中最大的**单元素绝对误差**。
- `mean_diff`：所有 Logits 的平均绝对误差；平均值可能掩盖少数位置大偏差。
- `top1_match`：每个输入位置预测分数最高的 Token ID 是否相同；不衡量整个概率分布，也不等于文本生成质量。

### 4.5 实测数值一致性

| S | Max Abs Diff | Mean Abs Diff | Top-1 Agreement | Finite |
|---|---:|---:|---:|---|
| 32 | **0.524947** | **0.012888** | **93.75%** | True |
| 128 | **0.314798** | **0.006356** | **100.00%** | True |

- `S=32`：`32 × (1-0.9375) = 2`，有 **2 个位置** Top-1 不一致。
- `S=128`：本次 128 个位置 Top-1 一致。
- **不能推出** FP16 在自然语言生成任务上“效果不变”；本次输入是随机 Token IDs，没有困惑度、真实问题或生成质量评价。
- Max Logit 误差是分数尺度，不能直接翻译成“答错了百分之多少”。

### 4.6 面试的正确回答

> 我先用真实 Qwen 做 FP32 基线，再保持模型和 Shape 基本一致，加载 FP16 对照。在 S=128 上 Forward 中位数从 46.921 ms 降到 40.539 ms，峰值已分配显存从 2.049 GiB 降到 1.065 GiB。但 S=32 只有约 0.75% 改善；Logits 在 S=32 有 2/32 个位置 Top-1 不同。因此我只声称当前条件下的延迟与显存收益，不声称模型无精度损失或已验证 Tensor Core 指令使用。

---

## 5. V6.2：Profiler 定位 GEMM 热点——怎样正确读表

对应文件：`v6/model_profiler_v6.py`。

### 5.1 为什么先 Profiling，不先选 RMSNorm？

AI Systems 的优化流程应遵循：

```text
确定性能指标 → 运行真实工作负载 → Profiling → 判断热点
      → 选择候选优化 → 检查正确性 → 重新测量整体收益
```

V4 有 RMSNorm、Softmax 并不意味着 V6 必须优先替换这两个 Kernel。**方便替换**不等于**性能瓶颈**。本次 S=128 的数据明确显示，主要 GPU 计算时间集中在 GEMM。

### 5.2 关键 Profiler 代码

```python
with torch.inference_mode():
    for _ in range(5):
        model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        output = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_device_time_total", row_limit=25
))
```

`record_shapes=True` 可以捕获 Shape 元数据，但**原先仅输出按名称聚合的 Top 25 表，不等于你已保存每个 GEMM 的完整输入 Shape 或执行调用栈**。以后要定位具体 Layer，可以增加调用栈/模块范围分析；本阶段不用重复大量 Profiling。

### 5.3 真实 Top CUDA 数据——保留原始统计口径

| Profiler 名称 | # Calls | Self CUDA | Self CUDA % | 说明 |
|---|---:|---:|---:|---|
| `volta_sgemm_128x64_tn` | 121 | **29.626 ms** | **86.74%** | 底层 GEMM Kernel |
| `aten::mm` | 97 | **28.356 ms** | **83.02%** | PyTorch 矩阵乘法算子 |
| `aten::addmm` | 72 | **2.074 ms** | **6.07%** | 矩阵乘与加法的算子路径 |
| `aten::mul` | 220 | **1.288 ms** | **3.77%** | 逐元素乘法，可来自多个模块 |
| `aten::_efficient_attention_forward` | 24 | **0.683388 ms** | **2.00%** | 当前模型 Attention 计算路径 |
| `aten::silu` | 24 | **0.416446 ms** | **1.22%** | MLP 激活之一 |
| `aten::mean` | 49 | **0.260119 ms** | **0.76%** | 归约操作之一 |

汇总：

```text
Self CPU time total:  50.441 ms
Self CUDA time total: 34.156 ms
```

### 5.4 很关键的统计重叠问题

Profiler 同时列出了上层 ATen Operator 和底层 CUDA Kernel。两层级的时间通常存在对应关系，例如 `aten::mm` 调用了 GEMM Kernel。**不要把 28.356 ms 与 29.626 ms 相加**，也不要把两个百分比相加。

`Self CUDA time total=34.156ms` 不是 V6.1 的 `46.921ms` 的“正确答案”，因为 Profiling 的统计范围、Profiler 引入的开销、Host 调度、异步执行及测试时刻都不同。CPU Self Total 与 CUDA Self Total 也不能直接相加得到模型 Forward 端到端时延。

### 5.5 这里到底证明了什么？

**可以证明/有充分依据：**

1. 此次 FP32 S=128 的 GPU 计算热点主要由 SGEMM 主导。
2. Qwen 的 Attention 实际出现过 `fmha_cutlassF_f32_aligned_64x64_rf_sm75`，说明不能未经验证就将 V4 不带 Mask 的 Attention 视为更优替代。
3. 单独 RMSNorm 即使大幅加速，对总体性能的作用也需要结合占比评估，不能先下结论。

**还不能证明：**

1. `volta_sgemm_*` 慢是因为特定 Warp Stall、DRAM 饱和、Occupancy 低等——尚无针对该 Shape 的 Nsight Compute 数据。
2. FP16 的加速已使用 Tensor Core MMA 指令——没有对应 Kernel 或指令级证据。
3. 86.74% 是未插桩模型完整墙钟时间中精确的 GEMM 占比。
4. 只要重写 GEMM 就一定能比 cuBLAS 更快。

### 5.6 为什么不立即优化 GEMM 全部位置？

真实模型中的 GEMM Shape 不同，包含小 M 大 N、不同权重 stride、Attention 投影和 MLP 等。用一个通用 Tiled16 全部替换可能明显变慢，而且新代码还要处理 bias、dtype、Shape、布局、梯度与兼容性。我们先对一个真实 Layer 做准确的局部实验，再决定是否值得扩展。

---

## 6. V6.3.2：从 Qwen 中提取真实 GEMM 输入

对应文件：`v6/inspect_real_gemm.py`。

### 6.1 目标函数：第一层 MLP `gate_proj`

在 PyTorch 中，`nn.Linear` 的核心计算为：

\[
Y=XW^T+b.
\]

本次选择的 Qwen `gate_proj` 在实际结构中为不带 bias 的线性层；本次自定义扩展仅实现 `Y=XW^T`，不能泛化为自动支持所有 `nn.Linear` 的完整替换。

以 Sequence Length 128、Batch 1 为例：

```text
X        [128, 896]    = M × K
W        [4864, 896]   = N × K
W.T      [896, 4864]   = K × N
Y        [128, 4864]   = M × N
M=128, K=896, N=4864
```

### 6.2 Hook 是如何捕获真实激活的？

```python
linear = model.model.layers[0].mlp.gate_proj
captured = {}

def capture_input(module, args):
    captured["x"] = args[0].detach()

handle = linear.register_forward_pre_hook(capture_input)

with torch.inference_mode():
    model(input_ids=input_ids, use_cache=False)

handle.remove()
X = captured["x"].reshape(-1, model.config.hidden_size)
W = linear.weight.detach()
```

逐行理解：

1. `model.model.layers[0]`：取第一个 Transformer Block（Python 下标从 0 开始）。
2. `.mlp.gate_proj`：取该层 MLP 内的线性投影。
3. `captured = {}`：用于临时存储 Tensor 引用的字典。
4. `register_forward_pre_hook`：在这个模块 `forward()` 执行**之前**调用回调，因此拿到的是进入 Linear 的真实 `X`。
5. `args[0]`：取该模块的第一个位置参数（输入激活）。
6. `.detach()`：返回不参与梯度链追踪的 Tensor 视图/引用；不会自动搬到 CPU。
7. `handle.remove()`：取消 Hook，避免后续 Forward 重复捕获及副作用。
8. `reshape(-1,896)`：把 `[1,128,896]` 变成 GEMM 需要的二维 `[128,896]`；`-1` 让 PyTorch 自动推算该维度。

**为什么不是直接随机生成 `[128,896]`？** 因为这样只能验证 Shape 匹配，无法证明输入确实来自真实 Qwen Forward。Hook 提取能建立真实模型数据链路。

### 6.3 实际运行输出

```text
X Shape:           (128, 896)
W Shape:           (4864, 896)
W.T Shape:         (896, 4864)
Y Shape:           (128, 4864)
X contiguous:      True
W contiguous:      True
W.T contiguous:    False
DType:             torch.float32
V6.3.2 Real GEMM Extraction: PASS
```

**这里最关键的不是 Shape，而是 `W.T contiguous=False`。**

---

## 7. Tensor Layout / Stride：为什么 `W.T` 不能直接传给 V2 GEMM？

### 7.1 先理解连续行主序

假设一个连续矩阵 `A` 的 Shape 为 `[M,K]`：

```cpp
A[row * K + col]
```

这是连续行主序（row-major）的典型索引方式。其逻辑 Stride 为 `[K,1]`，即：行号加一跳过 K 个元素，列号加一跳过 1 个元素。

二维 Tensor 的通用线性地址规则：

\[
\operatorname{offset}(i,j)=i\times\operatorname{stride}_0+j\times\operatorname{stride}_1.
\]

### 7.2 为什么 `.T` 只改变视图？

原始权重：

```text
W.shape    = [4864,896]
W.stride() ≈ (896,1)      # 连续行主序
```

转置视图：

```text
W.T.shape    = [896,4864]
W.T.stride() ≈ (1,896)   # 转置视图，通常不是连续行主序
```

上述 Stride 是从本次 `W contiguous=True` 的行主序布局可推导的预期值，**实验日志实际打印的是 Contiguous 状态，未单独打印 `.stride()`**。下次用 `print(W.stride(), W.T.stride())` 即可验证。

`.T` 通常不会复制 16.625 MiB 的权重内容，而是改变“下标怎样映射到底层 Storage”。一个指针配合 Shape 还不够，必须知道 Stride 或先调整物理布局。

### 7.3 为什么原 V2 算法会算错？

V2 Kernel 对 B 使用：

```cpp
B[b_row * N + col]
```

这个公式假定 B 的 Shape `[K,N]` 且 Stride 为 `[N,1]`。如果直接把非连续 `W.T` 的 `data_ptr()` 当成这样的 B，Kernel 的寻址就会按错误布局解释内存，**即使程序没有越界，也可能产生错误结果**。

正确办法至少有两种：

- **方案 A：预转置一次权重。** `W_packed = W.T.contiguous()`，复制成 `[896,4864]` 的连续布局，然后继续复用 V2 Tiled GEMM。
- **方案 B：改写 Kernel 的 B 索引。** 按原始 `W[N,K]` 连续布局直接读取 `W[col * K + k]`，避免额外权重复制，但需要针对新的加载布局重新分析 coalescing / Shared Memory 数据摆放。

本阶段采用**方案 A**，因为最小化新增算法变化，先验证真实模型集成链路。它不一定是最终更优的部署方案。

### 7.4 `contiguous()` 的代价必须解释

```python
W_packed = W.T.contiguous()
```

- **初始化时：** 需要复制一次权重并额外保留一份布局后的 Tensor。
- **稳态推理时：** 如果 `W_packed` 被复用，不需要每次重新转置。
- **实验报告：** 稳态 GEMM 延迟可以排除初始化转置，但必须注明有一次性预处理和额外显存。
- **不公平的做法：** 在每次 Forward 内调用 `.T.contiguous()`，却只给 CUDA Kernel 计时，不给转置与分配计时。

本次权重元素数量：

\[
896\times4864=4,358,144\ \text{elements}.
\]

FP32 预转置副本理论大小：

\[
4,358,144\times4=17,432,576\ \text{bytes}=\boxed{16.625\ \text{MiB}}.
\]

这只是额外权重数据的理论字节数，不含分配器元数据、临时工作空间，也不是该操作的实测耗时。

### 7.5 本节面试高频题

**问：为什么 `.T` 以后 Shape 正确却还需要关注 Stride？**

答：Shape 只定义逻辑维度，Stride 才定义数据在底层 Storage 中的寻址关系。自定义 CUDA Kernel 若假定行主序连续数据，就不能忽略非连续转置视图。可以预先打包权重，或者让 Kernel 显式支持原始布局。

---

## 8. CUDA Tiled16：从 V2 算法走到真实 Qwen GEMM

对应文件：`v6/gemm_tiled_v6.cu`。本次复用已有 V2 Shared Memory Tiling，**不是重新发明一套更快的 GEMM**。

### 8.1 计算公式与工作量

\[
C_{m,n}=\sum_{k=0}^{K-1}A_{m,k}B_{k,n},\quad
A:[128,896],\ B:[896,4864],\ C:[128,4864].
\]

采用常见的 GEMM FLOPs 估算，每次乘加算 2 FLOPs：

\[
2MKN = 2\times128\times896\times4864
=\boxed{1,115,684,864\ \text{FLOPs}} \approx 1.116\ \text{GFLOPs}.
\]

只估算矩阵乘法浮点运算，不计 CUDA 线程索引、数据加载、转置预处理或 Python 调用。

### 8.2 核心源码（需真正看懂）

```cpp
__global__ void gemm_tiled16(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    constexpr int TILE = 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 8.3 一张“人脑执行图”

```text
整个 GEMM C[128,4864]
    │
    ├── Grid:  (ceil(4864/16), ceil(128/16)) = (304,8)
    │           共 2432 个 Block
    │
    └── 每个 Block: (16,16) = 256 threads
           │
           ├── 负责一个 C 的 16×16 Tile
           ├── K 方向循环 896/16 = 56 次
           │      ├── 从 Global Memory 读 As 16×16
           │      ├── 从 Global Memory 读 Bs 16×16
           │      ├── __syncthreads()
           │      ├── 计算本线程局部 sum
           │      └── __syncthreads()
           └── 每个线程写出一个 C[row,col]
```

为何需要 Shared Memory：同一 Tile 中，相邻输出会重复使用 A 的行片段和 B 的列片段。把数据读入 Block 内共享区域，理论上减少对 Global Memory 的重复请求。实际效果仍取决于硬件缓存、访存合并和算子 Shape。

### 8.4 两次 `__syncthreads()` 的意义

1. **加载后同步：** 保证本轮各线程向 `As/Bs` 写入完毕，其他线程读取时数据已经可见。
2. **计算后同步：** 保证本轮对 Shared Memory 的读取完成，下一轮不会过早覆盖 `As/Bs` 中仍被使用的数据。

由于每个线程负责不同的 C 元素，最终 C 写回没有多个线程竞争同一地址；Block 内部分结果也只在各自寄存器 `sum` 中维护。

### 8.5 边界保护

```cpp
(row < M && a_col < K) ? A[...] : 0.0f
(b_row < K && col < N) ? B[...] : 0.0f
```

防止矩阵维度不能整除 TILE 时越界，并让无效 Tile 元素填零，不影响乘加结果。当前真实 Shape 的 M=128、K=896、N=4864 均可被 16 整除，但 Kernel 仍保留边界检查用于其他 Shape。

### 8.6 为什么这个 Kernel 不一定快？

本次源码是教学型通用 Tiled16：

- 每线程只累加**一个** C 元素，寄存器层面输出复用有限。
- 有大量共享内存加载与 Block Barrier，可能产生较高同步/指令开销。
- 没有专门针对大 N、小 M 的线程映射。
- 没有明确实现 Tensor Core/WMMA 路径。
- 没有高阶的流水化、向量化/双缓冲等优化。
- cuBLAS 会针对 GPU 架构、输入 Shape、转置布局选择高度优化实现。

**以上只是合理候选解释，不是对本次 5.23× 差距的已测量微架构归因。** 想判断哪些因素主导，需要进一步做 Nsight Compute 等分析，本次 V6 不重复大规模调参。

---

## 9. PyTorch C++/CUDA Extension：每层接口做了什么？

对应：`v6/gemm_binding.cpp`、`v6/gemm_tiled_v6.cu`、`v6/test_real_gemm.py`。

### 9.1 为什么不能在 Python 里直接调用 `__global__`？

PyTorch Tensor 提供形状、数据类型、设备、底层 Storage 等元数据；CUDA Kernel 运行时只接受设备指针及整数参数。C++/CUDA Extension 是连接二者的接口层：

```text
Python:  X/W_packed (torch.Tensor)
        │
        ▼
PyTorch C++ Binding：检查输入、创建输出、获得 data_ptr
        │
        ▼
CUDA Launcher：拿到 Stream，启动 <<<grid, block>>>
        │
        ▼
GPU：gemm_tiled16 读取 A/B、输出 C
        │
        ▼
返回 Python torch.Tensor
```

### 9.2 关键源码分解

```cpp
torch::Tensor gemm_forward(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.device() == B.device(), "device mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D only");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    c10::cuda::CUDAGuard guard(A.device());
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(A.get_device()).stream();
    launch_gemm_tiled16(A.data_ptr<float>(), B.data_ptr<float>(),
                        C.data_ptr<float>(), M, K, N, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
```

此处为关键流程摘录。**接口只支持两维、CUDA、FP32、连续行主序、维度兼容的矩阵**；没有处理 FP16、任意 Stride、Bias、Autograd Backward，也不声称支持任意超大维度的 `int` 安全转换。

### 9.3 初学者要理解的七个语法/API

| 语法/API | 目的 | 容易混淆点 |
|---|---|---|
| `torch::Tensor` | C++ 世界的 PyTorch Tensor 句柄 | 不等于 `float*` 裸指针 |
| `TORCH_CHECK(cond,msg)` | 输入前置条件失败时抛异常 | 不等于 GPU 数值校验 |
| `A.data_ptr<float>()` | 获取当前 CUDA Tensor 的浮点数据指针 | 不会因此自动搬到 CPU |
| `A.options()` | 保留输出的 Device/DType 等选项 | 并不复制 A 的数据 |
| `torch::empty({M,N},...)` | 申请输出 Tensor，不初始化内容 | 必须保证 Kernel 完整写回 |
| `CUDAGuard` | 设置/恢复正确 CUDA Device 上下文 | 与 Stream 的选择是两个问题 |
| `getCurrentCUDAStream()` | 在 PyTorch 当前 Stream 调度 Kernel | 错用其他 Stream 可能打破依赖 |

### 9.4 为什么一定要使用 PyTorch 当前 CUDA Stream？

模型前面的算子可能在某个 Stream 上产生 `X`，后面的算子需要读取自定义 GEMM 产生的 `C`。在相同 Stream 中按顺序提交，能维护正常的依赖顺序。

如果自定义 Kernel 擅自使用另一个 Stream，却没有合适的 Event 同步/依赖处理，可能出现：

- GEMM 读取到尚未完成写入的输入；
- 下游算子先于 GEMM 完成，读取未就绪的输出；
- 有时正确、有时错误的异步竞态。

仅调用 `cudaDeviceSynchronize()` 虽可能掩盖依赖问题，但引入全设备同步，不是正常集成方案。

### 9.5 `C10_CUDA_KERNEL_LAUNCH_CHECK()` 的边界

该宏用于检查 CUDA Launch 的错误状态，但**不能替代等待 GPU 执行完毕后的错误检查和数值对照**。异步运行时，部分错误可能在之后的同步操作中才浮现。我们的 Python 测试随后执行 `torch.cuda.synchronize()` 并比较输出。

### 9.6 `PYBIND11_MODULE` 与 `load()` 的作用

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward,
          "MiniGPU FP32 GEMM Forward");
}
```

```python
from torch.utils.cpp_extension import load
ext = load(
    name="minigpu_gemm_v6",
    sources=[".../gemm_binding.cpp", ".../gemm_tiled_v6.cu"],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
output = ext.gemm_forward(X, W_packed)
```

`load()` 在需要时调用构建工具编译和动态加载 `.so`。日志中的：

```text
[1/3] nvcc ... gemm_tiled_v6.cu
[2/3] c++ ... gemm_binding.cpp
[3/3] c++ ... -shared ... minigpu_gemm_v6.so
```

意味着 CUDA 部分、C++ Binding、动态库链接成功。后续出现 `ninja: no work to do.` 表示缓存的构建结果可复用，**不是失败，也不是说没有执行 GEMM**。

`CUDA_HOME=/usr/local/cuda-12.8` 指编译工具链路径；`TORCH_CUDA_ARCH_LIST=7.5` 指针对 RTX2060 生成代码；`MAX_JOBS=2` 限制并行编译占用。

### 9.7 一个必须诚实说明的事实

V6 并没有单独开发成熟的 PyTorch `torch.library` 算子注册、Dispatch Key、Autograd Function、Backward、动态 Shape 路由或完整算子打包。它是**通过 Python JIT 构建并加载、供推理使用的实验型 C++/CUDA 扩展**。面试中不要称为工业级自定义算子框架。

---

## 10. 真实 GEMM 正确性与三路径性能比较

### 10.1 正确性用的是什么 Reference？

```python
reference = torch.nn.functional.linear(X, W)
output = ext.gemm_forward(X, W_packed)

diff = (output - reference).abs()
torch.testing.assert_close(output, reference,
                           atol=1e-3, rtol=1e-3)
```

这与 V2 用 CPU FP64 作为 Reference 不同。本阶段检验的是**自定义 GEMM 与当前 PyTorch FP32 Linear 的相容性**；如果需要严格检查独立数值精度，仍可另外使用更高精度 Reference。

数值误差公式（PyTorch `assert_close` 的典型逐元素判定）：

\[
|y_{\rm custom}-y_{\rm ref}|\le\mathrm{atol}+\mathrm{rtol}\,|y_{\rm ref}|.
\]

当前阈值为 `atol=1e-3, rtol=1e-3`，比原始 V2 的小矩阵验收标准宽松，不能只说 PASS 就当作逐位相等。所幸这次实际误差本身很小。

**实测：**

```text
Input Shape:         (128, 896)
Packed Weight Shape: (896, 4864)
Output Shape:        (128, 4864)
Max Abs Diff:        5.245208740234375e-06
Mean Abs Diff:       1.9252415484061203e-07
V6.3.2 Real GEMM Correctness: PASS
```

### 10.2 Benchmark 采用三个对照路径

```python
cases = [
    ("PyTorch Linear",    lambda: F.linear(X, W)),
    ("PyTorch MM Packed", lambda: torch.mm(X, W_packed)),
    ("MiniGPU Tiled16",   lambda: ext.gemm_forward(X, W_packed)),
]
```

分别代表：

1. **真实原始布局**：`F.linear(X,W)`；无需在 Python 侧预先复制权重。
2. **预转置后布局**：`torch.mm(X,W_packed)`；单独研究 packed weight 对成熟库的影响。
3. **自定义 CUDA**：`ext.gemm_forward(X,W_packed)`；与第二条使用相同 packed weight。

**为什么要三条而不是两条？** 如果只比较 F.linear 与自定义 GEMM，无法区分部分布局影响与 Kernel 实现差异。增加 `torch.mm(X,W_packed)` 有助于隔离布局因素，但不是一个“免费预处理”的生产优化方案。

### 10.3 测时脚本大意

```python
WARMUP = 10
REPEAT = 50
TRIALS = 5

for trial in range(TRIALS):
    for step in range(3):
        name, fn = cases[(trial + step) % 3]
        start.record()
        for _ in range(REPEAT):
            result = fn()
        end.record()
        end.synchronize()
        timings[name].append(start.elapsed_time(end) * 1000 / REPEAT)
```

- 每条路径 10 次预热。
- 每个 Trial 连续调用 50 次，CUDA Event 测量区间除以 50，得到每次**平均区间耗时**，单位 μs。
- 5 个 Trial，按轮次轮换路径顺序，取中位数。
- 包含 Tensor 调用、结果 Tensor 创建及 Python 提交期间可能造成的设备空闲；因此**这是所定义测试条件下的平均执行区间，并非 isolating 单个裸 CUDA Kernel 的最纯粹测速**。
- 没有把一次性 `W.T.contiguous()` 的初始化成本放进循环；报告必须写明这一口径。

### 10.4 实测单算子性能

| 执行路径 | Median μs | Min μs | Max μs |
|---|---:|---:|---:|
| PyTorch `F.linear(X,W)` | **286.024** | 279.222 | 288.770 |
| PyTorch `torch.mm(X,W_packed)` | **228.155** | 224.092 | 231.143 |
| MiniGPU Tiled16 | **1496.290** | 1472.784 | 1509.346 |

计算：

\[
\frac{1496.290}{286.024}\approx \boxed{5.23},\qquad
\frac{1496.290}{228.155}\approx\boxed{6.56}.
\]

因此，**Tiled16 相较 F.linear 慢约 5.23 倍，相较 packed PyTorch MM 慢约 6.56 倍**。这里不应该写成“0.191× 加速”，更清晰的表达是“速度比 0.191×，实际性能退化”。

### 10.5 有哪些解释，但不能过度归因？

合理的候选影响：Tiled16 的寄存器分块不足、同步开销、未使用 Tensor Core、特定 Shape 优化不充分，以及成熟库有更好的算法选择。

但没有为此 GEMM 记录 `sm__`、`dram__`、`l1tex__`、`warps stalled` 等 Nsight Compute 指标；所以**不能给出具体百分比归因**，例如“80% 原因是 Shared Memory Bank Conflict”。

### 10.6 这组结果对决策意味着什么？

- 若目标是**提高真实推理速度**：不应把这个慢 5 倍的 GEMM 当默认替代方案。
- 若目标是**验证 CUDA Extension 集成能力**：正确性和执行链路已经有真实证据。
- 若面向**GPU Kernel 专岗**：还需要更深层的针对 Shape 的优化与底层 Profiling；这不是当前 V6 自动完成的事情。

---

## 11. V6.4：只替换真实模型一个 `gate_proj`，做完整 Forward A/B

对应文件：`v6/model_gemm_ab.py`。

### 11.1 为什么要在完整模型里测？

单算子 Benchmark 只能说明某个 GEMM 在被单独调用时的表现；真实模型还包括其他 GEMM、Attention、Layer Norm/RMSNorm、逐元素算子、Python 调度、内存分配等。

**是否值得优化，最终要看用户关心的真实工作负载指标。** 所以我们必须做完整模型 A/B，并检查模型输出是否被改变。

### 11.2 如何局部替换？

```python
linear = model.model.layers[0].mlp.gate_proj
original_forward = linear.forward
W_packed = linear.weight.detach().T.contiguous()

def custom_forward(x):
    original_shape = x.shape
    X = x.reshape(-1, original_shape[-1]).contiguous()
    Y = ext.gemm_forward(X, W_packed)
    return Y.reshape(*original_shape[:-1], W_packed.shape[1])

# 使用自定义路径
linear.forward = custom_forward
...
# 恢复原始路径
linear.forward = original_forward
```

逐段解释：

- 保存 `original_forward`：保证原始实现可恢复，可对照。
- 只替换**第一个 Block 的 gate_proj**：避免一次修改过多模块，降低排障复杂度。
- `W_packed` 只在模型初始化和测量前创建一次。
- 将输入的前面各维折叠成 M，保证 GEMM 接口接受二维 Tensor。
- 输出后恢复原 Shape，使上游/下游 PyTorch 模块无需改动。
- 实验结束后通过 `finally` 恢复原始 `forward`。

**本次不是永久修改权重，也不是完整模型发布格式中的正式算子替换。** `forward` 的 Python 方法替换仅适用于当前推理实验。

### 11.3 完整 Logits 的数值一致性

在相同 `input_ids` 与相同 FP32 模型权重下，对比完整模型：

```text
Logits Shape:      (1, 128, 151936)
Max Abs Diff:      4.0531158447265625e-05
Mean Abs Diff:     2.875729933293769e-06
Top-1 Agreement:   1.0  (100%)
Full Model Correctness: PASS
```

- 局部 GEMM 的很小误差经过后续网络层传播，最终 Logits 误差可能高于单算子输出误差；本次实测仍通过 `atol=1e-3,rtol=1e-3`。
- 该测试覆盖固定随机输入、S=128、B=1、FP32；不是“所有输入下数学严格等价”的证明。
- 仅测试推理前向，没有 Autograd Backward 验收。

### 11.4 完整模型 A/B Benchmark 方法

1. 分别对 Original 和 Custom 预热 5 次。
2. 两条路径使用同一个模型实例、同一组输入、同一精度。
3. 5 个 Trial，Original / Custom 交替顺序。
4. 每个 Trial 连续运行 20 次完整 Forward，在 CUDA Event 区间内统计平均每次耗时。
5. 最后对每条路径的 5 轮时间取中位数。

这比不同日期、不同进程的 `baseline.py` 结果更适合做一次局部替换对照；但仍没有 GPU 固定频率、深度微架构分析，不能把微小差异都解释为硬件层面的必然效果。

### 11.5 实际模型级结果

| 路径 | Median Forward |
|---|---:|
| Original PyTorch | **39.459 ms** |
| Replace 1st `gate_proj` with MiniGPU Tiled16 | **40.711 ms** |
| Original / Custom | **0.969×** |

\[
\Delta T=40.711-39.459=\boxed{1.252\ \text{ms}}.
\]

\[
\frac{40.711-39.459}{39.459}\times100\%\approx\boxed{3.17\%}\quad\text{更慢}.
\]

**验收：模型集成 PASS；模型加速 FAIL。**

### 11.6 一个很有用、但不能过度解释的比较

单算子平均测试区间差：

\[
1496.290-286.024=1210.266\ \mu s\approx 1.210\ \text{ms}.
\]

完整模型 A/B 差：

\[
40.711-39.459=1.252\ \text{ms}.
\]

二者数值接近，**与“被替换 GEMM 导致完整模型退化”这一解释一致**。但由于单算子与整模型 Benchmark 的调度、分配器状态、样本条件和统计口径不同，不能声称已经精确分解整机每一微秒的因果来源。

### 11.7 为什么一个很慢的 GEMM 只让整模型慢 3.17%？

因为我们只替换了 24 个 Transformer Block 中**一个 Block 的一条 `gate_proj`**。其他所有层仍然使用 PyTorch 原生算子，因此绝大部分模型计算没有改变。

可以用 Amdahl 定律理解“局部优化并不等于全局同倍加速”。如果某模块占总耗时比例为 `p`，它被加速 `s` 倍，理想速度提升为：

\[
S_{total}=\frac{1}{(1-p)+p/s}.
\]

当前实验不是加速而是替换后变慢，原理同样提醒我们：**改动影响到的那部分占比有限**，因此整体变化通常小于局部算子相对变化。这里没有精确测出 `p`，不要把某个估计值包装成实测值。

### 11.8 这一步与 V5 Runtime 的区别

V5 的 `DeviceBuffer` / Stream / Event 封装主要是独立 C++/CUDA 工作负载上的资源管理。V6 则在真实 PyTorch 模型的数据流中调用 CUDA Extension。

能说明你理解两种场景的差异：

- 独立 CUDA 程序可以自己维护显存所有权、CUDA Stream、cuBLAS Handle。
- PyTorch Extension 一般应遵循框架已有 Tensor 所有权、CUDA Device/Current Stream、分配器等规则。
- 不要为了“复用 V5 RAII”而在 PyTorch Extension 中重新手动管理所有 Tensor 的裸 `cudaMalloc`，那会增加无谓复制与复杂度。

---

## 12. V6 全部关键文件：谁调用谁、应该优先读哪几行？

### 12.1 文件清单

| 文件 | 作用 | 最关键的阅读位置 |
|---|---|---|
| `v6/baseline.py` | FP32 真实模型基线 | `from_pretrained`、`benchmark()`、CUDA Event |
| `v6/baseline_fp16.py` | FP16 同口径基线 | `DTYPE=torch.float16` |
| `v6/check_fp16.py` | FP32/FP16 同输入 Logits 对照 | `reference`、`diff`、`argmax` |
| `v6/model_profiler_v6.py` | 模型 Forward Profiler | `profile(...)`、`key_averages()` |
| `v6/inspect_real_gemm.py` | Hook 真实 `gate_proj` 输入 | `register_forward_pre_hook`、`X`、`W` |
| `v6/gemm_tiled_v6.cu` | V2 Tiled16 GEMM 适配 | `row/col`、Shared Memory、Barrier、Launcher |
| `v6/gemm_binding.cpp` | Tensor/Device/Stream 检查与 PyBind | `TORCH_CHECK`、`data_ptr`、`getCurrentCUDAStream` |
| `v6/test_real_gemm.py` | 独立真实 Linear 正确性与三路径 Benchmark | `W_packed`、`assert_close`、`cases` |
| `v6/model_gemm_ab.py` | 完整模型替换及 A/B | `original_forward`、`custom_forward`、`finally` |
| `docs/V6_REAL_MODEL.md` | 实验结果、技术推导、限制及面试资料 | 本文 |

### 12.2 项目级调用关系

```text
Python: model_gemm_ab.py
 ├── load CUDA extension
 │    ├── gemm_binding.cpp
 │    └── gemm_tiled_v6.cu
 ├── AutoModelForCausalLM.from_pretrained(...)
 ├── Qwen 原始完整 Forward → Reference logits
 ├── gate_proj.forward = custom_forward
 │    ├── reshape input Tensor
 │    ├── ext.gemm_forward(X, W_packed)
 │    │    ├── TORCH_CHECK shape/device/dtype/layout
 │    │    ├── torch::empty output
 │    │    ├── getCurrentCUDAStream()
 │    │    ├── launch_gemm_tiled16(...)
 │    │    └── gemm_tiled16<<<grid,block,0,stream>>>
 │    └── reshape GEMM output
 ├── Qwen 完整 Forward → Candidate logits
 ├── compare logits and A/B latency
 └── restore original forward
```

**3 个入口一定要分清：**

1. **模型入口**：Python `model(...)` 执行完整 Qwen。
2. **扩展入口**：Python `ext.gemm_forward(...)`，经 PyBind 调用 C++。
3. **GPU Kernel 入口**：CUDA `gemm_tiled16<<<...>>>`，由 C++ Launcher 异步提交。

---

## 13. 复现、构建、日志管理与问题排查

### 13.1 最小环境验收（已经通过，无需反复做）

```bash
python3 - <<'PY'
import torch, transformers
print(torch.__version__, transformers.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
a = torch.ones(4, device='cuda')
print((a * 2).sum().item())
PY
```

**目的：** 验证 PyTorch 能真正执行 GPU Tensor 运算，而不仅是 pip 安装成功。预期 `CUDA available=True`、RTX 2060、SM75、GPU 运算结果 8。**这是复现检查模板，非 V6 原始实测命令输出。**

### 13.2 运行入口命令及应该关注的结果

```bash
cd ~/projects/minigpu
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2
set -o pipefail

HF_HUB_OFFLINE=1 python3 v6/baseline.py          2>&1 | tee v6/baseline.log
HF_HUB_OFFLINE=1 python3 v6/baseline_fp16.py     2>&1 | tee v6/baseline_fp16.log
HF_HUB_OFFLINE=1 python3 v6/check_fp16.py       2>&1 | tee v6/fp16_accuracy.log
HF_HUB_OFFLINE=1 python3 v6/model_profiler_v6.py 2>&1 | tee v6/profile.log
HF_HUB_OFFLINE=1 python3 v6/inspect_real_gemm.py 2>&1 | tee v6/inspect_real_gemm.log
HF_HUB_OFFLINE=1 python3 v6/test_real_gemm.py   2>&1 | tee v6/gemm_benchmark.log
HF_HUB_OFFLINE=1 python3 v6/model_gemm_ab.py     2>&1 | tee v6/model_gemm_ab.log
```

命令作用：

- `export CUDA_HOME`：让 PyTorch 扩展构建找到 CUDA Toolkit。
- `TORCH_CUDA_ARCH_LIST=7.5`：面向 RTX 2060 的 `sm_75` 编译。
- `MAX_JOBS=2`：约束 Ninja 并行编译任务数，减轻本机内存压力。
- `HF_HUB_OFFLINE=1`：本地模型加载，避免外网访问。
- `2>&1 | tee`：把标准输出和错误输出写入日志同时显示在终端。
- `set -o pipefail`：管道中 Python 失败时，Shell 能返回非 0 状态，避免只看 `tee` 成功。

**不用每次从头跑全部命令。** 如果只是复习 Extension，则单跑 `test_real_gemm.py`；若只是复习模型级 A/B，则单跑 `model_gemm_ab.py`。原始日志和报告中的历史数据已经完整保留。

### 13.3 曾遇到的实际错误：`profile.py` 名字冲突

最初文件名 `v6/profile.py`，出现 `ImportError: cannot import name 'AutoModelForCausalLM'`。原因并非 Transformers 安装损坏：

```text
Transformers → torch._dynamo → import cProfile
                              → cProfile import profile
                              → 意外加载 v6/profile.py
                              → 再次导入 Transformers，产生循环导入
```

修复是将脚本重命名为 `model_profiler_v6.py`，而不是重装 PyTorch。**不要给项目源码取名 `profile.py`、`random.py`、`torch.py`、`typing.py` 等容易和标准库/依赖冲突的名字。**

### 13.4 其他常见故障排查清单

| 现象 | 首先检查 | 不应立刻做什么 |
|---|---|---|
| `torch.cuda.is_available() == False` | 驱动/WSL GPU 暴露、PyTorch CUDA 包、设备状态 | 盲目重装全部系统 CUDA |
| 模型文件找不到 | `MODEL_PATH`、`config.json`、`model.safetensors` | 默认重新从外网下载 |
| Extension 编译失败 | `CUDA_HOME`、nvcc、`TORCH_CUDA_ARCH_LIST`、完整编译错误 | 只盯着最后一行 traceback |
| `ninja: no work to do.` | 是否复用了已有编译产物 | 把它当成编译失败 |
| 正确 Shape 但结果错 | `W.T` 是否非连续；Stride 与寻址公式 | 只调大 `atol` 掩盖错误 |
| 偶发错误结果 | CUDA Stream 和异步依赖、是否同步后检查 | 无脑增加全设备同步并宣布解决 |
| Logits 相似但 top-1 不同 | 两个最高分是否接近、差异位置 | 直接宣称模型质量严重下降 |
| 性能差距不稳定 | Warmup、测量范围、GPU 负载、执行顺序 | 只保留最快的一次实验 |

### 13.5 GitHub 发布前检查（操作前验证状态）

用户在 V6 收尾时看到：

```text
?? .venv-v6/
?? v6/
```

这说明目录尚未纳入 Git 跟踪，不意味着实验失败。不要使用裸 `git add .` 把环境/模型/日志混入仓库。参考忽略规则：

```gitignore
.venv-v6/
v6/*.log
v6/__pycache__/
v6/*.so
v6/*.o
v6/*.safetensors
```

**模型权重位于 `~/models/`，应继续留在仓库外。**

建议在 V6 结束时同时更新：`README.md`、`PROJECT_STATUS.md` 和 `docs/V6_REAL_MODEL.md`。**已完成 V6 实机实验，不等于已通过 Git 提交并发布 `v6.0` 标签；需核验 `git status`、`git log`、`git tag`、远端仓库。**

---

## 14. 必须掌握的 20 个 AI Systems / CUDA 高频知识点

> 目标是“能解释真实项目源码”，不是只背概念。

1. **模型参数与显存：** 参数个数 × dtype 字节数仅得到权重理论大小，不等于完整进程显存。
2. **Forward 与 Decode：** 本次测的是不使用 KV Cache 的完整序列 Forward，不是逐 Token Decode。
3. **Logits：** `[B,S,V]`，最后一维是词表分数；不等于实际文本生成。
4. **`eval()` 与 `inference_mode()`：** 一个切换模型行为，一个减少梯度相关运行时开销。
5. **FP32/FP16：** 精度减少可能节省带宽和显存，但不保证整体稳定加速。
6. **Tensor Core：** 硬件支持 ≠ 当前 Kernel 已经调用相应指令；需要验证。
7. **CUDA Event：** 同 Stream 时间线的事件计时；正确同步保证测量完毕。
8. **Warmup：** 减少初始化影响，不代表消除所有频率/调度波动。
9. **Profiler：** 判断热点，避免“喜欢哪个算子就优化哪个”。
10. **ATen vs CUDA Kernel：** 不同统计层级有重叠，不能直接相加。
11. **GEMM Shape：** `M×K` 乘 `K×N` 输出 `M×N`；Shape 决定性能特征。
12. **FLOPs：** GEMM 常用 `2MKN` 估算计算量；不能据此直接得出真实耗时。
13. **Row-major 与 Stride：** `offset=i*s0+j*s1`；转置视图可能非连续。
14. **`.contiguous()`：** 可能分配并复制数据；推理热路径中要计入代价。
15. **Shared Memory Tiling：** 减少重复 Global Memory 读取，代价是同步与空间占用。
16. **`__syncthreads()`：** 保护跨线程共享数据的读写次序。
17. **C++/CUDA Extension：** Tensor → 形状/设备检查 → 裸指针 → CUDA Launcher → Tensor。
18. **Current CUDA Stream：** 与 PyTorch 计算图的执行依赖一致，避免异步竞态。
19. **数值一致性：** `atol+rtol×abs(ref)`，同时检查非有限值；误差阈值须说明。
20. **模型级优化：** 单 Kernel 变快/变慢不等于完整模型同倍变化；报告必须有 A/B。

---

## 15. 计算题与源码验收（建议先作答，后看第 16 章）

### 15.1 Tensor Shape / 基础计算（5 题）

**题 1：** 模型 hidden size=896，Q heads=14，KV heads=2，求 head_dim；每个 KV Head 对应几个 Q Heads？

**题 2：** 输入 `X=[128,896]`，权重 `W=[4864,896]`，求 `W.T` 和输出 `Y` Shape；写出 `Y[i,j]` 的求和公式。

**题 3：** 该 GEMM 的乘加 FLOPs（以 FMA=2 FLOPs 计）、FP32 预打包权重大小是多少 bytes 和 MiB？

**题 4：** Tiled16：Grid 有多少个 Block？每 Block 多少 Threads？K 循环多少轮？

**题 5：** 把 FP32 S=128 `46.921ms` 与 FP16 S=128 `40.539ms` 转为加速比与时间下降百分比；解释两者为何不相等。

### 15.2 源码解释（3 题）

**题 6：** 解释这三行中每个语法成分的作用：

```cpp
int row = blockIdx.y * TILE + threadIdx.y;
__shared__ float As[TILE][TILE];
__syncthreads();
```

**题 7：** 为什么不能把 `W.T.data_ptr<float>()` 直接传给按 `B[k*N+n]` 寻址的 Tiled16？用 Stride 说明。

**题 8：** 解释 `getCurrentCUDAStream()`、`CUDAGuard`、`C10_CUDA_KERNEL_LAUNCH_CHECK()` 三者的不同职责。

### 15.3 连续追问（4 题）

**题 9：** Profiler 中 `volta_sgemm_128x64_tn` 是 86.74%，`aten::mm` 是 83.02%，是否说明 GEMM 合计占 169.76%？为什么？

**题 10：** 如果 Tiled16 单算子慢 5.23 倍，为什么完整模型只慢 3.17%？你能从现有数据证明具体占比吗？

**题 11：** FP16 Logits 的 S=32 Top-1 一致率为 93.75%，你是否允许在简历中写“FP16 模型精度完全不变”？

**题 12：** 你是工程负责人，在看到自定义 GEMM 慢于 cuBLAS 后，会强行替换 Qwen 所有 Linear，还是停下来保留负结果？为什么？

---

## 16. 参考答案与面试官评价标准

### 答案 1：GQA 头维度

`head_dim=896/14=64`；`14/2=7`，每个 KV Head 对应 7 个 Query Heads（按分组共享关系理解）。

### 答案 2：Linear Shape

`W.T=[896,4864]`；`Y=[128,4864]`；

\[
Y[i,j]=\sum_{k=0}^{895}X[i,k]W[j,k].
\]

注意 `W[j,k]`，不要把原始权重按 `[K,N]` 误读。

### 答案 3：GEMM FLOPs 与权重字节数

`2×128×896×4864=1,115,684,864 FLOPs`；`896×4864×4=17,432,576 bytes =16.625 MiB`。

### 答案 4：Grid/Block

`grid.x=ceil(4864/16)=304`，`grid.y=ceil(128/16)=8`，共 `304×8=2432` Blocks；每 Block `16×16=256` Threads；K 循环 `ceil(896/16)=56` 轮。

### 答案 5：FP16 加速比

`46.921/40.539≈1.157×`，时间降低 `(46.921−40.539)/46.921≈13.60%`。加速比是速度/延迟的比值，不等于直接的百分比节约。

### 答案 6：CUDA 索引、Shared Memory、同步

`blockIdx.y` 指 Block 在 Grid 的 y 方向编号；`TILE` 表示每 Block 负责的输出 Tile 行数；`threadIdx.y` 是 Block 内局部行号；`__shared__` 申请同 Block 共享的片上 Shared Memory；`__syncthreads()` 用于 Block 内线程屏障，确保同一阶段的共享数据写入对其他线程可见。

### 答案 7：Stride

`W` 连续 Shape `[N,K]` 的 Stride 是 `[K,1]`，`W.T` 视图的 Stride 是 `[1,K]`，但 V2 连续 B 假设 Stride `[N,1]`，两者不同。应预转置 `.contiguous()` 或改寻址公式、支持显式 Stride。

### 答案 8：三个接口不同职责

- `CUDAGuard`：当前设备上下文。
- `getCurrentCUDAStream()`：当前设备上 PyTorch 正在使用的 Stream，保证依赖顺序。
- `C10_CUDA_KERNEL_LAUNCH_CHECK()`：检查 Kernel Launch 状态；异步错误和数值正确性仍需同步、断言验证。

### 答案 9：Profiler 重叠

ATen `aten::mm` 和底层 SGEMM Kernel 对应同一执行链路的不同统计层级，时间有重叠，不能相加。86.74% 也不等于原始非 Profiler 模型墙钟中的精确百分比。

### 答案 10：局部到整体

仅替换第一个 Block 的一个线性算子；大部分完整模型没变。两个 Benchmark 区间差约 1.210 ms 和 1.252 ms，结果相符，但因独立测试条件不能精确把它们当成同一个公式。要谨慎使用 Amdahl 分析。

### 答案 11：FP16 精度声明

不允许。本次随机输入 S32 有 2 个位置 Top-1 不同，没有自然语言质量评测，最多声称该实验的 Logits 数值比较和观察到的延迟/显存变化。

### 答案 12：工程取舍

保留正确性和负性能结果，生产/主优化路径继续采用成熟库。根据实际岗位/业务需求再评估 Tensor Core WMMA、针对特定 Shape 的优化；不为显示功能数量而给全模型替换更慢的 GEMM。

### 面试官如何评价“通过”？

只有在**不照着答案**的情况下，能做到以下四项才算对 V6 真正掌握：

1. 画出 `X → Binding → Kernel → Y` 数据流，并说明形状与内存布局。
2. 不看代码讲解 Tiled16 的 `row/col`、加载、Barrier、累加、写回。
3. 准确说出哪组实测性能加速、哪组性能退化，解释实验口径和局限。
4. 能实际修改一处 Shape/Stride/Stream 相关代码并验证正确性，而不只是读过 AI 给出的结果。

---

## 17. 项目面试表述、价值与下一步

### 17.1 一分钟项目介绍（真实且不夸大）

> MiniGPU 是我基于 RTX 2060 和 CUDA 12.8 构建的渐进式 GPU 计算与推理性能实验工程。前面实现了 Naive/Tiled/Register-Blocking GEMM，建立与 cuBLAS 的正确性和性能对照，学习 Nsight Profiling、Transformer 常见算子和 CUDA Stream/Buffer 管理。V6 使用真实 Qwen2.5-0.5B 预训练权重，做 FP32/FP16 Forward、显存和 Logits 对比；Profiler 显示 FP32 GEMM 占据主要 GPU 计算时间。之后我提取了第一层 MLP `gate_proj` 的真实输入和权重，把 Tiled16 封装成 PyTorch CUDA Extension，通过完整模型 Logits 验证，但实测自定义 GEMM 比成熟库明显慢，因此没有虚构加速，而是分析布局、执行口径与后续针对特定 Shape 优化的必要性。

### 17.2 简历可写 / 不可写

| 可以写 | 不可以写 |
|---|---|
| “在 RTX 2060 上完成 Qwen2.5-0.5B FP32/FP16 基线与 Profiling” | “自主开发高性能 Qwen 推理引擎” |
| “FP16 S128 延迟从 46.921ms 降到 40.539ms；峰值已分配显存从 2.049GiB 降到 1.065GiB” | “FP16 无精度损失” |
| “复用自定义 Tiled GEMM，并通过 PyTorch CUDA Extension 接入真实模型” | “自定义 GEMM 显著超过 cuBLAS” |
| “单算子及完整模型 Logits 数值校验通过，保留 0.969× 的负性能结果” | “全模型成功加速 20%” |
| “使用 Current CUDA Stream，解释布局打包与 A/B 方法” | “实现 Tensor Core WMMA GEMM / 完整 GPU Runtime” |

### 17.3 当前项目技术缺口（按优先级）

**面向一般 AI Systems / 推理优化实习：**

1. **优先补源码掌控：** 自己解释/局部重写 Binding、Kernel、A/B 三条主线。
2. **补基本数值分析：** 理解 FP16 误差和 `torch.testing.assert_close` 的条件。
3. **完善仓库交付：** V6 源码、本文、README、PROJECT_STATUS 同步，Git tag 真实核验。
4. **如需进一步求职区分度：** 深入调试一次典型 GPU 性能退化（形状、Stream、显存或 Kernel）。

**面向 GPU Kernel / 矩阵库专项实习：** 进一步研究 SM75 Tensor Core、WMMA、FP16/BF16 能力边界、针对真实 Linear Shape 的 Kernel 数据布局/寄存器分块，以及使用 Nsight Compute 的有证据优化；当前 V6 没有覆盖这些深入工作。

### 17.4 该阶段应该在这里停止扩需求

不要为 V6 额外增加无关的 RMSNorm 接入、完整 FlashAttention、动态 KV Cache、通用调度器、全模型 Linear 替换或多个大模型 Benchmark。是否深化 GEMM，应由下一轮求职目标和已有知识掌握情况决定，不应为了给项目堆数字而增加功能。

### 17.5 最终工程结论（必须准确）

- **项目是否真实运行？** 是：RTX 2060、真实权重、真实 PyTorch Forward、真实 CUDA Extension 已有日志。
- **自定义 Kernel 是否接入真实模型？** 是：第一层 `gate_proj` 的实验性 `forward` 替换。
- **数值正确性是否通过？** 在本次固定输入和指定容差下通过；不是所有输入严格等价证明。
- **模型是否因自定义 GEMM 加速？** **没有。** 完整模型从 39.459ms 变为 40.711ms，慢约 3.17%。
- **FP16 是否有收益？** 是：S128 的独立 FP16 基线延迟下降约 13.6%、峰值已分配显存下降约 48.0%；数值有差异，尚无生成质量评估。
- **V6 是否构成一个完整实验闭环？** 是：**基线 → Profiling → 真实 Tensor/布局 → 自定义算子 → 正确性 → 性能 A/B → 负结果与局限记录**。
- **学习/面试是否自动验收通过？** 否：工程测试通过 ≠ 已经能独立解释、修改核心源码；需完成第 15～16 章的实际口述和代码验收。

---

## 附录 A. 实验原始数据索引与追溯

本报告使用的实机日志来源（文件名为当时用户在 WSL 中保存的对应日志，本文未宣称这些 `.log` 已提交 Git）：

| 原始日志或测试 | 关键证据 |
|---|---|
| `v6/baseline.log` | FP32 S32/S128 延迟与已分配峰值显存 |
| `v6/baseline_fp16.log` | FP16 S32/S128 延迟与显存 |
| `v6/fp16_accuracy.log` | 同输入 FP16/FP32 的 Max/Mean Diff、Top-1 |
| `v6/profile.log` | 主要 SGEMM Kernel、ATen 算子、自身 CUDA 时间 |
| `v6/inspect_real_gemm.log` | 真实 `X/W/W.T/Y` Shape、Contiguous |
| `v6/gemm_correctness.log` | Extension 首次构建与独立线性结果误差 |
| `v6/gemm_benchmark.log` | PyTorch Linear / MM Packed / MiniGPU Tiled16 三路径延迟 |
| `v6/model_gemm_ab.log` | 完整模型 Logits 数值 PASS、Original/Custom A/B |

**注意：** 这份文档固化了从上述日志获得的关键数据，不能据此假称 `v6/*.log` 在远端仓库已经存在。

## 附录 B. 复习时最值得保留的四句原则

1. **先 Profile 再优化；选择瓶颈必须靠真实工作负载。**
2. **Shape 不是内存布局；Stride、DType、Device、Stream 都属于算子正确性前提。**
3. **正确性 PASS 与性能 PASS 是两个独立的验收项；单算子与整模型也必须分开测。**
4. **一个记录充分、解释准确的负结果，比不真实的加速数字更有工程价值。**

---

**文档结束。** 本版用于替换之前较简略的 `docs/V6_REAL_MODEL.md`；保存时请以该路径入库，并结合仓库源文件复习，不要仅凭文字替代亲自修改和运行核心代码。
