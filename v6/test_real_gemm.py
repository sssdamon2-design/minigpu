from statistics import median
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
MODEL_PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"

def main():
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False

    # 编译并加载自己的 CUDA GEMM
    ext = load(
        name="minigpu_gemm_v6",
        sources=[
            str(ROOT / "gemm_binding.cpp"),
            str(ROOT / "gemm_tiled_v6.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        local_files_only=True
    ).to("cuda").eval()

    linear = model.model.layers[0].mlp.gate_proj
    captured = {}

    def hook(module, args):
        captured["x"] = args[0].detach()

    handle = linear.register_forward_pre_hook(hook)

    ids = torch.randint(
        100, 10000, (1, 128), device="cuda"
    )

    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)

    handle.remove()

    X = captured["x"].reshape(-1, 896).contiguous()
    W = linear.weight.detach()

    # 只执行一次预转置，避免每次 GEMM 都复制权重
    W_packed = W.T.contiguous()

    with torch.inference_mode():
        reference = F.linear(X, W)
        output = ext.gemm_forward(X, W_packed)

    torch.cuda.synchronize()

    diff = (output - reference).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Input Shape:", tuple(X.shape))
    print("Packed Weight Shape:", tuple(W_packed.shape))
    print("Output Shape:", tuple(output.shape))
    print("Max Abs Diff:", max_diff)
    print("Mean Abs Diff:", mean_diff)

    torch.testing.assert_close(
        output, reference,
        atol=1e-3,
        rtol=1e-3
    )

    print("V6.3.2 Real GEMM Correctness: PASS")

    # 三条路径使用相同的 X 和模型权重
    cases = [
        ("PyTorch Linear",
         lambda: F.linear(X, W)),
        ("PyTorch MM Packed",
         lambda: torch.mm(X, W_packed)),
        ("MiniGPU Tiled16",
         lambda: ext.gemm_forward(X, W_packed)),
    ]

    WARMUP = 10
    REPEAT = 50
    TRIALS = 5

    timings = {name: [] for name, _ in cases}

    with torch.inference_mode():
        for _, fn in cases:
            for _ in range(WARMUP):
                fn()

        torch.cuda.synchronize()

        # 轮换测试顺序，降低固定顺序带来的偏差
        for trial in range(TRIALS):
            for step in range(len(cases)):
                name, fn = cases[
                    (trial + step) % len(cases)
                ]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                for _ in range(REPEAT):
                    result = fn()

                end.record()
                end.synchronize()

                latency_us = (
                    start.elapsed_time(end) * 1000 / REPEAT
                )
                timings[name].append(latency_us)

    print("\n=== V6 Real GEMM Benchmark ===")
    for name, _ in cases:
        values = timings[name]
        print(
            f"{name}: "
            f"median={median(values):.3f} us, "
            f"min={min(values):.3f} us, "
            f"max={max(values):.3f} us"
        )

    speedup = (
        median(timings["PyTorch Linear"])
        / median(timings["MiniGPU Tiled16"])
    )
    print(f"MiniGPU vs PyTorch Linear: {speedup:.3f}x")


if __name__ == "__main__":
    main()
