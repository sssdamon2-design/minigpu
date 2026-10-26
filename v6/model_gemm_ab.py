
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
MODEL_PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"

torch.manual_seed(2026)
torch.backends.cuda.matmul.allow_tf32 = False

# 复用已经通过正确性验证的 V6 GEMM Extension
ext = load(
    name="minigpu_gemm_v6",
    sources=[
        str(ROOT / "gemm_binding.cpp"),
        str(ROOT / "gemm_tiled_v6.cu"),
    ],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to("cuda").eval()

linear = model.model.layers[0].mlp.gate_proj
original_forward = linear.forward

# 预转置只在初始化时执行一次
W_packed = linear.weight.detach().T.contiguous()

def custom_forward(x):
    original_shape = x.shape

    X = x.reshape(-1, original_shape[-1]).contiguous()
    Y = ext.gemm_forward(X, W_packed)

    return Y.reshape(
        *original_shape[:-1],
        W_packed.shape[1]
    )

input_ids = torch.randint(
    100, 10000, (1, 128), device="cuda"
)

def forward_model():
    return model(
        input_ids=input_ids,
        use_cache=False
    ).logits

def benchmark():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(20):
        forward_model()

    end.record()
    end.synchronize()

    return start.elapsed_time(end) / 20

try:
    with torch.inference_mode():
        # 原始完整模型
        reference = forward_model()

        # 只替换第一层的 gate_proj
        linear.forward = custom_forward
        candidate = forward_model()

        diff = (candidate - reference).abs()

        print("=== Full Model Correctness ===")
        print("Logits Shape:", tuple(candidate.shape))
        print("Max Abs Diff:", diff.max().item())
        print("Mean Abs Diff:", diff.mean().item())
        print(
            "Top-1 Agreement:",
            (candidate.argmax(-1) == reference.argmax(-1))
            .float().mean().item()
        )

        torch.testing.assert_close(
            candidate,
            reference,
            atol=1e-3,
            rtol=1e-3
        )

        print("Full Model Correctness: PASS")

        # 两条路径分别预热
        for custom in [False, True]:
            linear.forward = (
                custom_forward if custom
                else original_forward
            )

            for _ in range(5):
                forward_model()

        torch.cuda.synchronize()

        # 交替测试，避免始终先测同一路径
        timings = {"original": [], "custom": []}

        for trial in range(5):
            order = (
                ["original", "custom"]
                if trial % 2 == 0
                else ["custom", "original"]
            )

            for name in order:
                linear.forward = (
                    custom_forward if name == "custom"
                    else original_forward
                )

                timings[name].append(benchmark())

        original_ms = median(timings["original"])
        custom_ms = median(timings["custom"])

        print("\n=== Full Model Benchmark ===")
        print(f"Original: {original_ms:.3f} ms")
        print(f"Custom:   {custom_ms:.3f} ms")
        print(f"Speedup:  {original_ms / custom_ms:.3f}x")

        print("\nV6.4 Model Integration: PASS")

finally:
    # 无论测试是否成功，都恢复原始模型
    linear.forward = original_forward
