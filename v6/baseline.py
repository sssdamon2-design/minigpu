
import statistics

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = "/home/damon/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 5
REPEAT = 20
SEQ_LENGTHS = [32, 128]


def benchmark(model, seq_len):
    # 固定随机输入，控制 Token 数量
    input_ids = torch.randint(
        100, 10000,
        (1, seq_len),
        device=DEVICE
    )

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # GPU 预热
        for _ in range(WARMUP):
            logits = model(
                input_ids=input_ids,
                use_cache=False
            ).logits

        torch.cuda.synchronize()

        samples = []

        for _ in range(REPEAT):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            logits = model(
                input_ids=input_ids,
                use_cache=False
            ).logits

            end.record()
            end.synchronize()

            samples.append(start.elapsed_time(end))

    # 不将正确性检查计入推理延迟
    valid = torch.isfinite(logits).all().item()

    peak_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(f"\nSequence Length: {seq_len}")
    print(f"Logits Shape: {tuple(logits.shape)}")
    print(f"Output finite: {valid}")
    print(f"Median Latency: {statistics.median(samples):.3f} ms")
    print(f"Min Latency: {min(samples):.3f} ms")
    print(f"Max Latency: {max(samples):.3f} ms")
    print(f"Peak Allocated Memory: {peak_gib:.3f} GiB")

    if not valid:
        raise RuntimeError("Non-finite model output")


def main():
    assert torch.cuda.is_available(), "CUDA is unavailable"

    torch.manual_seed(2026)

    # FP32 对照实验，避免 TF32 引入额外变量
    torch.backends.cuda.matmul.allow_tf32 = False

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA Runtime:", torch.version.cuda)
    print("Model:", MODEL_ID)
    print("DType:", DTYPE)

    print("\nLoading pretrained model...")

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    local_files_only=True
).to(DEVICE).eval()

    param_count = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {param_count:,}")
    print("Layers:", model.config.num_hidden_layers)
    print("Hidden Size:", model.config.hidden_size)
    print("Attention Heads:", model.config.num_attention_heads)
    print("KV Heads:", model.config.num_key_value_heads)

    for seq_len in SEQ_LENGTHS:
        benchmark(model, seq_len)

    print("\nV6.1 Model Baseline: PASS")


if __name__ == "__main__":
    main()
