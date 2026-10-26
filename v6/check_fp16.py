
import torch
from transformers import AutoModelForCausalLM

PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"
SEQ_LENGTHS = [32, 128]

torch.manual_seed(2026)
torch.backends.cuda.matmul.allow_tf32 = False

# 两种精度使用完全相同的输入
inputs = {
    s: torch.randint(100, 10000, (1, s), device="cuda")
    for s in SEQ_LENGTHS
}

def load_model(dtype):
    return AutoModelForCausalLM.from_pretrained(
        PATH,
        dtype=dtype,
        local_files_only=True
    ).to("cuda").eval()

# 先计算 FP32 参考结果
model = load_model(torch.float32)
reference = {}

with torch.inference_mode():
    for s in SEQ_LENGTHS:
        reference[s] = model(
            input_ids=inputs[s],
            use_cache=False
        ).logits.float().cpu()

del model
torch.cuda.empty_cache()

# 再计算 FP16，逐项与 FP32 比较
model = load_model(torch.float16)

with torch.inference_mode():
    for s in SEQ_LENGTHS:
        output = model(
            input_ids=inputs[s],
            use_cache=False
        ).logits.float().cpu()

        ref = reference[s]
        diff = (output - ref).abs()

        top1_match = (
            output.argmax(dim=-1) == ref.argmax(dim=-1)
        ).float().mean().item()

        print(f"\nSequence Length: {s}")
        print(f"Max Abs Diff: {diff.max().item():.6f}")
        print(f"Mean Abs Diff: {diff.mean().item():.6f}")
        print(f"Top-1 Agreement: {top1_match:.2%}")
        print(f"Output finite: {torch.isfinite(output).all().item()}")

print("\nV6.3.1 Accuracy Comparison: COMPLETED")
