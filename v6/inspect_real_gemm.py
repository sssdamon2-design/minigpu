
import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"

torch.manual_seed(2026)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to("cuda").eval()

# 取第一层 Transformer 的 MLP gate_proj
linear = model.model.layers[0].mlp.gate_proj

captured = {}

# 在 Linear 执行前，提取实际输入
def capture_input(module, args):
    captured["x"] = args[0].detach()

handle = linear.register_forward_pre_hook(capture_input)

input_ids = torch.randint(
    100, 10000, (1, 128), device="cuda"
)

with torch.inference_mode():
    model(input_ids=input_ids, use_cache=False)

handle.remove()

# 将 [1,128,896] 视为 GEMM 的 [128,896]
X = captured["x"].reshape(-1, model.config.hidden_size)
W = linear.weight.detach()

with torch.inference_mode():
    Y = torch.nn.functional.linear(X, W)

print("X Shape:", tuple(X.shape))
print("W Shape:", tuple(W.shape))
print("W.T Shape:", tuple(W.T.shape))
print("Y Shape:", tuple(Y.shape))

print("X contiguous:", X.is_contiguous())
print("W contiguous:", W.is_contiguous())
print("W.T contiguous:", W.T.is_contiguous())
print("DType:", X.dtype)

assert Y.shape == (128, model.config.intermediate_size)
assert torch.isfinite(Y).all().item()

print("V6.3.2 Real GEMM Extraction: PASS")
