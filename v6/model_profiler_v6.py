
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/damon/models/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 128

def main():
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False

    # 加载与 V6.1 相同的模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        local_files_only=True
    ).to("cuda").eval()

    input_ids = torch.randint(
        100, 10000,
        (1, SEQ_LEN),
        device="cuda"
    )

    with torch.inference_mode():
        # 预热，避免将首次初始化误认为模型瓶颈
        for _ in range(5):
            model(input_ids=input_ids, use_cache=False)

        torch.cuda.synchronize()

        # 只采集一次完整 Forward，控制 Profiling 开销
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True
        ) as prof:
            output = model(
                input_ids=input_ids,
                use_cache=False
            )
            torch.cuda.synchronize()

    print("Logits Shape:", tuple(output.logits.shape))
    print("\n=== Top 25 CUDA Operations ===")

    events = prof.key_averages()

    # 兼容不同版本的时间字段
    time_key = (
        "self_device_time_total"
        if hasattr(events[0], "self_device_time_total")
        else "self_cuda_time_total"
    )

    print(events.table(sort_by=time_key, row_limit=25))
    print("\nV6.2 Profiling: PASS")

if __name__ == "__main__":
    main()
