import deepspeed

def get_deepspeed_config(batch_size=4, gradient_accumulation_steps=2):
    ds_config = {
        "train_batch_size": batch_size,
        "zero_optimization": {
            "stage": 2,  # ZeRO Stage 2 적용 (VRAM 최적화)
            "offload_optimizer": {"device": "cpu"},  # 옵티마이저 CPU 오프로딩 (VRAM 절약)
            "contiguous_gradients": True,  # 연속적인 그래디언트 사용 (메모리 효율화)
        },
        "fp16": {"enabled": True},  # Mixed Precision (FP16) 사용
        "gradient_accumulation_steps": gradient_accumulation_steps,  # 작은 배치로 큰 배치 효과
    }

    return ds_config
