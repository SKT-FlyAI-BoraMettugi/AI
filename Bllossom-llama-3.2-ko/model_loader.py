import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_model(model_name):
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # 8-bit Quantization
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA 설정 (파라미터 축소)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Llama 모델의 주요 어텐션 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    return model