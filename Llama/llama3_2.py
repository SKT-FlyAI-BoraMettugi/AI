import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

# 데이터 로드
with open("train_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 메시지 포맷 변경
def format_messages(example):
    messages = example["messages"]
    formatted_text = ""
    
    for i in range(len(messages)-1):
        role = messages[i]["role"]
        content = messages[i]["content"]
        
        if role == "system":
            formatted_text += f"[SYSTEM] {content}\n"
        elif role == "user":
            formatted_text += f"[USER] {content}\n"
        elif role == "assistant":
            formatted_text += f"[ASSISTANT] {content}\n"
    
    return {"text": formatted_text}

# 데이터셋 변환
dataset = Dataset.from_list([format_messages(d) for d in raw_data])

# 모델 및 토크나이저 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit Quantization
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정 (파라미터 축소)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Llama 모델의 주요 어텐션 모듈
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 데이터 토크나이징
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # labels 추가
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./bllossom_finetune",
    per_device_train_batch_size=2,  # GPU VRAM 최적화
    gradient_accumulation_steps=4,  # 작은 배치로 큰 배치 효과
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Mixed Precision
    optim="adamw_torch",
)

# 학습
from transformers import Trainer
import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment("llama-3.2_파인튜닝")

with mlflow.start_run():
    mlflow.log_params({
        "model_name": model_name,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    })

    # 기존의 학습 코드
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    # 학습 후 메트릭 기록
    train_metrics = trainer.evaluate()
    mlflow.log_metrics(train_metrics)

    # 모델 저장 및 MLflow에 기록
    model.save_pretrained("./bllossom_finetuned")
    tokenizer.save_pretrained("./bllossom_finetuned")
    
    # MLflow에 모델 아티팩트 저장
    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="model",
        signature=infer_signature(tokenized_datasets["input_ids"], model.generate(tokenized_datasets["input_ids"])),
    )