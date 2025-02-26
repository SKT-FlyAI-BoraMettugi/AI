import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoModel
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import numpy as np

class ScoringLlamaModel(nn.Module):
    def __init__(self, model_id):
        super(ScoringLlamaModel, self).__init__()

        # 기존 LLaMA 모델 로드
        self.llama = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

        for param in self.llama.parameters():
            param.requires_grad = False

        # LoRA 설정 (파라미터 축소)
        lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Llama 모델의 주요 어텐션 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
        )

        self.llama = get_peft_model(self.llama, lora_config)

        # 기존 LLaMA의 lm_head 제거
        # del self.llama.lm_head

        # 새로운 회귀/분류 레이어 추가 (출력: 5개 점수)
        embedding_dim = self.llama.config.hidden_size  # 모델의 임베딩 차원
        self.llama.score_head = nn.Linear(embedding_dim, 5)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # LLaMA의 마지막 hidden state 가져오기
        outputs = self.llama(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, embedding_dim)

        # 마지막 토큰의 hidden state만 추출 (CLS 토큰처럼 사용)
        last_token_hidden = hidden_states[:, -1, :]  # (batch_size, embedding_dim)

        # 새로운 score_head를 거쳐 5개 점수 예측
        scores = self.llama.score_head(last_token_hidden)  # (batch_size, 5)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(scores, labels)

        return {"loss": loss, "logits": scores}



# 데이터 로드
with open("./Llama/data.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# 메시지 포맷 변경
def format_messages(example):
    question = example["question"]
    answer = example["answer"]
    logic = example["logic"]
    thinking = example["thinking"]
    creativity = example["creativity"]
    persuasion = example["persuasion"]
    depth = example["depth"]

    formatted_text = f"[ASSISTANT] {question}" + f"[USER] {answer}"
    scores = [logic, thinking, creativity, persuasion,depth]
    
    # for i in range(len(messages)):
    #     role = messages[i]["role"]
    #     content = messages[i]["content"]
        
    #     if role == "system":
    #         formatted_text += f"[SYSTEM] {content}\n"
    #     elif role == "user":
    #         formatted_text += f"[USER] {content}\n"
    #     elif role == "assistant":
    #         formatted_text += f"[ASSISTANT] {content}\n"
    
    return {"text": formatted_text, "labels": scores}

# 데이터셋 변환
dataset = Dataset.from_list([format_messages(d) for d in raw_data])

# 모델 및 토크나이저 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = ScoringLlamaModel(model_name).to("cuda")

# LoRA 설정 (파라미터 축소)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Llama 모델의 주요 어텐션 모듈
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, lora_config)

# 데이터 토크나이징
# def tokenize_function(examples):
#     tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
#     tokenized["labels"] = tokenized["input_ids"].copy()  # labels 추가
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

def preprocess_data(examples):
    input_texts = examples["text"]
    labels = np.array(examples["labels"]).astype(np.float32)

    tokenized_inputs = tokenizer(input_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float32)
    }

train_dataset = Dataset.from_list(dataset).map(preprocess_data, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./bllossom_finetune",
    per_device_train_batch_size=4,  # GPU VRAM 최적화
    gradient_accumulation_steps=2,  # 작은 배치로 큰 배치 효과
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
        train_dataset=train_dataset,
    )

    trainer.train()

    torch.cuda.empty_cache()

    # # 학습 후 메트릭 기록
    # train_metrics = trainer.evaluate()
    # mlflow.log_metrics(train_metrics)

    # 모델 저장 및 MLflow에 기록
    model.save_pretrained("./bllossom_finetuned")
    tokenizer.save_pretrained("./bllossom_finetuned")
    
    # # MLflow에 모델 아티팩트 저장
    # mlflow.transformers.log_model(
    #     transformers_model=model,
    #     artifact_path="model",
    #     signature=infer_signature(tokenized_datasets["input_ids"], model.generate(tokenized_datasets["input_ids"])),
    # )
