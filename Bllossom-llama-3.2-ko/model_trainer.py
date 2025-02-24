import torch
from transformers import Trainer, TrainingArguments

def train_model(output_dir, model, tokenized_train_dataset, tokenized_validation_dataset):
    # 학습 설정
    training_args = TrainingArguments(
    output_dir="./finetuned_model",
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

    # 기존의 학습 코드
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset 
    )

    trainer.train()
    torch.cuda.empty_cache()

    return model