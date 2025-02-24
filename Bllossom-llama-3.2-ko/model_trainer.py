import torch
import mlflow
import json
import os
import deepspeed
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator 
from deepspeed_config import get_deepspeed_config 

def train_model(experiment_name, model, tokenizer, tokenized_train_dataset, tokenized_validation_dataset):
    # MLflow 실험 설정
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():  # MLflow 실행 시작
        # MLflow에 학습 파라미터 기록
        hyperparams = {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "num_train_epochs": 3
        }
        mlflow.log_params(hyperparams)

        #os.makedirs(experiment_name, exist_ok=True)
        #with open(f"{experiment_name}/config.json", "w") as f:
            #json.dump(hyperparams, f)
    
        # DeepSpeed 설정 로드 (별도 파일에서 가져옴)
        ds_config = get_deepspeed_config(
            batch_size=hyperparams["batch_size"],
            gradient_accumulation_steps=2
        )

        # TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=experiment_name,
            per_device_train_batch_size=hyperparams["batch_size"],   # GPU VRAM 최적화
            gradient_accumulation_steps=ds_config["gradient_accumulation_steps"],  # 작은 배치로 큰 배치 효과
            learning_rate=hyperparams["learning_rate"],
            num_train_epochs=hyperparams["num_train_epochs"],
            logging_dir=f"{experiment_name}/logs",
            logging_steps=10,
            save_strategy="epoch",
            fp16=True, 
            optim="adamw_torch",
            deepspeed=ds_config,
        )

        # ✅ DeepSpeed 모델 초기화
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        # Trainer 설정
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset 
        )

        trainer.train()

        # 학습 후 성능 평가 결과 기록
        train_metrics = trainer.evaluate()
        mlflow.log_metrics(train_metrics)

        # 모델 저장
        model.save_pretrained(f"{experiment_name}/finetuned_model")
        tokenizer.save_pretrained(f"{experiment_name}/finetuned_tokenizer")

        # 모델을 AMP 적용 해제 후 저장
        #accelerator = Accelerator()
        #unwrapped_model = accelerator.unwrap_model(model) 

        #mlflow.pytorch.log_model(unwrapped_model, "model_checkpoint")

        #mlflow.log_artifact(f"{experiment_name}/config.json")  
        #mlflow.log_artifact(f"{experiment_name}/logs")  

        torch.cuda.empty_cache()

    return model