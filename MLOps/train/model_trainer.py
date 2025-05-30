import os, json, torch, mlflow, deepspeed
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator 
from deepspeed_config import get_deepspeed_config 

def train_model(experiment_name, model, tokenizer, tokenized_train_dataset, tokenized_validation_dataset):
    # MLflow 실험 설정
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():  # MLflow 실행 시작
        # MLflow에 학습 파라미터 기록
        hyperparams = {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "num_train_epochs": 3
        }
        mlflow.log_params(hyperparams)

        os.makedirs(experiment_name, exist_ok=True)
        with open(f"{experiment_name}/config.json", "w") as f:
            json.dump(hyperparams, f)
    
        # DeepSpeed 설정 로드 (별도 파일에서 가져옴)
        ds_config = get_deepspeed_config(gradient_accumulation_steps=2)

        # TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=experiment_name,
            per_device_train_batch_size=hyperparams["batch_size"],   # GPU VRAM 최적화
            gradient_accumulation_steps=2,  # 작은 배치로 큰 배치 효과
            learning_rate=hyperparams["learning_rate"],
            num_train_epochs=hyperparams["num_train_epochs"],
            logging_dir=f"{experiment_name}/logs",
            logging_steps=10,
            save_strategy="epoch",
            fp16=True, # AMP 사용
            optim="adamw_torch",
            report_to="none",
            deepspeed=ds_config,
        )

        # DeepSpeed 모델 초기화
        #model, optimizer, _, _ = deepspeed.initialize(
            #model=model,
            #model_parameters=model.parameters(),
            #config=ds_config
        #)

        # Trainer 설정
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset 
        )

        print("모델 학습 시작...")
        trainer.train()
        print("모델 학습 완료!")

        # 학습 후 성능 평가
        train_metrics = trainer.evaluate()
        print(f"학습 평가 결과: {train_metrics}")
        mlflow.log_metrics(train_metrics)

        # 모델 저장
        model.save_pretrained(f"{experiment_name}/finetuned_model")
        tokenizer.save_pretrained(f"{experiment_name}/finetuned_tokenizer")

        # 모델을 AMP 적용 해제 후 저장
        accelerator = Accelerator()
        # 모델을 FP32로 변환 후 저장
        unwrapped_model = accelerator.unwrap_model(model).to(torch.float32)

        model_save_path = f"{experiment_name}/finetuned_model.pt"
        torch.save(unwrapped_model.state_dict(), model_save_path)

        mlflow.log_artifact(model_save_path)

        mlflow.log_artifact(f"{experiment_name}/config.json")  
        mlflow.log_artifact(f"{experiment_name}/logs")  

        torch.cuda.empty_cache()

    return model