import optuna
from data_loader import load_file_data
from model_loader import load_model
from tokenizer_loader import load_tokenizer
from data_processing import tokenize_function
from model_trainer import train_model

def objective(trial):
    """Optuna가 사용할 최적화 함수"""
    # 최적화할 하이퍼파라미터 범위 설정
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6, 8])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)

    # 데이터 로드 (훈련 & 검증 데이터 분리)
    train_dataset, validation_dataset = load_file_data('final_dataset.jsonl', 0.2)

    # 모델 및 토크나이저 로드
    model = load_model('Bllossom/llama-3.2-Korean-Bllossom-3B')
    tokenizer = load_tokenizer('Bllossom/llama-3.2-Korean-Bllossom-3B')

    # 토크나이징 (훈련 데이터 & 검증 데이터 각각 처리)
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),  
        batched=True, 
        remove_columns=["text"]
    )
    tokenized_validation_dataset = validation_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),  
        batched=True, 
        remove_columns=["text"]
    )

    # 학습 실행 및 성능 평가 (eval_loss 최소화)
    return train_model(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        experiment_name="Optuna-Tuning",
        model=model,
        tokenizer=tokenizer,
        tokenized_train_dataset=tokenized_train_dataset,
        tokenized_validation_dataset=tokenized_validation_dataset
    )

# Optuna 스터디 생성 및 최적화 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", study.best_params)
