import optuna
from model_trainer import train_model

def objective(trial):
    """Optuna가 사용할 최적화 함수"""
    # 최적화할 하이퍼파라미터 범위 설정
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)

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
study.optimize(objective, n_trials=10)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", study.best_params)
