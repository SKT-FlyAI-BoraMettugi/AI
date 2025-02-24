from data_loader import load_data
from model_loader import load_model
from tokenizer_loader import load_tokenizer
from data_processing import tokenize_function
from model_trainer import train_model

# 데이터 로드 (훈련 & 검증 데이터 분리)
train_dataset, validation_dataset = load_data('test.jsonl', 0.2)

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

model = train_model('Nolli-test', model, tokenizer, tokenized_train_dataset, tokenized_validation_dataset)
