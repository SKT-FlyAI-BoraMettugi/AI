import json
from datasets import Dataset
from data_processing import format_messages

def load_data(file_name, test_size=0.2):
    # 데이터 로드
    with open(file_name, "r", encoding="utf-8-sig") as f:
        raw_data = [json.loads(line) for line in f]

    # 데이터셋 변환
    dataset = Dataset.from_list([format_messages(d) for d in raw_data])

    # 학습 데이터와 검증 데이터로 나누기
    split_dataset = dataset.train_test_split(test_size=test_size)
    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    return train_dataset, validation_dataset