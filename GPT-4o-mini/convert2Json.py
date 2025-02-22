import pandas as pd
import json

import os

input_dir = '/content/drive/MyDrive/colab_outputs/'
output_dir = '/content/drive/MyDrive/colab_json_outputs/'

# 입력 및 출력 폴더 생성
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 디렉토리 내 파일 순회
for file_name in os.listdir(input_dir):
    if file_name.startswith('output_') and file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path)

        # JSONL로 변환
        jsonl_data = []
        for _, row in df.iterrows():
            entry = {
                "messages": [
                    {"role": "system", "content": "너는 아동 논리 퀴즈 답변을 채점하는 AI 모델이야. 논리력, 사고력, 창의력, 설득력, 추론의 깊이 점수를 1~10으로 평가하고 근거를 설명해야 해."},
                    {"role": "user", "content": row['source']},
                    {"role": "assistant", "content": row['생성된 답변']}
                ]
            }
            jsonl_data.append(entry)

        # JSONL 파일 저장
        jsonl_file = os.path.join(output_dir, f"{file_name.replace('.csv', '.jsonl')}")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"파인튜닝 데이터 저장 완료: {jsonl_file}")