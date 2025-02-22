import pandas as pd
import numpy as np

# CoT 데이터셋 불러오기
df = pd.read_parquet("hf://datasets/heegyu/CoT-collection-ko/train.parquet")
df_selected = df[['source', 'rationale']]

# 입력 데이터를 2000개로 분할
chunks = np.array_split(df_selected, 2000)

# 각 chunk 저장
for i, chunk in enumerate(chunks):
    filename = f"input_{i+1}.csv"
    chunk.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"입력 데이터 저장 완료: {filename}")