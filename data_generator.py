import os
import sys
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

def generate_eval_data(theme, diff, questions):
    result_list = []

    for i in range(1, 501):  # 10쌍 생성

        prompt = f"""
        테마: {theme}
        난이도: {diff}
        질문: {questions}

        초등학생을 위한 퀴즈 플랫폼의 채점 모델 학습 데이터를 생성합니다.
        당신은 질문에 답변을 작성하는 초등학생이자, 기준에 따라 채점하는 선생님입니다.

        1) 우선 당신은 입력된 질문에 대해 초등학생 입장에서 답변을 생성합니다.
        2) 이후 당신은 선생님이 되어 평가 기준에 따라 1)에서 생성한 답변을 채점합니다.

        **답변 생성 지침**
        1. 질문에 대한 답변을 **초등학생 수준**으로 생성하세요.
           - 답변 생성 시, 각 채점 항목 별로 아래와 같이 확률적으로 생각의 깊이를 조절하세요:
             - 20% 확률로 깊이 생각 (9-10점 수준; 풍부한 논리, 창의적 접근, 구체적 근거)
             - 30% 확률로 중간 수준 생각 (7-8점 수준; 보통의 논리 전개)
             - 30% 확률로 단순하게 생각 (4-6점 수준; 표면적, 간단한 설명)
             - 20% 확률로 무식하게 생각 (1-3점 수준; 엉뚱하고 허무맹랑하며 근거가 매우 부족)
           - 이전 반복과는 전혀 다른 새로운 상황, 예시, 스타일을 사용해 다양하게 표현해 주세요.
        2. 생성된 답변에 대해, 아래 **평가 기준**에 따라 각 평가 항목의 점수를 무작위로 매기고, 그 근거를 높임말로 상세히 작성하세요.
           - **주의**: 반드시 평가 기준을 준수하여 일관성있게 채점하고, 각 점수에 맞는 구체적인 근거를 제시해 주세요.

        **평가 기준**
        '논리력':
            10: '체계적이고 완벽한 논리',
            9: '매우 논리적이지만 약간 부족',
            8: '논리적이나 일부 연결이 약함',
            7: '논리적이지만 일부 비약 있음',
            6: '논리적이지만 보완 필요',
            5: '다소 논리적이지만 약점 많음',
            4: '논리적 비약이 많음',
            3: '논리가 부족함',
            2: '논리가 거의 없음',
            1: '완전히 비논리적'
        '사고력':
            10: '깊은 통찰력',
            9: '높은 사고력',
            8: '다각도로 분석',
            7: '나름대로 분석',
            6: '보통 수준의 사고',
            5: '사고가 제한적',
            4: '단순한 생각',
            3: '매우 단순한 사고',
            2: '사고의 폭이 매우 좁음',
            1: '깊이 없는 사고'
        '창의력':
            10: '혁신적 아이디어',
            9: '매우 창의적',
            8: '독창적인 접근',
            7: '새롭지만 약간 익숙한 접근',
            6: '창의성이 있지만 평범',
            5: '평범한 수준',
            4: '거의 창의성 없음',
            3: '전혀 창의적이지 않음',
            2: '전혀 색다른 접근이 아님',
            1: '기존 답변 복사 수준'
        '설득력':
            10: '강력한 근거 제시',
            9: '강한 설득력',
            8: '설득력 있음',
            7: '보통의 설득력',
            6: '설득력이 부족',
            5: '설득력이 부족함',
            4: '설득이 미흡',
            3: '설득력이 거의 없음',
            2: '설득이 불가능함',
            1: '설득력이 전혀 없음'
        '추론의 깊이':
            10: '깊은 분석',
            9: '깊이 있는 접근',
            8: '꽤 깊이 탐구',
            7: '적당한 깊이',
            6: '표면적 추론',
            5: '깊이가 없음',
            4: '단순한 추론',
            3: '얕은 이해',
            2: '피상적 접근',
            1: '아무런 추론 없음'

        **출력 형식**
        아래 형식을 그대로 유지하여 답변을 작성하세요.

        0. 답변
        - 생성한 답변을 여기에 작성하세요.

        1. 논리력 (X/10)
        - 이 점수에 대한 근거를 작성하세요.

        2. 사고력 (X/10)
        - 이 점수에 대한 근거를 작성하세요.

        3. 창의력 (X/10)
        - 이 점수에 대한 근거를 작성하세요.

        4. 설득력 (X/10)
        - 이 점수에 대한 근거를 작성하세요.

        5. 추론의 깊이 (X/10)
        - 이 점수에 대한 근거를 작성하세요.

         (주의)
        - 같은 질문이라도, 반복 호출마다 점수와 답변이 다르게 나오도록 해 주세요.
        - 반드시 위 평가 기준을 준수하여, 각 항목에 대해 해당 점수에 맞는 근거를 상세히 작성해 주세요.

        """

        # GPT 호출
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.9
            )
            generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            generated_text = f"GPT 호출 실패: {str(e)}"
            print(e)
            sys.exit(1)

        # 결과 저장
        result_list.append({
            "theme": theme,
            "difficulty": diff,
            "question": questions,
            "생성된 답변": generated_text
        })

        print(f'{i}번째 데이터')

    return result_list

# 입력 파일 처리 및 출력 파일 생성

input_file = f"input_data.csv"
output_file = f"learning_dataset/output_data.csv" ## 공(하) 공(중) 공(상) / 우(하) 우(중) 우(상)
data = []

try:
  chunk = pd.read_csv(input_file) ## input file 테마 난이도 문제

  for index, row in chunk.iterrows():## 각 문제 별로
    data.extend(generate_eval_data(row['테마'], row['난이도'], row['문제'])) # 100개 생성

  df_output = pd.DataFrame(data)
  df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
  print(f"출력 데이터 저장 완료: {output_file}")

except FileNotFoundError:
  print(f"파일이 존재하지 않습니다: {input_file}")

print("모든 입력 파일 처리 및 출력 파일 생성 완료!")