import random
import openai

# OpenAI API 키 설정
openai.api_key = "OPENAI_API_KEY"

def generate_eval_data(source, rationale):
    result_list = []

    for i in range(1, 4):  # 3쌍 생성
        # 랜덤 점수
        scores = {
            "logic": random.randint(1, 10),
            "thinking": random.randint(1, 10),
            "creativity": random.randint(1, 10),
            "persuasion": random.randint(1, 10),
            "depth": random.randint(1, 10)
        }

        prompt = f"""
        질문: {source}
        답변: {rationale}

        아래 기준에 따라 **평가 점수**와 **근거**, **답변**을 생성해 주세요.

        **평가 기준**
        1. **논리력 (1~10)**: 주어진 단서를 얼마나 체계적으로 분석하고 결론을 도출했는가?
        2. **사고력 (1~10)**: 단서를 다양하게 해석하고 여러 가능성을 고려했는가?
        3. **창의력 (1~10)**: 정해진 틀에서 벗어나 색다른 관점으로 접근했는가?
        4. **설득력 (1~10)**: 자신의 주장을 뒷받침하는 근거를 얼마나 효과적으로 제시했는가?
        5. **추론의 깊이 (1~10)**: 표면적인 정보뿐만 아니라 숨겨진 가능성까지 탐구했는가?

        **최종 답변:**
        rationale이 source에 대해 논리력 사고력 창의력 설득력 추론의 깊이가 모두 10점의 답변이라고 할 때 아래의 점수에 맞는 source에 대한 답변을 생성해주세요
        그리고 이 답변이 각 항목 별로 아래의 점수를 갖게 된 근거를 아래에서 작성해주세요

        1. 논리력 ({scores['logic']}/10)
           - 이 점수에 대한 근거를 작성해 주세요.

        2. 사고력 ({scores['thinking']}/10)
           - 이 점수에 대한 근거를 작성해 주세요.

        3. 창의력 ({scores['creativity']}/10)
           - 이 점수에 대한 근거를 작성해 주세요.

        4. 설득력 ({scores['persuasion']}/10)
           - 이 점수에 대한 근거를 작성해 주세요.

        5. 추론의 깊이 ({scores['depth']}/10)
           - 이 점수에 대한 근거를 작성해 주세요.

        **최종 답변:**
        rationale이 source에 대해 논리력 사고력 창의력 설득력 추론의 깊이가 모두 10점의 답변이라고 할 때 위의 점수에 맞는 source에 대한 답변을 생성해주세요
        """

        # GPT 호출
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            generated_text = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            generated_text = f"GPT 호출 실패: {str(e)}"

        # 결과 저장
        result_list.append({
            "source": source,
            "rationale": rationale,
            "논리력": scores['logic'],
            "사고력": scores['thinking'],
            "창의력": scores['creativity'],
            "설득력": scores['persuasion'],
            "추론의 깊이": scores['depth'],
            "생성된 답변": generated_text
        })

    return result_list

# 입력 파일 처리 및 출력 파일 생성
for i in range(1930,1939):
    input_file = f"input_{i}.csv"
    output_file = f"output_{i}.csv"
    data = []

    try:
        chunk = pd.read_csv(input_file)
        for index, row in chunk.iterrows():
            data.extend(generate_eval_data(row['source'], row['rationale']))

        df_output = pd.DataFrame(data)
        df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"출력 데이터 저장 완료: {output_file}")

    except FileNotFoundError:
        print(f"파일이 존재하지 않습니다: {input_file}")

print("모든 입력 파일 처리 및 출력 파일 생성 완료!")