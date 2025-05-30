# 메시지 포맷 변경
def format_messages(example):
    messages = example["messages"]
    formatted_text = ""
    
    for i in range(len(messages)):
        role = messages[i]["role"]
        content = messages[i]["content"]
        
        if role == "system":
            formatted_text += f"[SYSTEM] {content}\n"
        elif role == "user":
            formatted_text += f"[USER] {content}\n"
        elif role == "assistant":
            formatted_text += f"[ASSISTANT] {content}\n"
    
    return {"text": formatted_text}

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"], 
        padding="longest", 
        truncation=True, 
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"]  # labels 추가
    return tokenized

def convert_to_message(data):
    return {
        "role": "user",
        "content": data['question']
    }, {
        "role": "assistant",
        "content": f"""
            답변: {data['answer']}
            창의력: ({data['creativity']}/10)점 {data['creativity_review']}
            논리력: ({data['logic']}/10)점 {data['logic_review']}
            사고력: ({data['thinking']}/10)점 {data['thinking_review']}
            설득력: ({data['persuasion']}/10)점 {data['persuasion_review']}
            추론의 깊이: ({data['depth']}/10)점 {data['depth_review']}
        """        
    }

def convert_to_raw(data):
    SYSTEM_PROMPT = {
        "role": "system",
        "content": "너는 아동 논리 퀴즈 답변을 채점하는 AI 모델이야. 논리력, 사고력, 창의력, 설득력, 추론의 깊이 점수를 1~10으로 평가하고 근거를 설명해야 해."
    }
    converted_data = []
    for d in data:
        user, assistant = convert_to_message(d)
        messages = [
            SYSTEM_PROMPT,
            user,
            assistant
        ]
        converted_data.append({"messages": messages})
    return converted_data