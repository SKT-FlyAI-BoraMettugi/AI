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
