import torch, os, boto3, tempfile
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tokenizer_loader import load_tokenizer

def load_model(model_name):
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # 8-bit Quantization
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA 설정 (파라미터 축소)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Llama 모델의 주요 어텐션 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    return model

def download_model(bucket_name):
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{os.environ["MINIO_URL"]}",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

    # 임시 폴더 생성
    temp_dir = tempfile.mkdtemp()

    # 모델 파일들 다운로드
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix='models')

    for obj in objects['Contents']:
        key = obj['Key']
        file_name = os.path.basename(key)
        if not file_name: 
            continue    
        local_path = os.path.join(temp_dir, file_name)
        s3.download_file(bucket_name, key, local_path)

    return temp_dir

def load_trained_model(model_name):
    bucket_name = os.environ["S3_BUCKET_NAME"]
    local_dir = download_model(bucket_name)

    # base model 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,  # 원본 모델 이름
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, local_dir)
    tokenizer = load_tokenizer(local_dir)

    return model, tokenizer

