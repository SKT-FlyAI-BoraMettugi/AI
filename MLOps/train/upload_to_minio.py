import os, boto3

def upload_model_to_minio(model_path, s3_model_path):
    local_model_path = f'{model_path}/finetuned_model'
    tokenizer_path = f'{model_path}/finetuned_tokenizer'
        
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{os.environ["MINIO_URL"]}",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

    # safetensors 및 adapter_config.json
    safetensors_files = [os.path.normpath(os.path.join(local_model_path, f)) for f in os.listdir(local_model_path) if f.endswith(".safetensors")]
    adapter_config_files = [os.path.normpath(os.path.join(local_model_path, f)) for f in os.listdir(local_model_path) if f.endswith("adapter_config.json")]
    tokenizer_files = [os.path.normpath(os.path.join(tokenizer_path, f)) for f in os.listdir(tokenizer_path) if f.endswith("tokenizer.json")]

    # config.json 확인
    config_path = os.path.normpath(os.path.join(model_path, "config.json"))
    config_files = [config_path] if os.path.exists(config_path) else []

    # 업로드할 파일 목록
    model_files = safetensors_files + adapter_config_files + tokenizer_files + config_files 

    bucket_name = os.environ["S3_BUCKET_NAME"]
    for file_path in model_files:
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            file_name = os.path.basename(file_path)
            # Multipart Upload 대신 put_object 사용
            with open(abs_path, "rb") as data:
                s3.put_object(Bucket=bucket_name, Key=f"{s3_model_path}/{file_name}", Body=data)
            print(f"Uploaded: {bucket_name}/{s3_model_path}/{file_name}")
        else:
            print(f"파일 없음: {abs_path}")
    print(f"Model uploaded to MinIO: {bucket_name}/{s3_model_path}")