import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 설정
bucket_name = os.getenv("S3_BUCKET_NAME")
local_base_dir = "/Users/skku_aws19/Desktop/aws_project/AWS-SAY-1-Team4/pneumo-ai-pipeline/data/masks"
s3_prefix = "cxr-pneumonia-4/masks"

s3 = boto3.client("s3")

for root, _, files in os.walk(local_base_dir):
    for file in files:
        if file.endswith(".txt"):  # 마스크 파일만
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_base_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Windows 대응

            print(f"☁️ Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)

print("✅ 전체 마스크 업로드 완료")