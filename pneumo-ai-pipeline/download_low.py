import boto3
import pandas as pd
import os
from botocore.exceptions import ClientError

# 1. S3 설정
s3 = boto3.client('s3')
bucket_name = 'say1-4team-bucket'
base_prefix = 'cxr-pneumonia-4/preprocessed/test'

# 2. CSV 읽기
df = pd.read_csv('/Users/skku_aws19/Desktop/aws_project/AWS-SAY-1-Team4/pneumo-ai-pipeline/low_confidence_images.csv')
image_paths = df['Image'].tolist()

# 3. 로컬 저장 디렉토리
local_base_dir = 'downloaded_low_conf'
os.makedirs(local_base_dir, exist_ok=True)

# 4. 하위 폴더들 순회 (0, 1, 2)
subfolders = ['0', '1', '2']

# 5. 파일 다운로드
for relative_path in image_paths:
    found = False
    for subfolder in subfolders:
        s3_key = f"{base_prefix}/{subfolder}/{os.path.basename(relative_path)}"
        local_path = os.path.join(local_base_dir, subfolder, os.path.basename(relative_path))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            s3.download_file(bucket_name, s3_key, local_path)
            print(f"✅ Downloaded: {s3_key} → {local_path}")
            found = True
            break
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                continue  # 다른 폴더로 계속 시도
            else:
                print(f"❌ Error downloading {s3_key}: {e}")
                break
    if not found:
        print(f"❌ Not found in any subfolder: {relative_path}")