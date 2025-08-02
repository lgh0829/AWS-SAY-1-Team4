import boto3
import json
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dotenv
from pathlib import Path
import os
import base64

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3_prefix = 'test/segmentation/output'

# 설정
endpoint_name = "say1-4team-seg-25-08-02-18-17"
runtime = boto3.client("sagemaker-runtime")

# 1. 이미지 불러오기
image_path = 'test_raw_image2.jpg'
image = Image.open(image_path).convert('RGB')
# 원본 해상도 그대로 전송 → inference.py에서 Resize 수행
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

# 2. JSON payload 생성 (바이트 배열을 리스트로 변환)
image_base64 = base64.b64encode(image_bytes).decode('utf-8')
original_image_np = np.array(image).astype(np.float32)  # 원본 해상도 (1024x1024)
payload = json.dumps({
    "image": image_base64,
    "s3_bucket": BUCKET_NAME,
    "s3_prefix": s3_prefix
})

# 3. 엔드포인트 호출
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=payload
)

# 4. 결과 파싱
result = json.loads(response['Body'].read().decode())
print("Segmentation Result:")
print("Mask S3 URI:", result["mask_s3_uri"])
print("Overlay S3 URI:", result["overlay_s3_uri"])