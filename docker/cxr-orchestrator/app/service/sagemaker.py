import boto3
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
from app.core.config import settings

# SageMaker 엔드포인트
endpoint_name = settings.SEGMENTATION_ENDPOINT
runtime = boto3.client("sagemaker-runtime")

def invoke_segmentation(image_path: Path, endpoint: str = endpoint_name) -> dict:
    # 이미지 로드 및 리사이즈 (inference.py 기준: 512x512)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # 이미지 → JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # JSON payload
    payload = json.dumps({"image": list(image_bytes)})

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload
    )

    result = json.loads(response["Body"].read().decode())
    return result

def invoke_classification(image_path: Path, endpoint: str = settings.CLASSIFICATION_ENDPOINT) -> dict:
    # 이미지 로드 및 리사이즈 (inference.py 기준: 512x512)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # 이미지 → JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # JSON payload
    payload = json.dumps({"image": list(image_bytes)})

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload
    )

    result = json.loads(response["Body"].read().decode())
    return result