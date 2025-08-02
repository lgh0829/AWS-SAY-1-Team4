import boto3
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64
from app.core.config import settings
from app.model.schema import SegmentationPayload, SegmentationResult

# SageMaker 엔드포인트
endpoint_name = settings.SEGMENTATION_ENDPOINT
runtime = boto3.client("sagemaker-runtime")

# S3 설정
s3_bucket = settings.S3_BUCKET
s3_prefix = settings.S3_PREFIX.rstrip('/')

def invoke_segmentation(image_path: Path, study_uid: str, s3_bucket: str = s3_bucket, s3_prefix: str = f'{s3_prefix}/seg', endpoint: str = endpoint_name) -> SegmentationResult:
    """
    SageMaker 엔드포인트를 호출하여 이미지 세그멘테이션 수행
    Args:
        image_path (Path): 입력 이미지 경로
        study_uid (str): StudyInstanceUID
        endpoint (str): SageMaker 엔드포인트 이름
    Returns:
        S3_uri (str): S3에 업로드된 세그멘테이션 마스크의 URI
        S3_uri (str): S3에 업로드된 세그멘테이션 오버레이의 URI
    """

    # JPEG 이미지 → bytes (입력 크기 그대로 유지)
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # 이미지 base64 인코딩
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # payload 모델 생성
    payload_model = SegmentationPayload(
        image=image_base64,  # bytes를 list로 변환
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix
    )

    # JSON payload
    payload = payload_model.model_dump_json()

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload
    )

    result = json.loads(response["Body"].read().decode())
    return SegmentationResult(mask=result["mask"])

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