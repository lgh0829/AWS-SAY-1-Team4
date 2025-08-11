import boto3
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64
from app.core.config import settings
from app.model.schema import SegmentationPayload, SegmentationResult, ClassificationPayload, ClassificationResult

# SageMaker 엔드포인트
endpoint_segmentation = settings.SEGMENTATION_ENDPOINT
endpoint_classification = settings.CLASSIFICATION_ENDPOINT
runtime = boto3.client("sagemaker-runtime", region_name=settings.AWS_REGION)

# S3 설정
s3_bucket = settings.S3_BUCKET
s3_prefix = settings.S3_PREFIX.rstrip('/')
def invoke_segmentation(image_path: Path, study_uid: str, s3_bucket: str = s3_bucket, s3_prefix: str = s3_prefix, endpoint: str = endpoint_segmentation) -> SegmentationResult:
    """
    SageMaker 엔드포인트를 호출하여 이미지 세그멘테이션 수행
    Args:
        image_path (Path): 입력 이미지 경로
        study_uid (str): StudyInstanceUID
        endpoint (str): SageMaker 엔드포인트 이름
    Returns:
        SegmentationResult:
            mask (str): S3에 업로드된 세그멘테이션 마스크의 URI
            overlay (str): S3에 업로드된 세그멘테이션 오버레이의 URI
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
        s3_prefix=f'{s3_prefix}/studies/{study_uid}/segmentation',
    )

    # JSON payload
    payload = payload_model.model_dump_json()

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload
    )

    # 결과 파싱
    result = json.loads(response["Body"].read().decode())
    return SegmentationResult(
        mask=result["mask_s3_uri"],
        overlay=result["overlay_s3_uri"]
    )

def invoke_classification(raw_image_path: Path, mask_image_path: Path,study_uid: str, s3_bucket: str = s3_bucket, s3_prefix: str = s3_prefix, endpoint: str = endpoint_classification) -> ClassificationResult:
    """
    SageMaker 엔드포인트를 호출하여 이미지 분류 수행
    Args:
        image_path (Path): 입력 이미지 경로
        study_uid (str): StudyInstanceUID
        endpoint (str): SageMaker 엔드포인트 이름
    Returns:
        ClassificationResult:
            label (str): 분류 결과 레이블
            confidence (float): 분류 확신도
            xai (str): XAI 결과 이미지의 S3 URI
    """
    # JPEG 이미지 → bytes (입력 크기 그대로 유지)
    image = Image.open(raw_image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # 마스크 이미지 → bytes (입력 크기 그대로 유지)
    mask_image = Image.open(mask_image_path).convert("RGB")
    mask_buffer = io.BytesIO()
    mask_image.save(mask_buffer, format="JPEG")
    mask_bytes = mask_buffer.getvalue()

    # 이미지 base64 인코딩
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')

    # payload 모델 생성
    payload_model = ClassificationPayload(
        primary_image=image_base64,  # bytes를 list로 변환
        background_image=mask_base64,  # bytes를 list로 변환
        s3_bucket=s3_bucket,
        s3_prefix=f'{s3_prefix}/studies/{study_uid}/classification',
    )

    # JSON payload
    payload = payload_model.model_dump_json()

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload
    )

    # 결과 파싱
    result = json.loads(response["Body"].read().decode())
    return ClassificationResult(
        label=result["label"],   # 예. label="Normal"
        confidence=result["confidence"],  # 예. confidence=0.95
        xai=result["xai"]  # 예. xai="s3://example-bucket/xai_result.png"
    )