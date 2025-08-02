from pydantic import BaseModel
from typing import List

class WebhookPayload(BaseModel):
    StudyInstanceUID: str

class WebhookResponse(BaseModel):
    message: str

class SegmentationResult(BaseModel):
    mask: str  # S3 URI of the segmentation mask

class SegmentationPayload(BaseModel):
    image: List[int]  # JPEG 바이트 배열을 정수 리스트로 표현
    s3_bucket: str
    s3_prefix: str
    original_image: List[List[List[float]]]  # 예: 1024x1024x3 RGB 이미지