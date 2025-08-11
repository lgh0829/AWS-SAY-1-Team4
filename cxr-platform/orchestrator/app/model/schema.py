from pydantic import BaseModel, Field
from typing import List, Optional

class WebhookPayload(BaseModel):
    StudyInstanceUID: str
    OrthancStudyID: Optional[str] = None
    Status: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True

class WebhookResponse(BaseModel):
    message: str

class SegmentationResult(BaseModel):
    mask: str  # S3 URI of the segmentation mask
    overlay: str  # S3 URI of the segmentation overlay

class SegmentationPayload(BaseModel):
    image: str  # base64 인코딩된 문자열
    s3_bucket: str
    s3_prefix: str

class ClassificationPayload(BaseModel):
    primary_image: str  # base64 인코딩된 문자열
    background_image: str  # base64 인코딩된 문자열
    s3_bucket: str
    s3_prefix: str

class ClassificationResult(BaseModel):
    label: str  # 분류 결과 레이블
    confidence: float  # 분류 확신도
    xai: str  # XAI 결과 (예: Grad-CAM 이미지의 S3 URI)