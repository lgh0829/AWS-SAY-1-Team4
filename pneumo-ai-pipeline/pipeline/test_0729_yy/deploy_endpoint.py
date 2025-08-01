# deploy_endpoint.py

import os
import sagemaker
import boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
from time import gmtime, strftime

# ─────────────────────────────────────────────────────────────
# 1) 환경 변수 / 상수 설정
# ─────────────────────────────────────────────────────────────
# (1) SageMaker 실행에 사용할 IAM Role ARN
#     또는 환경변수로도 지정 가능합니다.
ROLE_ARN = 'arn:aws:iam::666803869796:role/SKKU_SageMaker_Role'

# (2) 모델 아티팩트(.tar.gz) S3 위치
MODEL_S3_PATH = "s3://say1-4team-bucket/sagemaker/inference/models/model.tar.gz"

# (3) 히트맵 업로드용 S3 버킷/프리픽스
HEATMAP_BUCKET = "say1-4team-bucket"
HEATMAP_PREFIX = "sagemaker/test"

# (4) 생성할 엔드포인트 이름
ENDPOINT_NAME = 'say1-4team-inference-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# ─────────────────────────────────────────────────────────────
# 2) SageMaker 세션 생성
# ─────────────────────────────────────────────────────────────
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
s3_client = boto3.client('s3', region_name=region)

# ─────────────────────────────────────────────────────────────
# 3) Model 객체 정의
#    - entry_point: inference.py
#    - source_dir : 핸들러와 의존 모듈들이 있는 디렉토리
#    - env        : 컨테이너 내 환경 변수
# ─────────────────────────────────────────────────────────────
pytorch_model = PyTorchModel(
    model_data=MODEL_S3_PATH,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="source_dir",
    framework_version="1.13.1",  # 사용 중인 PyTorch 버전과 맞추세요
    py_version="py39",
    sagemaker_session=sagemaker_session,
    env={
        "HEATMAP_S3_BUCKET": HEATMAP_BUCKET,
        "HEATMAP_S3_PREFIX": HEATMAP_PREFIX
    }
)

# ─────────────────────────────────────────────────────────────
# 5) 엔드포인트 배포
# ─────────────────────────────────────────────────────────────
print(f"▶ Deploying realtime endpoint '{ENDPOINT_NAME}' ...")
predictor = pytorch_model.deploy(
    endpoint_name=ENDPOINT_NAME,
    instance_type="ml.m5.xlarge",           # ← 인스턴스 타입 지정
    initial_instance_count=1,               # ← 인스턴스 개수 지정
    tags=[
        {
            'Key': 'project',
            'Value': 'pre-4team'
        },
    ],
    wait=True
)
print(f"✅ Endpoint '{ENDPOINT_NAME}' is ready!")
print(f" Invoke with boto3/sagemaker-runtime on '{ENDPOINT_NAME}'")
print(f" Heatmaps will be uploaded to s3://{HEATMAP_BUCKET}/{HEATMAP_PREFIX}/")
