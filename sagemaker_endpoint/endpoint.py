import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from time import gmtime, strftime

# 세션 및 기본 설정
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::666803869796:role/SKKU_SageMaker_Role' # SageMaker 실행 역할 ARN
region = sagemaker_session.boto_region_name
s3_client = boto3.client('s3', region_name=region)

# --- 설정 변수 ---
model_s3_path = 's3://say1-4team-bucket/sagemaker/inference/models/model.tar.gz' # 1단계에서 업로드한 파일 경로
endpoint_name = 'say1-4team-serverless-endpoint' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())


# 1. PyTorchModel 객체 생성
# entry_point: code/ 폴더 기준 상대 경로
pytorch_model = PyTorchModel(
    model_data=model_s3_path,
    role=role,
    entry_point='inference.py',
    source_dir='code',
    framework_version='1.13', # 사용하는 버전에 맞게 수정
    py_version='py39'
)

# 2. 서버리스 추론을 위한 Endpoint 배포
# serverless_inference_config를 통해 메모리 크기와 동시성 설정
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t3.xlarge', # 서버리스에서는 이 타입이 청구 기준이 아님
    endpoint_name=endpoint_name,
    serverless_inference_config=sagemaker.serverless.ServerlessInferenceConfig(
        memory_size_in_mb=6144,  # 메모리 (MB): 1024, 2048, 3072, 4096, 5120, 6144 중 선택
        max_concurrency=1,       # 최대 동시 요청 수
    ),
    tags=[
        {
            'Key': 'project',
            'Value': 'pre-4team'
        },
    ],
)




print(f"✅ SageMaker 서버리스 엔드포인트 [{endpoint_name}] 배포를 시작했습니다.")
print("배포 완료까지 몇 분 정도 소요될 수 있습니다.")