import os
import argparse
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import yaml
import os
import re
from torch.optim import lr_scheduler
from datetime import datetime
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path
import dotenv


dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

def replace_env_vars(value):
    """환경 변수 치환 함수"""
    if isinstance(value, str):
        # ${VAR_NAME} 패턴 찾기
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, value)
        
        # 각 환경 변수 치환
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value:
                value = value.replace(f"${{{var_name}}}", env_value)
        return value
    return value

def process_yaml_dict(yaml_dict):
    """중첩된 딕셔너리에서 환경 변수 치환"""
    result = {}
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            result[key] = process_yaml_dict(value)
        elif isinstance(value, list):
            result[key] = [process_yaml_dict(item) if isinstance(item, dict) else replace_env_vars(item) for item in value]
        else:
            result[key] = replace_env_vars(value)
    return result

def load_config(config_path):
    """설정 파일 로드 및 환경 변수 치환"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 치환 처리
    config = process_yaml_dict(config)
    return config

def check_s3_images(session, bucket, prefix):
    """S3 경로에 이미지 파일이 있는지 확인"""
    s3_client = session.boto_session.client('s3')
    
    # 이미지 파일 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.pt', '.txt'}
    
    try:
        # S3 객체 리스트 가져오기
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        # 이미지 파일 확인
        if 'Contents' not in response:
            return False
            
        for obj in response['Contents']:
            ext = os.path.splitext(obj['Key'])[1].lower()
            if ext in image_extensions:
                return True
        
        return False
    except Exception as e:
        print(f"Error checking S3 path: {str(e)}")
        return False

def run_training():
    # 설정 파일 로드
    config = load_config('configs/train_config.yaml')
    
    try:
        # SageMaker 세션 설정
        session = sagemaker.Session()
        role = os.environ.get('SAGEMAKER_ROLE_ARN')
        
        # S3 데이터 채널 설정
        s3_config = config['s3']
        bucket_name = s3_config['bucket_name']
        
        # 각 채널의 이미지 파일 존재 확인
        channels = {
            'train': f"{s3_config['prefix']}/{s3_config['train_prefix']}",
            'val': f"{s3_config['prefix']}/{s3_config['val_prefix']}",
            'test': f"{s3_config['prefix']}/{s3_config['test_prefix']}",
            'mask': f"{s3_config['prefix']}/{s3_config['mask_prefix']}"
        }
        
        # 각 채널 확인
        for channel_name, prefix in channels.items():
            if not check_s3_images(session, bucket_name, prefix):
                raise ValueError(f"No image files found in {channel_name} channel: s3://{bucket_name}/{prefix}")
        
        # 데이터 채널 구성
        data_channels = {
            name: f"s3://{bucket_name}/{prefix}"
            for name, prefix in channels.items()
        }
        
        # requirements.txt 파일 경로 설정
        requirements_path = Path(__file__).parent.parent / config['source_dir'] / 'requirements.txt'

        if not requirements_path.exists():
            raise FileNotFoundError(f"Requirements 파일을 찾을 수 없습니다: {requirements_path}")
        
        print(f"사용할 requirements 파일: {requirements_path}")
        
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        base_job_name = f"{config['base_job_name']}-{timestamp}"
        
        # MLflow 환경 변수 설정
        environment = {}
        if config.get('mlflow') and config['mlflow'].get('tracking_uri'):
            environment.update({
                'MLFLOW_TRACKING_URI': config['mlflow']['tracking_uri'],
                'MLFLOW_EXPERIMENT_NAME': config['mlflow'].get('experiment_name', 'default')
            })
        
        # Estimator 생성
        estimator = PyTorch(
            entry_point=config['entry_point'],
            source_dir=config['source_dir'],
            role=role,
            instance_count=config['instance_count'],
            instance_type=config['instance_type'],
            framework_version=config['framework_version'],
            py_version=config['py_version'],
            hyperparameters=config['hyperparameters'],
            base_job_name=base_job_name,
            sagemaker_session=session,
            output_path=f"s3://{bucket_name}/{s3_config['prefix']}/output",
            code_location=f"s3://{bucket_name}/{s3_config['prefix']}/code",
            requirements_file=str(requirements_path),
            environment=environment,
            tags=[{'Key': 'project', 'Value': 'pre-4team'}]
        )
        
        # 훈련 실행
        print("Starting training job...")
        estimator.fit(data_channels, wait=True)
        print("Training completed!")
        print(f"Model artifacts stored at: {estimator.model_data}")
        
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    run_training()
