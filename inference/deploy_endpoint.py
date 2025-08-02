import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import dotenv
import os
from datetime import datetime
import argparse
import yaml
import boto3
import tarfile
from pathlib import Path
import re

dotenv.load_dotenv()

# 전역 환경 변수 불러오기
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
timestamp = datetime.now().strftime('%y-%m-%d-%H-%M')

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

def compress_model(model_dir, output_path):
    """
    Compress the model directory into a tar.gz file
    
    Parameters:
    -----------
    model_dir : str
        Path to the directory containing the model files
    output_path : str
        Path where the compressed file will be saved
    
    Returns:
    --------
    str : Path to the compressed model file
    """
    with tarfile.open(output_path, "w:gz") as tar:
        for filename in os.listdir(model_dir):
            full_path = os.path.join(model_dir, filename)
            tar.add(full_path, arcname=filename)
    return output_path

def upload_model_to_s3(model_path, bucket_name, s3_key):
    """
    Upload the model.tar.gz file to S3
    
    Parameters:
    -----------
    model_path : str
        Local path to the model.tar.gz file
    bucket_name : str
        Name of the S3 bucket to upload to
    s3_key : str
        S3 key (path) where the model will be stored
    
    Returns:
    --------
    str : S3 URI of the uploaded model
    """

    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket_name, s3_key)
    return f's3://{bucket_name}/{s3_key}'

def deploy_model_to_sagemaker(config_path, role_arn=SAGEMAKER_ROLE_ARN):
    """
    Deploy the model to SageMaker
    
    Parameters:
    -----------
    model_data : str
        S3 URI of the model.tar.gz file
    role_arn : str, optional
        ARN of the IAM role to use for deployment. If None, will try to get the execution role
    
    Returns:
    --------
    predictor : sagemaker.pytorch.model.PyTorchPredictor
        Predictor object for the deployed model
    """
    
    # load configuration
    config = load_config(config_path)

    framework_version = config['Sagemaker'].get('framework_version', "1.13.1")
    py_version = config['Sagemaker'].get('py_version', 'py39')
    instance_type = config['Sagemaker'].get('instance_type', 'ml.m5.large')
    instance_count = config['Sagemaker'].get('instance_count', 1)
    key = config['Sagemaker'].get('key', 'project')
    value = config['Sagemaker'].get('value', 'pre-4team')

    if config['infernece_type'] == 'classification':
        # ─────────────────────────────────────────────────────────────
        # classification deployment
        # ───────────────────────────────────────────────────────────── 
        
        # define model directory
        s3_model_uri = f"s3://{BUCKET_NAME}/{config['S3']['output_path']}"

        # create PyTorchModel object
        pytorch_model = PyTorchModel(
            model_data=s3_model_uri,
            role=role_arn,
            entry_point="inference.py",
            source_dir="endpoint_classification",
            framework_version=framework_version,  # 사용 중인 PyTorch 버전과 맞추세요
            py_version=py_version,
            sagemaker_session=sagemaker.Session(),
            env={
                "HEATMAP_S3_BUCKET": BUCKET_NAME,
                "HEATMAP_S3_PREFIX": "sagemaker/test"
            }
        )

        # Deploy the model to an endpoint
        predictor = pytorch_model.deploy(
            endpoint_name=f'{config['base_job_name']}-{timestamp}',
            instance_type=instance_type,           # ← 인스턴스 타입 지정
            initial_instance_count=instance_count,               # ← 인스턴스 개수 지정
            tags=[{'Key': key, 'Value': value}],
            wait=True
        )
    elif config['infernece_type'] == 'segmentation':
        # ─────────────────────────────────────────────────────────────
        # segmentation deployment
        # ───────────────────────────────────────────────────────────── 
        
        # compress model directory
        files_dir = Path(__file__).parent / config['local']['files_dir']
        model_path = Path(__file__).parent / f'model-{timestamp}.tar.gz'
        model_path = compress_model(model_dir=files_dir, output_path=model_path)

        # upload model to S3
        s3_key = f'{config['S3']['prefix']}{config['base_job_name']}-{timestamp}.tar.gz'
        s3_model_uri = upload_model_to_s3(model_path, bucket_name=BUCKET_NAME, s3_key=s3_key)

        # create PyTorchModel object
        pytorch_model = PyTorchModel(
            model_data=s3_model_uri,
            role=role_arn,
            framework_version=framework_version,
            py_version=py_version,
            sagemaker_session=sagemaker.Session()
        )
        
        # Deploy the model to an endpoint
        predictor = pytorch_model.deploy(
            instance_type=instance_type,  # Choose an appropriate instance type
            initial_instance_count=instance_count,
            endpoint_name=f'{config['base_job_name']}-{timestamp}',
            tags=[{'Key': key, 'Value': value}]
        )
        if model_path.exists():
            model_path.unlink()
    else:
        raise ValueError("Invalid inference type specified in the configuration file. Use 'classification' or 'segmentation'.")

    return predictor


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Upload model.tar.gz file to a specified S3 bucket for SageMaker deployment"
    )
    parser.add_argument('--config', type=str, required=True, default='deploy_to_sagemaker.yaml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    # yaml path
    config_path = args.config
    
    # Deploy the model
    predictor = deploy_model_to_sagemaker(
        config_path=config_path,
        role_arn=SAGEMAKER_ROLE_ARN
    )
    print(f"Model deployed to endpoint: {predictor.endpoint_name}")
    
    # Example of how to use the endpoint for inference
    """
    import cv2
    import numpy as np
    import json
    import base64
    
    # Load and preprocess an image
    image = cv2.imread('test.jpg')
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # Create payload for SageMaker endpoint
    payload = {
        "image": list(img_bytes)  # Convert bytes to list for JSON serialization
    }
    
    # Call the endpoint
    response = predictor.predict(json.dumps(payload))
    
    # Process the response
    mask = np.array(json.loads(response)["mask"])
    
    # Save or display the result
    cv2.imwrite('output_mask.png', mask * 255)  # Save as binary image
    """
