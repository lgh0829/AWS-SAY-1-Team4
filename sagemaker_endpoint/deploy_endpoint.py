import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from time import gmtime, strftime
import dotenv
import os
from pathlib import Path
import re
import yaml
import json
import tarfile
import argparse
import sys

# 환경 변수 로드
dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent.parent / '.env')

def replace_env_vars(value):
    if isinstance(value, str):
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value:
                value = value.replace(f"${{{var_name}}}", env_value)
        return value
    return value

def process_yaml_dict(yaml_dict):
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = process_yaml_dict(config)
    return config

def deploy_endpoint(config=None, model_config=None, endpoint_name=None):
    if config is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'deploy_config.yaml'
        config = load_config(config_path)
        if not config:
            raise ValueError(f"설정 파일을 로드할 수 없습니다. 경로를 확인하세요.\nconfig_path: {config_path}")

    sagemaker_session = sagemaker.Session()
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    region = sagemaker_session.boto_region_name
    bucket_name = config['s3']['bucket_name']

    timestamp = strftime('%y-%m-%d-%H-%M-%S', gmtime())

    if not endpoint_name:
        endpoint_name = f"pre-4team-{timestamp}"

    model_s3_path = f"s3://{bucket_name}/{config['s3']['prefix']}/output/{config['s3']['job_name']}/output/model.tar.gz"

    print(f"설정 정보:")
    print(f"- S3 버킷: {bucket_name}")
    print(f"- 모델 경로: {model_s3_path}")
    print(f"- 엔드포인트 이름: {endpoint_name}")
    print(f"- 모델 타입: {config['model']['type']}")
    print(f"- 프레임워크 버전: {config['framework_version']}")

    # code 디렉토리 하위에 temp 디렉토리 생성
    code_dir = Path(__file__).parent.parent / 'code'
    temp_dir = code_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    model_dir = temp_dir / 'model'
    model_dir.mkdir(exist_ok=True)

    # # 기존 평가에서 사용한 model.pth 파일 경로
    # eval_results_dir = Path(__file__).parent.parent / f'{config["local"]["result_dir"]}/{config["s3"]["job_name"]}'
    # existing_model_path = eval_results_dir / 'model.pth'

    # if existing_model_path.exists():
    #     print(f"기존 model.pth 파일을 재사용합니다: {existing_model_path}")

    # if not existing_model_path.exists():
    #     raise FileNotFoundError(f"기존 model.pth 파일을 찾을 수 없습니다: {existing_model_path}")

    # 새로운 deploy_model.tar.gz 생성
    deploy_model_path = temp_dir / 'deploy_model.tar.gz'
    with tarfile.open(deploy_model_path, 'w:gz') as tar:
        # 기존 평가에서 사용한 model.pth 파일 확인
        eval_results_dir = Path(__file__).parent.parent / f'{config["local"]["result_dir"]}/{config["s3"]["job_name"]}'
        existing_model_path = eval_results_dir / 'model.pth'
        if existing_model_path.exists():
            print(f"기존 model.pth 파일을 재사용합니다: {existing_model_path}")
            tar.add(existing_model_path, arcname='model.pth')
        else:
            raise FileNotFoundError(f"기존 model.pth 파일을 찾을 수 없습니다: {existing_model_path}\n 가중치 다운로드부터 먼저 실행하세요.")
        
        # 코드 파일 추가
        if (code_dir / 'inference.py').exists():
            tar.add(code_dir / 'inference.py', arcname='code/inference.py')
        else:
            raise FileNotFoundError(f"inference.py 파일을 찾을 수 없습니다: {code_dir / 'inference.py'}")

        # model_factory.py 추가
        if (code_dir / 'model_factory.py').exists():
            tar.add(code_dir / 'model_factory.py', arcname='code/model_factory.py')
        else:
            raise FileNotFoundError(f"model_factory.py 파일을 찾을 수 없습니다: {code_dir / 'model_factory.py'}")
        
        # config.properties 추가
        if (code_dir / 'config.properties').exists():
            tar.add(code_dir / 'config.properties', arcname='code/config.properties')
        else:
            raise FileNotFoundError(f"config.properties 파일을 찾을 수 없습니다: {code_dir / 'config.properties'}")

        # model_config.json 업데이트 및 추가
        model_config = {
            "model": {
                "type": config['model'].get('type', 'resnet34'),
                "num_classes": config['model'].get('num_classes', 3),
                "pretrained_weight": config['model'].get('pretrained_weight', 'IMAGENET1K_V1')
            }
        }
        model_config_path = code_dir / 'model_config.yaml'
        with open(model_config_path, 'w') as f:
            yaml.safe_dump(model_config, f, default_flow_style=False)
        print(f"model_config.yaml 파일이 업데이트되었습니다: {model_config_path}")
        tar.add(model_config_path, arcname='code/model_config.yaml')

        # preprocess_config.yaml 업데이트 및 추가
        preprocess_config = {
            'preprocessing': {
                'steps': {
                    'convert_grayscale': config['preprocessing']['steps'].get('convert_grayscale', True),
                    'apply_clahe': config['preprocessing']['steps'].get('apply_clahe', True),
                    'apply_gaussian_blur': config['preprocessing']['steps'].get('apply_gaussian_blur', True),
                    'apply_min_max_stretch': config['preprocessing']['steps'].get('apply_min_max_stretch', True),
                    'apply_sharpening': config['preprocessing']['steps'].get('apply_sharpening', True),
                    'convert_to_rgb': config['preprocessing']['steps'].get('convert_to_rgb', True),
                },
                'params': {
                    'clahe': {
                        'clip_limit': config['preprocessing']['params']['clahe'].get('clip_limit', 2.0),
                        'grid_size': config['preprocessing']['params']['clahe'].get('grid_size', (8, 8))
                    },
                    'gaussian_blur': {
                        'radius': config['preprocessing']['params']['gaussian_blur'].get('radius', 2)
                    },
                    'min_max_stretch': {
                        'lower_percentile': config['preprocessing']['params']['min_max_stretch'].get('lower_percentile', 1),
                        'upper_percentile': config['preprocessing']['params']['min_max_stretch'].get('upper_percentile', 99)
                    },
                    'sharpening': {
                        'radius': config['preprocessing']['params']['sharpening'].get('radius', 2),
                        'percent': config['preprocessing']['params']['sharpening'].get('percent', 150),
                        'threshold': config['preprocessing']['params']['sharpening'].get('threshold', 3)
                    }
                }
            }
        }
        preprocess_config_path = code_dir / 'preprocess_config.yaml'
        with open(preprocess_config_path, 'w') as f:
            yaml.safe_dump(preprocess_config, f, default_flow_style=False)
        print(f"preprocess_config.yaml 파일이 업데이트되었습니다: {preprocess_config_path}")
        tar.add(preprocess_config_path, arcname='code/preprocess_config.yaml')

        # transform_config.yaml 업데이트 및 추가
        transform_config = {
            'image_resize': config.get('image_resize', [224, 224]),
            'image_normalize': {
                'mean': config.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
                'std': config.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
            }
        }
        transform_config_path = code_dir / 'transform_config.yaml'
        with open(transform_config_path, 'w') as f:
            yaml.safe_dump(transform_config, f, default_flow_style=False)
        print(f"transform_config.yaml 파일이 업데이트되었습니다: {transform_config_path}")
        tar.add(transform_config_path, arcname='code/transform_config.yaml')
            

        # requirements.txt 추가 (존재하는 경우)
        requirements_path = code_dir / 'requirements.txt'
        if requirements_path.exists():
            tar.add(requirements_path, arcname='code/requirements.txt')
        else:
            print(f"requirements.txt 파일이 존재하지 않습니다: {requirements_path}\n requirements.txt 파일 저장을 생략합니다.")

        # lung_seg 디렉토리 추가
        lung_seg_dir = Path(__file__).parent.parent / 'common/pneumo_utils/lungs_seg'
        tar.add(lung_seg_dir / 'inference.py', arcname='code/lung_seg/inference.py')
        tar.add(lung_seg_dir / 'pre_trained_models.py', arcname='code/lung_seg/pretrained_models.py')
        tar.add(lung_seg_dir / 'unet.py', arcname='code/lung_seg/unet.py')
        tar.add(lung_seg_dir / 'densenet.py', arcname='code/lung_seg/densenet.py')
        tar.add(lung_seg_dir / 'resnet.py', arcname='code/lung_seg/resnet.py')
        tar.add(lung_seg_dir / 'resunets.py', arcname='code/lung_seg/resunets.py')
        tar.add(lung_seg_dir / 'senet.py', arcname='code/lung_seg/senet.py')
        init_file = lung_seg_dir / '__init__.py'
        if init_file.exists():
            tar.add(init_file, arcname='code/lung_seg/__init__.py')
        else:
            open(init_file, 'a').close()  # 비어 있는 파일 생성
            tar.add(init_file, arcname='code/lung_seg/__init__.py')
    
    print(f"새로운 deploy_model.tar.gz 파일이 생성되었습니다: {deploy_model_path}")

    # 3. deploy_model.tar.gz를 S3에 업로드
    s3_prefix = f"{config['s3']['prefix']}/deploy"
    s3_key = f"{s3_prefix}/model-{endpoint_name}.tar.gz"
    s3_client.upload_file(str(deploy_model_path), bucket_name, s3_key)
    deploy_model_s3_path = f"s3://{bucket_name}/{s3_key}"

    # PyTorchModel 생성 시 새로운 모델 경로 사용
    pytorch_model = PyTorchModel(
        model_data=deploy_model_s3_path,  # 새로 생성한 model.tar.gz 사용
        role=role,
        entry_point='inference.py',
        source_dir=None,  # model.tar.gz에 모든 파일이 포함되어 있으므로 source_dir 불필요
        framework_version=config['framework_version'],
        py_version=config['py_version']
    )

    memory_size = config.get('serverless', {}).get('memory_size_in_mb', 6144)
    max_concurrency = config.get('serverless', {}).get('max_concurrency', 5)

    # instance_type = config.get('realtime', {}).get('instance_type', 'ml.g4dn.xlarge')
    # instance_count = config.get('realtime', {}).get('instance_count', 1)
    # print(f"배포할 인스턴스 타입: {instance_type}, 인스턴스 개수: {instance_count}")
    # print(f"type: {type(instance_type)}, count: {type(instance_count)}")

    try:
        # predictor = pytorch_model.deploy(
        #     endpoint_name=endpoint_name,
        #     instance_type=instance_type,
        #     instance_count=instance_count,
        #     tags=[
        #         {'Key': 'project', 'Value': 'pre-4team'},
        #         {'Key': 'model', 'Value': config['model']['type']},
        #         {'Key': 'job_name', 'Value': config['s3']['job_name']}
        #     ],
        #     env={
        #         # "TS_CONFIG_FILE": "/opt/ml/model/code/config.properties"
        #     }
        # )
        predictor = pytorch_model.deploy(
            endpoint_name=endpoint_name,
            serverless_inference_config=sagemaker.serverless.ServerlessInferenceConfig(
                memory_size_in_mb=memory_size,
                max_concurrency=max_concurrency,
            ),
            tags=[
                {'Key': 'project', 'Value': 'pre-4team'},
                {'Key': 'model', 'Value': config['model']['type']},
                {'Key': 'job_name', 'Value': config['s3']['job_name']}
            ],
            env={
                "TS_CONFIG_FILE": "/opt/ml/model/code/config.properties"
            }
        )
        print(f"새 엔드포인트 '{endpoint_name}'가 성공적으로 생성되었습니다.")

    except Exception as e:
        print(f"엔드포인트 배포 중 오류가 발생했습니다: {e}")
        raise

    print(f"SageMaker 서버리스 엔드포인트 [{endpoint_name}] 배포/업데이트를 시작했습니다.")
    print("배포 완료까지 몇 분 정도 소요될 수 있습니다.")
    return endpoint_name

def parse_args():
    parser = argparse.ArgumentParser(description='SageMaker 엔드포인트 배포')
    parser.add_argument('--config', type=str, default='configs/deploy_config.yaml', help='설정 파일 경로')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        config_path = Path(__file__).parent.parent / args.config
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    endpoint_name = deploy_endpoint(config=config)
    print(f"배포된 엔드포인트 이름: {endpoint_name}")