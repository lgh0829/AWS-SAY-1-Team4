import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Add this import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path
import yaml
import os
import sys
import dotenv
import re
import tarfile
import torchvision.models as models
from torchvision import datasets, transforms
import json

# 현재 파일의 상위 디렉토리(프로젝트 루트)로 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.cloud_utils.s3_handler import S3Handler

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

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

def get_device():
    """
    사용 가능한 최적의 디바이스 선택
    1. CUDA GPU
    2. MPS (Apple Silicon)
    3. CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_model(model_type,num_classes, bucket_name, model_path, results_dir, device, weights=None):
    """
    S3에서 모델 가중치 로드 및 다운로드
    :param bucket_name: S3 버킷 이름
    :param model_path: S3 모델 경로
    :param results_dir: 로컬 결과 디렉토리
    :param device: torch 디바이스
    :param weights: 사전학습 가중치 옵션
    """
    
    # 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    
    # 가중치 파일 다운로드
    s3_handler = S3Handler(bucket_name)
    local_tar_path = results_dir / Path(model_path).name
    s3_handler.download_file(model_path, local_tar_path)

    # 압축 해제
    with tarfile.open(local_tar_path, 'r:gz') as tar:
        tar.extractall(path=results_dir)

    # 모델 로드
    model = get_model(model_type, num_classes, weights)

    # state_dict 불러오기
    local_dict_path = results_dir / 'model.pth'
    if not local_dict_path.exists():
        raise FileNotFoundError(f"모델 가중치 파일이 {local_dict_path}에 존재하지 않습니다.")
    state_dict = torch.load(local_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"모델 가중치가 {local_tar_path}에서 로드되었습니다.")
    
    return model

def evaluate_model(model, test_loader, device):
    """모델 평가 및 예측 수집"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

def create_evaluation_plots(y_true, y_score, class_names, output_path):
    """평가 결과 시각화 및 저장"""
    # 예측값 계산
    y_pred = np.argmax(y_score, axis=1)
    
    # Confusion Matrix 및 Classification Report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # ROC Curve 계산
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")
    
    # 2. Precision per Class
    precisions = [report[str(i)]['precision'] for i in range(n_classes)]
    axes[0, 1].bar(class_names, precisions, color='skyblue')
    axes[0, 1].set_title("Precision per Class")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_ylabel("Precision")
    
    # 3. ROC Curve
    colors = ['red', 'green', 'blue']
    for i in range(n_classes):
        axes[1, 0].plot(fpr[i], tpr[i], color=colors[i], lw=2,
                        label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1, 0].set_title("ROC Curve")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True)
    
    # 4. Confidence Histogram
    confidences = np.max(y_score, axis=1)
    axes[1, 1].hist(confidences, bins=20, color='orange', edgecolor='black')
    axes[1, 1].set_title("Prediction Confidence Histogram")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 1].set_ylabel("Frequency")
    
    # 저장
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return report, roc_auc

class ModelRegistry:
    """모델 레지스트리 클래스"""
    
    @staticmethod
    def get_resnet34(num_classes, weights=None):
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    
    @staticmethod
    def get_resnet50(num_classes, weights=None):
        if weights == "xrv":
            import torchxrayvision as xrv
            model = xrv.models.ResNet(weights="resnet50-res512-all")
            in_features = model.model.fc.in_features
            model.model.fc = torch.nn.Linear(in_features, num_classes)
            model.op_threshs = None
            model.n_outputs = num_classes
            model.pathologies = [f"class_{i}" for i in range(num_classes)]
        elif weights == "IMAGENET1K_V1":
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    
    @staticmethod
    def get_resnet18(num_classes, weights=None):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
        
    @staticmethod
    def get_efficientnet_b0(num_classes, weights=None):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    @staticmethod
    def get_mobilenet_v2(num_classes, weights=None):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    @staticmethod
    def get_densenet121(num_classes, weights=None):
        """
        weights:
            - "xrv": torchxrayvision pretrained weight 사용
            - None: torchvision 기본 모델 사용
        """
        if weights == "xrv":
            import torchxrayvision as xrv
            import os
            import urllib.request

            # torchxrayvision 모델 다운로드 및 로딩
            model_dir = os.path.expanduser("~/.torchxrayvision/models_data")
            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(
                model_dir,
                "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
            )

            if not os.path.exists(model_path):
                print(f"모델 파일 다운로드 중: {model_path}")
                url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
                urllib.request.urlretrieve(url, model_path)

            model = torch.load(model_path, map_location="cpu", weights_only=False)

            # classifier 교체
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
            model.n_outputs = num_classes
            return model
        
        else:
            # torchvision 기본 densenet121 사용
            from torchvision import models
            model = models.densenet121(weights=None)
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
            return model

def get_model(model_type: str, num_classes: int, weights: str = None):
    """
    모델 타입에 따라 해당하는 모델을 반환
    :param model_type: 모델 타입 문자열
    :param num_classes: 클래스 수
    :param weights: 사전학습 가중치 옵션
    :return: PyTorch 모델
    """
    model_map = {
        'resnet34': lambda nc, w: ModelRegistry.get_resnet34(nc, w),
        'resnet50': lambda nc, w: ModelRegistry.get_resnet50(nc, w),
        'resnet18': lambda nc, w: ModelRegistry.get_resnet18(nc, w),
        'efficientnet_b0': lambda nc, w: ModelRegistry.get_efficientnet_b0(nc, w),
        'mobilenet_v2': lambda nc, w: ModelRegistry.get_mobilenet_v2(nc, w),
        'densenet121': lambda nc, w: ModelRegistry.get_densenet121(nc, w)
    }
    
    model_fn = model_map.get(model_type.lower())
    if model_fn is None:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}. "
                       f"지원되는 모델: {list(model_map.keys())}")
    
    return model_fn(num_classes, weights)

def main():
    # 설정 로드
    config = load_config('configs/evaluation_config.yaml')
    
    # 디바이스 설정
    device = get_device()
    
    # 모델 로드
    try:
        bucket_name=config['s3']['bucket_name']
        model_path=f"{config['s3']['prefix']}/output/{config['s3']['job_name']}/output/model.tar.gz"
        results_dir=Path(__file__).parent.parent / f'{config["local"]["result_dir"]}/{config["s3"]["job_name"]}'
        weights = config['model'].get('pretrained_weight', None)
        
        model = load_model(
            model_type=config['model']['type'],
            num_classes=config['model']['num_classes'],
            bucket_name=bucket_name,
            model_path=model_path,
            results_dir=results_dir,
            device=device,
            weights=weights
        )
    except Exception as e:
        print(f's3://{bucket_name}/{model_path}')
        print(f"모델 로드 실패: {str(e)}")

        try: 
            # 로컬 모델 로드
            model = get_model(
                model_type=config['model']['type'],
                num_classes=config['model']['num_classes'],
                weights=weights
            ).to(device)
            print("로컬 모델 로드 성공")
        except Exception as e:
            print(f"로컬 모델 로드 실패: {str(e)}")
            return
    
    # 테스트 데이터 로드
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 표준 정규화 값
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_data_path = Path(__file__).parent.parent / config['local']['data']['data_dir'] / config['local']['data']['test_dir']
    if not test_data_path.exists():
        raise FileNotFoundError(f"테스트 데이터 디렉토리를 찾을 수 없습니다: {test_data_path}")

    test_dataset = datasets.ImageFolder(
        root=str(test_data_path),
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    print(f"클래스: {test_dataset.classes}")
    
    # 모델 평가
    y_true, y_score = evaluate_model(model, test_loader, device)
    
    # 결과 시각화 및 저장
    class_names = ['normal', 'pneumonia', 'not_normal']
    report, roc_auc = create_evaluation_plots(
        y_true, 
        y_score, 
        class_names,
        Path(f'{results_dir}/{config["local"]["output"]["visualization_path"]}')
    )
    
    # 결과 출력
    report = classification_report(y_true, np.argmax(y_score, axis=1), target_names=class_names, output_dict=True)
    with open(f'{results_dir}/{config["local"]["output"]["metrics_path"]}', "w") as f:
        json.dump(report, f, indent=4)
    print("\nClassification Report:")
    print(classification_report(y_true, np.argmax(y_score, axis=1)))
    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {roc_auc[i]:.3f}")

if __name__ == '__main__':
    main()
