import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from transformers import AutoModelForImageClassification
import pandas as pd
import glob
from typing import Optional, List, Dict, Any
import torch.nn as nn  # ModelRegistry에서 torch.nn 사용 중이라면 유지

# 현재 파일의 상위 디렉토리(프로젝트 루트)로 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.cloud_utils.s3_handler import S3Handler

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

def replace_env_vars(value):
    """환경 변수 치환 함수"""
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
    return process_yaml_dict(config)

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

def _strip_prefix_from_state_dict(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.")) -> Dict[str, torch.Tensor]:
    """
    DataParallel/Lightning 등으로 저장된 state_dict 키 앞의 접두사(module./model.) 제거
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        new_sd[new_k] = v
    return new_sd

def _find_weight_file(search_dir: Path) -> Optional[Path]:
    """
    검색 우선순위대로 가중치 파일 탐색
    """
    priority_patterns = [
        "model.pth",
        "model.pt",
        "best.pth",
        "best.pt",
        "checkpoint.pth",
        "checkpoint.pt",
        "pytorch_model.bin",  # HF 포맷
        "*.pth",
        "*.pt",
    ]
    for pat in priority_patterns:
        for p in search_dir.rglob(pat):
            # tensorboard 이벤트/기타 잡파일은 제외
            if "events.out.tfevents" in p.name:
                continue
            return p
    return None

def _load_weights_into_model(model: nn.Module, weight_path: Path, device: torch.device) -> nn.Module:
    """
    단일 파일(.pt/.pth/.bin)에서 모델/가중치를 로드
    - 전체 모델 저장(torch.save(model, ...))이면 그대로 로드
    - state_dict 저장(torch.save(model.state_dict(), ...))이면 load_state_dict
    - {'state_dict': ...} 형태도 지원
    """
    print(f"Loading weights from: {weight_path}")

    # HuggingFace 포맷인 경우(옵션)
    if weight_path.name == "pytorch_model.bin":
        # 디렉토리 전체가 HF 디렉토리인 경우가 많음
        try:
            from transformers import AutoModelForImageClassification
            hf_dir = str(weight_path.parent)
            print(f"Detected HF weights. Trying to load from directory: {hf_dir}")
            model = AutoModelForImageClassification.from_pretrained(
                hf_dir,
                ignore_mismatched_sizes=True
            )
            return model.to(device)
        except Exception as e:
            print(f"[WARN] HF-style load failed: {e}. Will try torch.load fallback.")

    obj = torch.load(str(weight_path), map_location=device)

    # 전체 모델 객체로 저장된 경우
    if isinstance(obj, nn.Module):
        print("Loaded a full torch.nn.Module object from .pt/.pth")
        return obj.to(device).eval()

    # state_dict 또는 dict wrapper로 저장된 경우
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unsupported dict format in weight file: keys={list(obj.keys())[:5]}")
        sd = _strip_prefix_from_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[INFO] Missing keys when loading (ignored): {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[INFO] Unexpected keys when loading (ignored): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")
        return model.to(device).eval()

    raise RuntimeError(f"Unsupported weight object type: {type(obj)}")

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, allowed_classes=None, **kwargs):
        self._allowed_classes = set(allowed_classes) if allowed_classes else None
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        classes, class_to_idx = super().find_classes(directory)
        if self._allowed_classes is not None:
            classes = [c for c in classes if c in self._allowed_classes]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def load_model(model_type, num_classes, bucket_name, model_path, results_dir, device, weights=None):
    """
    S3에서 모델 가중치 로드:
    - .tar.gz: 압축 해제 후 내부에서 가중치 파일 탐색
    - .pt/.pth: 단일 파일 다운로드 후 즉시 로딩
    - HF 형식 pytorch_model.bin: 필요 시 transformers로 로딩
    """
    os.makedirs(results_dir, exist_ok=True)
    s3_handler = S3Handler(bucket_name)
    local_path = results_dir / Path(model_path).name

    # 1) S3에서 다운로드
    print(f"Downloading from s3://{bucket_name}/{model_path} -> {local_path}")
    s3_handler.download_file(model_path, local_path)

    # 2) 압축 vs 단일 파일 분기
    extracted_root = results_dir
    if str(local_path).endswith(".tar.gz"):
        print(f"Extracting tar.gz: {local_path}")
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(path=results_dir)
        # 보편적으로 /opt/ml/model/ 혹은 같은 디렉토리에 풀림
        extracted_root = results_dir
    else:
        print("Single weight file detected (not tar.gz).")

    # 3) 가중치 파일 찾기
    weight_file = None
    if local_path.suffix in [".pt", ".pth", ".bin"]:
        weight_file = local_path
    else:
        weight_file = _find_weight_file(extracted_root)

    if weight_file is None:
        raise FileNotFoundError(
            f"No weight file found under {extracted_root}. "
            f"Expected something like model.pth, model.pt, best.pth, pytorch_model.bin"
        )

    # 4) 모델 생성
    print(f"Building model: type={model_type}, num_classes={num_classes}, weights-hint={weights}")
    model = get_model(model_type, num_classes, weights)

    # 5) 가중치 주입
    model = _load_weights_into_model(model, weight_file, device)
    print(f"Model loaded successfully from {weight_file}")
    return model

def evaluate_model(model, test_loader, device):
    """모델 평가 및 예측 수집 (torchvision/HF 호환, binary (N,1)도 안전)"""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            # HF ModelOutput 호환
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # 로짓 → 확률
            if logits.ndim == 2 and logits.shape[1] == 1:
                # 이진 (N,1) → (N,2)
                pos = torch.sigmoid(logits)
                probs = torch.cat([1.0 - pos, pos], dim=1)
            else:
                probs = F.softmax(logits, dim=1)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

def create_evaluation_plots(y_true, y_score, class_names, output_path):
    """평가 결과 시각화 및 저장(이진/다중 안전)"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 점수 정규화: (N,1) or (N,) → (N,2)
    if y_score.ndim == 1:
        y_score = y_score[:, None]
    if y_score.shape[1] == 1:
        pos = y_score
        y_score = np.concatenate([1.0 - pos, pos], axis=1)

    n_classes = y_score.shape[1]
    assert n_classes == len(class_names), f"class_names({len(class_names)})와 y_score.shape[1]({n_classes})가 다릅니다."

    # 예측 클래스
    y_pred = np.argmax(y_score, axis=1)

    # Confusion Matrix / Report
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # ===== ROC 계산 =====
    fpr, tpr, roc_auc = {}, {}, {}

    if n_classes == 2:
        # 클래스 0
        y0 = (y_true == 0).astype(int)
        s0 = y_score[:, 0]
        if y0.sum() == 0 or y0.sum() == y0.shape[0]:
            fpr[0], tpr[0], roc_auc[0] = None, None, np.nan
        else:
            f0, t0, _ = roc_curve(y0, s0)
            fpr[0], tpr[0] = f0, t0
            roc_auc[0] = auc(f0, t0)

        # 클래스 1
        y1 = (y_true == 1).astype(int)
        s1 = y_score[:, 1]
        if y1.sum() == 0 or y1.sum() == y1.shape[0]:
            fpr[1], tpr[1], roc_auc[1] = None, None, np.nan
        else:
            f1, t1, _ = roc_curve(y1, s1)
            fpr[1], tpr[1] = f1, t1
            roc_auc[1] = auc(f1, t1)

    else:
        # 다중 클래스: one-vs-rest
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            col = y_true_bin[:, i]
            if col.sum() == 0 or col.sum() == col.shape[0]:
                fpr[i], tpr[i], roc_auc[i] = None, None, np.nan
                continue
            fi, ti, _ = roc_curve(col, y_score[:, i])
            fpr[i], tpr[i] = fi, ti
            roc_auc[i] = auc(fi, ti)

    # ===== 시각화 =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    # 2) Precision per Class
    precisions = [report.get(cls, {}).get('precision', 0.0) for cls in class_names]
    axes[0, 1].bar(class_names, precisions)
    axes[0, 1].set_title("Precision per Class")
    axes[0, 1].set_ylim([0, 1]); axes[0, 1].set_ylabel("Precision")

    # 3) ROC Curve
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label="Chance")
    for i in range(n_classes):
        if fpr.get(i) is None:
            continue
        axes[1, 0].plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    axes[1, 0].set_title("ROC Curve")
    axes[1, 0].set_xlabel("False Positive Rate"); axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc='lower right'); axes[1, 0].grid(True)

    # 4) Confidence Histogram
    confidences = np.max(y_score, axis=1)
    axes[1, 1].hist(confidences, bins=20, edgecolor='black')
    axes[1, 1].set_title("Prediction Confidence Histogram")
    axes[1, 1].set_xlabel("Confidence"); axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return report, roc_auc

class ModelRegistry:
    """모델 레지스트리 클래스"""
    @staticmethod
    def get_resnet34(num_classes, weights=None):
        if weights == "IMAGENET1K_V1":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
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
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif weights == 'IMAGENET1K_V2':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, 1)
        elif weights == "microsoft/resnet-50":
            model = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
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
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
            model.n_outputs = num_classes
            return model
        else:
            from torchvision import models as tvm
            model = tvm.densenet121(weights=None)
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
            return model

def get_model(model_type: str, num_classes: int, weights: str = None):
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
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}. 지원되는 모델: {list(model_map.keys())}")
    return model_fn(num_classes, weights)

def plot_training_history(history_data, save_path):
    """학습 히스토리 시각화"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_data.get('train_loss', []), label='Training Loss')
    plt.plot(history_data.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_data.get('train_acc', []), label='Training Accuracy')
    plt.plot(history_data.get('val_acc', []), label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def load_training_logs(bucket_name, s3_path, local_dir):
    from tensorboard.backend.event_processing import event_accumulator
    s3_handler = S3Handler(bucket_name)
    local_tar_path = local_dir / Path(s3_path).name
    s3_handler.download_file(s3_path, local_tar_path)

    with tarfile.open(local_tar_path, 'r:gz') as tar:
        tar.extractall(path=local_dir)

    event_file = None
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                break
    if not event_file:
        raise FileNotFoundError("Tensorboard event file not found")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    print("[TB] available scalar tags:", available)

    prefer_tags = {
        'train_loss': ['Loss/train', 'loss/train', 'training_loss'],
        'val_loss':   ['Loss/val',   'loss/val',   'validation_loss'],
        'train_acc':  ['Acc/train',  'Accuracy/train', 'acc/train', 'training_accuracy'],
        'val_acc':    ['Acc/val',    'Accuracy/val',   'acc/val',   'validation_accuracy'],
    }
    history = {k: [] for k in prefer_tags.keys()}

    for hist_key, candidates in prefer_tags.items():
        tag = next((t for t in candidates if t in available), None)
        if tag:
            events = ea.Scalars(tag)
            history[hist_key] = [ev.value for ev in events]
        else:
            print(f"[WARN] tag for {hist_key} not found among {available}")

    return history

def main():
    # 설정 로드
    config = load_config('configs/evaluation_config.yaml')

    # 디바이스 설정
    device = get_device()

    # 결과 디렉토리
    results_dir = Path(__file__).parent.parent / f'{config["local"]["result_dir"]}/{config["s3"]["job_name"]}'
    os.makedirs(results_dir, exist_ok=True)

    # 모델 로드
    try:
        bucket_name = config['s3']['bucket_name']
        model_path = f"{config['s3']['prefix']}/output/{config['s3']['job_name']}/output/model.tar.gz"
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
        print(f's3://{config["s3"]["bucket_name"]}/{config["s3"]["prefix"]}/output/{config["s3"]["job_name"]}/output/model.tar.gz')
        print(f"모델 로드 실패: {str(e)}")
        try:
            model = get_model(
                model_type=config['model']['type'],
                num_classes=config['model']['num_classes'],
                weights=weights
            ).to(device)
            print("로컬 모델 로드 성공")
        except Exception as e2:
            print(f"로컬 모델 로드 실패: {str(e2)}")
            return

    # 테스트 데이터 로드
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data_path = Path(__file__).parent.parent / config['local']['data']['data_dir'] / config['local']['data']['test_dir']
    if not test_data_path.exists():
        raise FileNotFoundError(f"테스트 데이터 디렉토리를 찾을 수 없습니다: {test_data_path}")

    # 허용할 클래스 폴더 제한
    allowed_classes = None
    if config['model']['num_classes'] == 2:
        allowed_classes = {"0", "1"}

    test_dataset = FilteredImageFolder(
        root=str(test_data_path),
        transform=transform,
        allowed_classes=allowed_classes
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    print(f"클래스: {test_dataset.classes}")  # num_classes==2면 ['0','1']만 나와야 정상

    # 모델 평가
    y_true, y_score = evaluate_model(model, test_loader, device)

    # 결과 시각화 및 저장
    class_names = ['normal', 'pneumonia'] if config['model']['num_classes'] == 2 else ['normal', 'pneumonia', 'abnormal']
    vis_path = results_dir / f'{config["local"]["output"]["visualization_path"]}'
    report, roc_auc = create_evaluation_plots(y_true, y_score, class_names, vis_path)

    # 결과 출력/저장
    report_dict = classification_report(
        y_true, np.argmax(y_score, axis=1),
        target_names=class_names, output_dict=True, zero_division=0
    )
    metrics_path = results_dir / f'{config["local"]["output"]["metrics_path"]}'
    os.makedirs(metrics_path.parent, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    print("\nClassification Report:")
    print(classification_report(
        y_true, np.argmax(y_score, axis=1),
        target_names=class_names, zero_division=0
    ))

    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        v = roc_auc.get(i, np.nan)
        s = "nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
        print(f"{class_name}: {s}")

    # 학습 히스토리 로드 및 시각화
    try:
        training_logs_path = f"{config['s3']['prefix']}/output/{config['s3']['job_name']}/output/output.tar.gz"
        history = load_training_logs(
            bucket_name=config['s3']['bucket_name'],
            s3_path=training_logs_path,
            local_dir=results_dir
        )
        history_plot_path = results_dir / 'training_history.png'
        plot_training_history(history, history_plot_path)
        print(f"\n학습 히스토리 그래프가 저장되었습니다: {history_plot_path}")
    except Exception as e:
        print(f"\n학습 히스토리 로드 실패: {str(e)}")
        print("tensorboard 이벤트 파일 경로를 확인해주세요.")

if __name__ == '__main__':
    main()