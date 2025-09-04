import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path
import os
import json
import tarfile
from collections import Counter

# NPZDataset 클래스는 resnet50_npz_v1.1.py에서 그대로 가져와야 합니다
from models.resnet50_npz_v1_1 import NPZDataset, create_model, get_device

def load_npz_weights(model, weight_path, device):
    """4채널 모델 가중치 로드"""
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

def create_evaluation_plots(y_true, y_score, class_names, output_path):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        y_score = y_score[:, None]
    if y_score.shape[1] == 1:
        pos = y_score
        y_score = np.concatenate([1.0 - pos, pos], axis=1)

    n_classes = y_score.shape[1]
    y_pred = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        bin_true = (y_true == i).astype(int)
        if bin_true.sum() == 0 or bin_true.sum() == bin_true.shape[0]:
            fpr[i], tpr[i], roc_auc[i] = None, None, np.nan
        else:
            fi, ti, _ = roc_curve(bin_true, y_score[:, i])
            fpr[i], tpr[i], roc_auc[i] = fi, ti, auc(fi, ti)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    precisions = [report.get(cls, {}).get('precision', 0.0) for cls in class_names]
    axes[0, 1].bar(class_names, precisions)
    axes[0, 1].set_title("Precision per Class")
    axes[0, 1].set_ylim([0, 1]); axes[0, 1].set_ylabel("Precision")

    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label="Chance")
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i in range(n_classes):
        if fpr.get(i) is None:
            continue
        axes[1, 0].plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                        label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    axes[1, 0].set_title("ROC Curve")
    axes[1, 0].set_xlabel("False Positive Rate"); axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc='lower right'); axes[1, 0].grid(True)

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

def main():
    # 경로 및 파라미터 설정
    model_dir = Path('./model')
    test_dir = Path('./test')
    weight_path = model_dir / 'model.pth'
    results_dir = Path('./results_npz')
    results_dir.mkdir(exist_ok=True)

    # 클래스명 (예시: 3클래스)
    class_names = ['normal', 'pneumonia', 'not_normal']

    # 디바이스
    device = get_device()

    # 데이터셋/로더 (4채널)
    test_dataset = NPZDataset(str(test_dir), with_labels=True, expect_channels=4,
                              image_keys=('image','img','x'), mask_keys=('mask','lung_mask','seg','msk'),
                              target_size=224)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 생성 및 가중치 로드
    model = create_model('resnet50/IMAGENET1K_V1', num_classes=len(class_names), device=device, in_channels=4)
    model = load_npz_weights(model, weight_path, device)

    # 평가
    y_true, y_score = evaluate_model(model, test_loader, device)

    # 시각화 및 메트릭 저장
    vis_path = results_dir / 'visualization.png'
    report, roc_auc = create_evaluation_plots(y_true, y_score, class_names, vis_path)
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\nClassification Report:")
    print(classification_report(y_true, np.argmax(y_score, axis=1), target_names=class_names, zero_division=0))

    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        v = roc_auc.get(i, np.nan)
        s = "nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
        print(f"{class_name}: {s}")

if __name__ == '__main__':
    main()