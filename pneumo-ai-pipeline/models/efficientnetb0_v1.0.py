import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter  # 추가

# ---------------------------
# 모델 생성
# ---------------------------
def build_model(model_type: str, num_classes: int):
    mt = (model_type or "").lower()

    # torchvision 버전 호환: weights API 우선, 없으면 pretrained=True 폴백
    def _get_weights_safe(weights_name):
        try:
            return getattr(models, weights_name).DEFAULT
        except Exception:
            return None

    # efficientnet_b0
    if mt.startswith("efficientnet_b0") or mt in {"efficientnetb0", "efficientnet-b0"}:
        ctor = models.efficientnet_b0
        weights = _get_weights_safe("EfficientNet_B0_Weights")
        model = ctor(weights=weights) if weights is not None else ctor(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        tfm = weights.transforms() if weights is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return model, tfm

    # resnet50/IMAGENET1K_V1 같은 기존 표기도 지원
    if mt.startswith("resnet50"):
        ctor = models.resnet50
        weights = _get_weights_safe("ResNet50_Weights")
        if "imagenet1k_v1" in mt:
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V1
            except Exception:
                pass
        model = ctor(weights=weights) if weights is not None else ctor(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        tfm = weights.transforms() if weights is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return model, tfm

    raise ValueError(f"Unsupported model-type: {model_type}")

# ---------------------------
# 데이터 로더
# ---------------------------
def make_loaders(train_dir, val_dir, tfm, batch_size=32, num_workers=4):
    ds_train = datasets.ImageFolder(train_dir, transform=tfm)
    ds_val   = datasets.ImageFolder(val_dir,   transform=tfm)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, ds_train.classes

# ---------------------------
# 학습/검증 루프
# ---------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

# ---------------------------
# 메인
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker 채널 경로
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--val",   type=str, default="/opt/ml/input/data/val")
    # 하이퍼파라미터 (YAML hyperparameters 매핑)
    parser.add_argument("--epochs", type=int, default=int(os.getenv("SM_HP_EPOCHS", "1")))
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=int(os.getenv("SM_HP_BATCH_SIZE", "32")))
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=float(os.getenv("SM_HP_LEARNING_RATE", "1e-5")))
    parser.add_argument("--model-type", dest="model_type", type=str, default=os.getenv("SM_HP_MODEL_TYPE", "resnet50/IMAGENET1K_V1"))
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=int(os.getenv("SM_HP_NUM_CLASSES", "3")))
    parser.add_argument("--patience", type=int, default=int(os.getenv("SM_HP_PATIENCE", "10")))
    # TensorBoard 로그 디렉토리
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=os.path.join(os.environ.get("SM_OUTPUT_DATA_DIR", "./output"), "tensorboard")
    )
    # 모델 저장 디렉토리 (SageMaker 규약)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # 로그/모델 디렉토리 생성
    os.makedirs(args.tb_logdir, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=args.tb_logdir)
    writer.add_text("hparams", str(vars(args)))

    # 모델/변환
    model, tfm = build_model(args.model_type, args.num_classes)
    model = model.to(device)

    # 로더
    # GPU 환경일 때 num_workers를 4로, 아닐 때 0 권장(환경에 맞춰 조정 가능)
    num_workers = 4 if device.type == "cuda" else 0
    train_loader, val_loader, classes = make_loaders(args.train, args.val, tfm, batch_size=args.batch_size, num_workers=num_workers)

    # 옵티마이저/스케줄/로스
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 2)
    )

    best_acc = 0.0
    epochs_no_improve = 0
    try:
        for epoch in range(args.epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)
            scheduler.step(va_acc)

            # Console 출력
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

            # TensorBoard 로깅
            step = epoch + 1
            writer.add_scalar("Loss/train", tr_loss, step)
            writer.add_scalar("Loss/val",   va_loss, step)
            writer.add_scalar("Acc/train",  tr_acc, step)
            writer.add_scalar("Acc/val",    va_acc, step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

            # Best 모델 저장 & Early Stopping
            if va_acc > best_acc:
                best_acc = va_acc
                epochs_no_improve = 0
                torch.save({"state_dict": model.state_dict(), "classes": classes}, os.path.join(args.model_dir, "model_last.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print("Early stopping")
                    break
    finally:
        writer.close()
        print(f"TensorBoard logs saved to: {args.tb_logdir}")
        print(f"Model saved to: {args.model_dir}")

if __name__ == "__main__":
    main()