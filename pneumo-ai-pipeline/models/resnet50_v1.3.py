import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForImageClassification
import numpy as np
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker 기본 인자
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './train'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', './val'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', './test'))

    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--model-type', type=str, default='resnet50')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5)

    # TensorBoard 로그 디렉토리 (기본: SageMaker 출력 경로)
    parser.add_argument('--tb-logdir', type=str,
                        default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', './output'), 'tensorboard'))
    return parser.parse_args()

def get_transforms():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transforms, data_transforms

def create_model(model_type, num_classes, device):
    if model_type == "resnet50/microsoft":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return model.to(device)
    elif model_type == "resnet50/IMAGENET1K_V1":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    with tqdm(train_loader, desc='Training') as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            raw_output = model(images)
            outputs = raw_output.logits if hasattr(raw_output, "logits") else raw_output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            raw_output = model(images)
            outputs = raw_output.logits if hasattr(raw_output, "logits") else raw_output
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100. * correct / total

def get_device():
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

def main():
    args = parse_args()
    os.makedirs(args.tb_logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = get_device()

    # CUDA 최적화 설정
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    # 데이터
    train_transforms, data_transforms = get_transforms()
    train_dataset = datasets.ImageFolder(args.train, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(args.val,   transform=data_transforms)
    test_dataset  = datasets.ImageFolder(args.test,  transform=data_transforms)

    # 클래스 불균형 샘플러
    targets = torch.tensor(train_dataset.targets)
    class_count = Counter(train_dataset.targets)
    class_sample_count = torch.tensor([class_count[i] for i in range(len(class_count))], dtype=torch.float)
    weights = 1. / class_sample_count
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Class distribution: {class_count}")

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))

    # 모델/옵티마이저
    model = create_model(args.model_type, args.num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_logdir)
    # 하이퍼파라미터 기록
    writer.add_text('hparams', str(vars(args)))

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Val
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/val',   val_loss,   epoch+1)
        writer.add_scalar('Acc/train',  train_acc,  epoch+1)
        writer.add_scalar('Acc/val',    val_acc,    epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Early stopping & best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(args.model_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_acc)

    # Test
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    writer.add_scalar('Acc/test', test_acc, 0)
    writer.close()

    # 주의: SageMaker는 /opt/ml/model 과 /opt/ml/output 하위 내용을 자동으로 S3에 업로드합니다.
    # - 모델: SM_MODEL_DIR -> 모델 아티팩트 S3
    # - 텐서보드 로그: SM_OUTPUT_DATA_DIR/tensorboard -> OutputDataConfig S3 (학습 종료 후)
    print(f"TensorBoard logs saved to: {args.tb_logdir}")
    print(f"Model saved to: {args.model_dir}")

if __name__ == '__main__':
    main()