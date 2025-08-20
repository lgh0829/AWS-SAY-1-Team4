import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import mlflow
from pathlib import Path
import kornia.augmentation as K
import kornia.geometry.transform as Kgeom
import kornia as knn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--model-type', type=str, default='resnet50')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--augment', type=lambda x: x.lower() == 'true', default=False)
    return parser.parse_args()


class NpzDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = list(Path(root_dir).rglob('*.npz'))
        self.augment = augment
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        

        if self.augment:
            self.aug = nn.Sequential(
                K.RandomRotation(15.0, p=0.5),
                K.RandomAffine(degrees=0.0, translate=(0.1, 0.1), p=0.5),
                K.RandomHorizontalFlip(p=0.5)
                # K.RandomGaussianBlur((3, 3), sigma=(0.1, 1.0), p=0.2),
                # K.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
                
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = str(self.samples[idx])
        data = np.load(path)['data']  # [4, H, W]
        tensor = torch.tensor(data, dtype=torch.float32)  # [4, H, W]

        # ✅ Resize to 224x224 (before normalization/augmentation)
        tensor = Kgeom.resize(tensor.unsqueeze(0), (224, 224), interpolation='bilinear').squeeze(0)  # [4, 224, 224]

        rgb = (tensor[:3] - self.mean) / self.std  # normalize RGB
        mask = tensor[3:].clone()  # mask는 0,1 유지
        combined = torch.cat([rgb, mask], dim=0)  # [4, 224, 224]

        if self.augment:
            combined = self.aug(combined.unsqueeze(0)).squeeze(0)

        label = self.get_label_from_path(path)
        return combined, label



    def get_label_from_path(self, path):
        if "/0/" in path:
            return 0
        elif "/1/" in path:
            return 1
        else:
            return 2


def create_model(model_type, num_classes, device):
    if model_type == 'resnet50':
        # 1. ImageNet pretrained 모델 로드
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 2. 기존 conv1 백업
        old_conv = model.conv1

        # 3. 새 4채널 conv1 생성
        model.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 4. 가중치 초기화: RGB는 기존 weight 복사, mask 채널은 He 초기화
        with torch.no_grad():
            w_old = old_conv.weight  # [64, 3, 7, 7]
            w_new = model.conv1.weight  # [64, 4, 7, 7]
            w_new[:, :3, :, :] = w_old  # RGB 복사
            nn.init.kaiming_normal_(w_new[:, 3:, :, :], mode='fan_out', nonlinearity='relu')  # mask 채널

        # 5. fc 레이어 수정
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)

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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    running_loss = 0.0

    with tqdm(loader, desc='Train') as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

            # 실시간 출력
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    

    with torch.no_grad():
        with tqdm(loader, desc='Val') as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                # ✅ Softmax로 confidence 계산
                probs = F.softmax(out, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                total_loss += loss.item()
                correct += preds.eq(y).sum().item()
                total += y.size(0)

               

                acc = 100. * correct / total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    overall_acc = 100. * correct / total

    return total_loss / len(loader), overall_acc

def validate_with_confidence(model, loader, criterion, device, threshold=0.6):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    conf_correct, conf_total = 0, 0

    with torch.no_grad():
        with tqdm(loader, desc='Test') as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                probs = F.softmax(out, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                total_loss += loss.item()
                correct += preds.eq(y).sum().item()
                total += y.size(0)

                # ✅ confidence > 0.6 subset 정확도
                mask = max_probs >= threshold
                if mask.any():
                    conf_correct += preds[mask].eq(y[mask]).sum().item()
                    conf_total += mask.sum().item()

                acc = 100. * correct / total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

    overall_acc = 100. * correct / total
    subset_acc = 100. * conf_correct / conf_total if conf_total > 0 else 0.0

    print(f"\n🔍 Test Subset Accuracy (confidence > {threshold}): {subset_acc:.2f}% ({conf_total} samples)")

    return total_loss / len(loader), overall_acc, subset_acc


def main():
    args = parse_args()
    device = get_device()

    train_dataset = NpzDataset(args.train, augment=True)
    val_dataset = NpzDataset(args.val, augment=False)
    test_dataset = NpzDataset(args.test, augment=False)

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)

    model = create_model(args.model_type, args.num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        best_acc = 0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                'train_loss': train_loss, 'train_accuracy': train_acc,
                'val_loss': val_loss, 'val_accuracy': val_acc
                }, step=epoch)

            print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                path = os.path.join(args.model_dir, 'model.pth')
                torch.save(model.state_dict(), path)
                mlflow.log_artifact(path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping.")
                    break

            scheduler.step(val_acc)
            
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
        test_loss, test_acc, test_subset_acc = validate_with_confidence(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"🔍 Test Subset Accuracy (confidence ≥ 0.6): {test_subset_acc:.2f}%")
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.log_metric('test_conf_accuracy', test_subset_acc)


if __name__ == '__main__':
    main()