import os
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


# ----------------------------
# Args
# ----------------------------
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
    parser.add_argument('--model-type', type=str, default='resnet50/IMAGENET1K_V1')
    parser.add_argument('--patience', type=int, default=5)

    # TensorBoard 로그 디렉토리
    parser.add_argument('--tb-logdir', type=str,
                        default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', './output'), 'tensorboard'))
    return parser.parse_args()


# ----------------------------
# Transforms
# ----------------------------
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transforms, eval_transforms


# ----------------------------
# Dataset wrapper: ImageFolder -> binary label(0/1)
# 규칙(캐논 인덱스):
#   0: normal
#   1: pneumonia (또는 opacity)
#   2: abnormal (그 외)  → 본 스크립트에서는 무시
# 폴더명이 '0/1/2'이거나 'normal/pneumonia/abnormal'이어도 자동 인식
# ----------------------------
class BinaryImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # ImageFolder가 만든 (class_name -> orig_idx) 매핑
        self.idx2class = {v: k for k, v in self.class_to_idx.items()}

        # 원 인덱스 -> 캐논 인덱스(0/1/2) 매핑
        self.orig_to_canon = {}
        for orig_idx, cname in self.idx2class.items():
            self.orig_to_canon[orig_idx] = self._class_name_to_canon_idx(cname)

        # 0/1만 남기고 2(abnormal)는 제거
        filtered = []
        for path, orig_idx in self.samples:
            canon = self.orig_to_canon[orig_idx]
            if canon in (0, 1):
                # 라벨을 0/1로 재설정
                filtered.append((path, canon))
        self.samples = filtered
        self.targets = [y for _, y in self.samples]

        # 디버깅 출력
        print("== Original class_to_idx (ImageFolder) ==")
        print(self.class_to_idx)
        print("== Canonical index mapping (0:normal, 1:pneumonia, 2:abnormal) ==")
        for orig_idx, cname in self.idx2class.items():
            print(f"{cname} (orig_idx={orig_idx}) -> canon_idx={self.orig_to_canon[orig_idx]}")
        print(f"== Filtered to binary (kept only 0/1). Total: {len(self.samples)}")

    @staticmethod
    def _class_name_to_canon_idx(cname: str) -> int:
        lc = cname.lower()
        # 숫자 폴더명도 지원
        if lc in {'0', '1', '2'}:
            return int(lc)
        # 의미 기반 폴더명
        if 'normal' in lc:
            return 0
        if ('pneumonia' in lc) or ('opacity' in lc):
            return 1
        # 나머지는 abnormal
        return 2


# ----------------------------
# Model: ResNet50 backbone + single binary head
# ----------------------------
class BinaryResNet(nn.Module):
    def __init__(self, backbone_name='resnet50/IMAGENET1K_V1', dropout=0.2):
        super().__init__()
        if backbone_name == 'resnet50/IMAGENET1K_V1':
            base = models.resnet50(weights='IMAGENET1K_V1')
        elif backbone_name == 'resnet50/IMAGENET1K_V2':
            base = models.resnet50(weights='IMAGENET1K_V2')
        else:
            raise ValueError("지원하는 backbone: 'resnet50/IMAGENET1K_V1' 또는 'resnet50/IMAGENET1K_V2'")

        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)  # logit for pneumonia (1) vs normal (0)
        )

    def forward(self, x):
        f = self.backbone(x)
        logit = self.head(f)
        return logit


def create_model(model_type, device):
    model = BinaryResNet(backbone_name=model_type)
    return model.to(device)


# ----------------------------
# Loss helpers
# ----------------------------
def compute_pos_weight(positives: int, negatives: int) -> float:
    # BCEWithLogitsLoss pos_weight = neg/pos
    if positives == 0:
        return 1.0
    return float(negatives) / float(positives)


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, loader, optimz, criterion, device):
    model.train()
    loss_meter, acc_meter, n_total = 0.0, 0, 0

    with tqdm(loader, desc='Training') as pbar:
        for images, labels in pbar:
            images = images.to(device)
            y = labels.float().to(device).unsqueeze(1)  # [B,1]

            optimz.zero_grad()
            logit = model(images)  # [B,1]
            loss = criterion(logit, y)
            loss.backward()
            optimz.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).long().squeeze(1)  # [B]
                acc = (pred == labels.to(device).long()).sum().item()

            bsz = y.size(0)
            loss_meter += loss.item() * bsz
            acc_meter += acc
            n_total += bsz

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(acc_meter/n_total)*100:.2f}%'
            })

    return (loss_meter / n_total, 100.0 * acc_meter / n_total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc='Validation'):
    model.eval()
    loss_meter, acc_meter, n_total = 0.0, 0, 0

    with tqdm(loader, desc=desc) as pbar:
        for images, labels in pbar:
            images = images.to(device)
            y = labels.float().to(device).unsqueeze(1)
            logit = model(images)
            loss = criterion(logit, y)

            pred = (torch.sigmoid(logit) > 0.5).long().squeeze(1)
            acc = (pred == labels.to(device).long()).sum().item()

            bsz = y.size(0)
            loss_meter += loss.item() * bsz
            acc_meter += acc
            n_total += bsz

            pbar.set_postfix({
                'loss': f'{(loss_meter/n_total):.4f}',
                'acc': f'{(acc_meter/n_total)*100:.2f}%'
            })

    return (loss_meter / n_total, 100.0 * acc_meter / n_total)


# ----------------------------
# Device
# ----------------------------
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


# ----------------------------
# Build loaders + class imbalance handling
#   - WeightedRandomSampler: binary label 균형
#   - BCEWithLogitsLoss pos_weight 계산
# ----------------------------
def build_loaders_and_weights(args, device):
    train_t, eval_t = get_transforms()

    train_ds = BinaryImageFolder(args.train, transform=train_t)
    val_ds   = BinaryImageFolder(args.val,   transform=eval_t)
    test_ds  = BinaryImageFolder(args.test,  transform=eval_t)

    # binary 분포 집계 (train)
    labels = train_ds.targets  # 0/1
    cnt = Counter(labels)
    pos = cnt.get(1, 0)   # pneumonia
    neg = cnt.get(0, 0)   # normal
    print(f"[Train] (binary) pos/neg = {pos}/{neg}")

    if pos == 0 or neg == 0:
        raise RuntimeError(f"Binary labels not both present (pos={pos}, neg={neg}). "
                           f"폴더 구조/이름을 확인하세요 (정의: 0 normal, 1 pneumonia, 2 abnormal[무시]).")

    # Sampler: label 균형
    weights = np.array([1.0 / cnt[y] for y in labels], dtype=np.float32)
    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = 4 if device.type == 'cuda' else 0
    pin_mem = (device.type == 'cuda')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    # BCE pos_weight (for pneumonia=1)
    pw = compute_pos_weight(pos, neg)  # neg/pos
    return train_loader, val_loader, test_loader, pw


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.tb_logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = get_device()

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    # Data & weights
    train_loader, val_loader, test_loader, pos_weight = build_loaders_and_weights(args, device)

    # Model
    model = create_model(args.model_type, device)

    # Loss (BCEWithLogits)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # Optim / Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_logdir)
    writer.add_text('hparams', str(vars(args)))

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc='Validation')

        writer.add_scalar('Loss/train', tr_loss, epoch+1)
        writer.add_scalar('Loss/val',   val_loss, epoch+1)
        writer.add_scalar('Acc/train',  tr_acc,   epoch+1)
        writer.add_scalar('Acc/val',    val_acc,  epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

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
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc='Test')
    print(f"[Test] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    writer.add_scalar('Acc/test', test_acc, 0)
    writer.close()

    print(f"TensorBoard logs saved to: {args.tb_logdir}")
    print(f"Model saved to: {args.model_dir}")


if __name__ == '__main__':
    main()