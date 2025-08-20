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

    # 멀티태스크 손실 가중치
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for L_pneumonia')
    parser.add_argument('--beta',  type=float, default=1.0, help='weight for L_abnormal')

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
# Dataset wrapper: ImageFolder -> two binary labels
# 규칙(캐논 인덱스):
#   0: normal
#   1: pneumonia (또는 opacity)
#   2: abnormal (그 외)
# 폴더명이 '0/1/2'이거나 'normal/pneumonia/abnormal'이어도 자동 인식
# ----------------------------
class MultiTaskImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # 원래 ImageFolder의 class_to_idx (알파벳 순)
        # idx -> class name
        self.idx2class = {v: k for k, v in self.class_to_idx.items()}
        # 원 인덱스 -> 캐논 인덱스(0/1/2) 매핑
        self.orig_to_canon = {}
        for orig_idx, cname in self.idx2class.items():
            canon = self._class_name_to_canon_idx(cname)
            self.orig_to_canon[orig_idx] = canon

        # 디버깅용 출력
        print("== Original class_to_idx (ImageFolder) ==")
        print(self.class_to_idx)
        print("== Mapped to canonical indices (0:normal, 1:pneumonia, 2:abnormal) ==")
        for orig_idx, cname in self.idx2class.items():
            print(f"{cname} (orig_idx={orig_idx}) -> canon_idx={self.orig_to_canon[orig_idx]}")

    @staticmethod
    def _class_name_to_canon_idx(cname: str) -> int:
        """폴더명으로 캐논 인덱스 결정"""
        lc = cname.lower()
        # 숫자 폴더명도 지원
        if lc in {'0', '1', '2'}:
            return int(lc)
        # 의미 기반 폴더명
        if ('normal' in lc):
            return 0
        if ('pneumonia' in lc) or ('opacity' in lc):
            return 1
        # 나머지는 abnormal
        return 2

    def _to_two_labels(self, class_idx: int):
        """
        캐논 인덱스를 기준으로 이중 라벨 생성
        - 폐렴(pneumonia) 헤드: canon==1 -> 1 else 0
        - 비정상(abnormal) 헤드: canon==0(normal) -> 0 else 1
        """
        canon = self.orig_to_canon[class_idx]
        is_pneu = 1 if canon == 1 else 0
        is_abn  = 0 if canon == 0 else 1
        return is_pneu, is_abn

    def __getitem__(self, index):
        img, class_idx = super().__getitem__(index)
        pneu, abn = self._to_two_labels(class_idx)
        y = torch.tensor([pneu, abn], dtype=torch.float32)  # shape [2]
        return img, y


# ----------------------------
# Model: ResNet50 backbone + two binary heads
# ----------------------------
class MultiHeadResNet(nn.Module):
    def __init__(self, backbone_name='resnet50/IMAGENET1K_V1', dropout=0.2):
        super().__init__()
        if backbone_name == 'resnet50/IMAGENET1K_V1':
            base = models.resnet50(weights='IMAGENET1K_V1')
        elif backbone_name == 'resnet50/IMAGENET1K_V2':
            base = models.resnet50(weights='IMAGENET1K_V2')
        else:
            raise ValueError("지원하는 backbone: 'resnet50/IMAGENET1K_V1' 또는 'resnet50/IMAGENET1K_V2'")

        in_features = base.fc.in_features
        base.fc = nn.Identity()  # feature f

        self.backbone = base
        self.head_pneu = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)  # logit_pneumonia
        )
        self.head_abn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)  # logit_abnormal
        )

    def forward(self, x):
        f = self.backbone(x)  # [B, C]
        logit_pneumonia = self.head_pneu(f)  # [B,1]
        logit_abnormal  = self.head_abn(f)   # [B,1]
        return {
            "feat": f,
            "logit_pneumonia": logit_pneumonia,
            "logit_abnormal": logit_abnormal
        }


def create_model(model_type, device):
    model = MultiHeadResNet(backbone_name=model_type)
    return model.to(device)


# ----------------------------
# Loss helpers
# ----------------------------
def compute_pos_weight(positives: int, negatives: int) -> float:
    # pos_weight = neg/pos (BCEWithLogitsLoss convention)
    if positives == 0:
        return 1.0
    return float(negatives) / float(positives)


# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, loader, optimz, criterion_p, criterion_a, device, alpha, beta):
    model.train()
    loss_meter, acc_p_meter, acc_a_meter, n_total = 0.0, 0, 0, 0

    with tqdm(loader, desc='Training') as pbar:
        for images, y in pbar:
            images = images.to(device)
            y = y.to(device)  # [B,2]; [:,0]=pneumonia, [:,1]=abnormal

            optimz.zero_grad()
            out = model(images)
            logit_p = out['logit_pneumonia']  # [B,1]
            logit_a = out['logit_abnormal']   # [B,1]

            loss_p = criterion_p(logit_p, y[:, [0]])
            loss_a = criterion_a(logit_a, y[:, [1]])
            loss = alpha * loss_p + beta * loss_a

            loss.backward()
            optimz.step()

            # metrics
            with torch.no_grad():
                pred_p = (torch.sigmoid(logit_p) > 0.5).long().squeeze(1)  # [B]
                pred_a = (torch.sigmoid(logit_a) > 0.5).long().squeeze(1)
                acc_p = (pred_p == y[:, 0].long()).sum().item()
                acc_a = (pred_a == y[:, 1].long()).sum().item()

            bsz = y.size(0)
            loss_meter += loss.item() * bsz
            acc_p_meter += acc_p
            acc_a_meter += acc_a
            n_total += bsz

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc_p': f'{(acc_p_meter/n_total)*100:.2f}%',
                'acc_a': f'{(acc_a_meter/n_total)*100:.2f}%'
            })

    return (loss_meter / n_total,
            100.0 * acc_p_meter / n_total,
            100.0 * acc_a_meter / n_total)


@torch.no_grad()
def evaluate(model, loader, criterion_p, criterion_a, device, alpha, beta, desc='Validation'):
    model.eval()
    loss_meter, acc_p_meter, acc_a_meter, n_total = 0.0, 0, 0, 0

    with tqdm(loader, desc=desc) as pbar:
        for images, y in pbar:
            images = images.to(device)
            y = y.to(device)

            out = model(images)
            logit_p = out['logit_pneumonia']
            logit_a = out['logit_abnormal']

            loss_p = criterion_p(logit_p, y[:, [0]])
            loss_a = criterion_a(logit_a, y[:, [1]])
            loss = alpha * loss_p + beta * loss_a

            pred_p = (torch.sigmoid(logit_p) > 0.5).long().squeeze(1)
            pred_a = (torch.sigmoid(logit_a) > 0.5).long().squeeze(1)
            acc_p = (pred_p == y[:, 0].long()).sum().item()
            acc_a = (pred_a == y[:, 1].long()).sum().item()

            bsz = y.size(0)
            loss_meter += loss.item() * bsz
            acc_p_meter += acc_p
            acc_a_meter += acc_a
            n_total += bsz

            pbar.set_postfix({
                'loss': f'{(loss_meter/n_total):.4f}',
                'acc_p': f'{(acc_p_meter/n_total)*100:.2f}%',
                'acc_a': f'{(acc_a_meter/n_total)*100:.2f}%'
            })

    return (loss_meter / n_total,
            100.0 * acc_p_meter / n_total,
            100.0 * acc_a_meter / n_total)


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
#   - WeightedRandomSampler: pneumonia 라벨 기준으로 균형
#   - BCEWithLogitsLoss pos_weight: pneumonia/abnormal 각각 계산
# ----------------------------
def build_loaders_and_weights(args, device):
    train_t, eval_t = get_transforms()

    train_ds = MultiTaskImageFolder(args.train, transform=train_t)
    val_ds   = MultiTaskImageFolder(args.val,   transform=eval_t)
    test_ds  = MultiTaskImageFolder(args.test,  transform=eval_t)

    # pneumonia/abnormal 분포 집계 (train)
    pneu_labels, abn_labels = [], []
    for _, class_idx in train_ds.samples:
        pneu, abn = train_ds._to_two_labels(class_idx)
        pneu_labels.append(pneu)
        abn_labels.append(abn)

    pneu_pos = int(np.sum(pneu_labels))
    pneu_neg = len(pneu_labels) - pneu_pos
    abn_pos  = int(np.sum(abn_labels))
    abn_neg  = len(abn_labels) - abn_pos

    print(f"[Train] Pneumonia pos/neg = {pneu_pos}/{pneu_neg}")
    print(f"[Train] Abnormal  pos/neg = {abn_pos}/{abn_neg}")

    # sanity check: 두 헤드 모두 양/음성 존재해야 함
    if pneu_pos == 0 or pneu_neg == 0:
        raise RuntimeError(f"Pneumonia labels not both present (pos={pneu_pos}, neg={pneu_neg}). "
                           f"폴더 구조/이름을 확인하세요 (정의: 0 normal, 1 pneumonia, 2 abnormal).")
    if abn_pos == 0 or abn_neg == 0:
        raise RuntimeError(f"Abnormal labels not both present (pos={abn_pos}, neg={abn_neg}).")

    # Sampler: pneumonia 기준으로 binary class balancing
    pneu_class_count = Counter(pneu_labels)  # {0: neg, 1: pos}
    weights = np.array([1.0 / pneu_class_count[l] for l in pneu_labels], dtype=np.float32)
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

    # BCE pos_weight
    pw_pneu = compute_pos_weight(pneu_pos, pneu_neg)  # scalar
    pw_abn  = compute_pos_weight(abn_pos, abn_neg)

    return train_loader, val_loader, test_loader, pw_pneu, pw_abn


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
    train_loader, val_loader, test_loader, pw_pneu, pw_abn = build_loaders_and_weights(args, device)

    # Model
    model = create_model(args.model_type, device)

    # Losses (BCEWithLogits)
    criterion_p = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_pneu], device=device))
    criterion_a = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_abn],  device=device))

    # Optim / Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_logdir)
    writer.add_text('hparams', str(vars(args)))

    best_val_score = 0.0  # mean of two-head accuracies
    patience_counter = 0

    for epoch in range(args.epochs):
        tr_loss, tr_acc_p, tr_acc_a = train_epoch(
            model, train_loader, optimizer, criterion_p, criterion_a, device,
            alpha=args.alpha, beta=args.beta
        )
        val_loss, val_acc_p, val_acc_a = evaluate(
            model, val_loader, criterion_p, criterion_a, device,
            alpha=args.alpha, beta=args.beta, desc='Validation'
        )

        mean_tr_acc  = 0.5 * (tr_acc_p + tr_acc_a)
        mean_val_acc = 0.5 * (val_acc_p + val_acc_a)

        writer.add_scalar('Loss/train', tr_loss, epoch+1)
        writer.add_scalar('Loss/val',   val_loss, epoch+1)
        writer.add_scalar('AccP/train', tr_acc_p, epoch+1)
        writer.add_scalar('AccA/train', tr_acc_a, epoch+1)
        writer.add_scalar('AccP/val',   val_acc_p, epoch+1)
        writer.add_scalar('AccA/val',   val_acc_a, epoch+1)
        writer.add_scalar('AccMean/val', mean_val_acc, epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {tr_loss:.4f} | AccP: {tr_acc_p:.2f}% | AccA: {tr_acc_a:.2f}% | Mean: {mean_tr_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | AccP: {val_acc_p:.2f}% | AccA: {val_acc_a:.2f}% | Mean: {mean_val_acc:.2f}%")

        # Early stopping & best model 저장 (두 헤드 평균 정확도 기준)
        if mean_val_acc > best_val_score:
            best_val_score = mean_val_acc
            patience_counter = 0
            model_path = os.path.join(args.model_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(mean_val_acc)

    # Test
    test_loss, test_acc_p, test_acc_a = evaluate(
        model, test_loader, criterion_p, criterion_a, device,
        alpha=args.alpha, beta=args.beta, desc='Test'
    )
    mean_test_acc = 0.5 * (test_acc_p + test_acc_a)
    print(f"[Test] Loss: {test_loss:.4f} | AccP: {test_acc_p:.2f}% | AccA: {test_acc_a:.2f}% | Mean: {mean_test_acc:.2f}%")
    writer.add_scalar('AccP/test', test_acc_p, 0)
    writer.add_scalar('AccA/test', test_acc_a, 0)
    writer.add_scalar('AccMean/test', mean_test_acc, 0)
    writer.close()

    print(f"TensorBoard logs saved to: {args.tb_logdir}")
    print(f"Model saved to: {args.model_dir}")


if __name__ == '__main__':
    main()