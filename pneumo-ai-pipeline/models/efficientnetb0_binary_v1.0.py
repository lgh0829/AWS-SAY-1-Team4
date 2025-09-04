import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# ---------------------- Utils ----------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def imagefolder_keep_classes(root, transform, keep=('0','1')):
    ds = datasets.ImageFolder(root, transform=transform)
    keep_ids = {ds.class_to_idx[k] for k in keep if k in ds.class_to_idx}
    keep_idx = [i for i, y in enumerate(ds.targets) if y in keep_ids]
    print(f"[INFO] class_to_idx={ds.class_to_idx} -> keep={keep} -> kept {len(keep_idx)} / {len(ds.targets)}")
    return Subset(ds, keep_idx)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps"); print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu"); print("Using CPU")
    return device

def find_best_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thr[best_idx])

# ---------------------- Transforms ----------------------
def get_transforms(image_size):
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

# ---------------------- Model ----------------------
from torchvision.models import EfficientNet_B0_Weights  # 상단 import 근처에 추가

def create_model(model_type, device):
    model_type = (model_type or "").lower()

    if model_type in ('efficientnet_b0', 'efficientnetb0', 'efnb0', 'b0'):
        # ImageNet 사전학습 가중치
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, 1)

    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 1)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model.to(device)

# ---------------------- Train / Validate ----------------------
def train_epoch(model, loader, criterion, optimizer, device):
    """
    반환: avg_loss, avg_acc(%)  ← TensorBoard 기록을 위해 정확도도 함께 반환
    """
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    with tqdm(loader, desc='Training', leave=False) as pbar:
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 누적
            running_loss += float(loss.item()) * labels.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)

            # 진행 상황 표시(배치 기준)
            batch_acc = (preds == labels.long()).float().mean().item() * 100.0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}%'})

    avg_loss = running_loss / max(1, total)
    avg_acc = 100.0 * correct / max(1, total)
    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device, return_probs=False):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        val_loss += float(loss.item()) * labels.size(0)

        probs = torch.sigmoid(logits)          # device tensor
        preds = (probs >= 0.5).long()          # device tensor

        total += labels.size(0)
        correct += (preds == labels.long()).sum().item()

        if return_probs:
            ps.append(probs.detach().cpu())
            ys.append(labels.detach().cpu())

    avg_loss = val_loss / max(1, total)
    acc = 100. * correct / max(1, total)

    if return_probs:
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        return avg_loss, acc, y_true, y_prob
    else:
        return avg_loss, acc

# ---------------------- Args ----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './train'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', './val'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', './test'))

    # 일반 학습 파라미터
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--model-type', type=str, default='efficientnet_b0')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_pos_weight', action='store_true')

    # TensorBoard 로그 디렉토리 (기본: SageMaker 출력 경로 하위)
    parser.add_argument(
        '--tb-logdir',
        type=str,
        default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', './output'), 'tensorboard')
    )
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)
    best_ckpt_path = os.path.join(args.model_dir, 'best_model.pt')

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_logdir)
    writer.add_text('hparams', str(vars(args)))

    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 데이터
    train_tf, eval_tf = get_transforms(args.image_size)
    train_ds = imagefolder_keep_classes(args.train, train_tf, keep=('0','1'))
    val_ds   = imagefolder_keep_classes(args.val,   eval_tf, keep=('0','1'))
    test_ds  = imagefolder_keep_classes(args.test,  eval_tf, keep=('0','1'))

    # pos_weight (옵션)
    pos_weight = None
    if args.use_pos_weight:
        ds = train_ds.dataset
        idxs = train_ds.indices if isinstance(train_ds, Subset) else range(len(ds))
        targets = np.array([ds.targets[i] for i in idxs])
        one_label = ds.class_to_idx.get('1', None)
        if one_label is None:
            raise RuntimeError("ImageFolder class '1'을 찾을 수 없습니다.")
        y = (targets == one_label).astype(int)
        p = y.mean()
        if p == 0.0:
            print("[WARN] 학습 세트에 양성이 없습니다. pos_weight 미적용.")
        else:
            pos_weight = torch.tensor([(1 - p) / p], dtype=torch.float32).to(device)
            print(f"[INFO] use_pos_weight -> pos_weight={pos_weight.item():.4f} (pos_rate={p:.4f})")

    num_workers = 4 if device.type == 'cuda' else 0
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    
    # 모델/손실/옵티마/스케줄러
    model = create_model(args.model_type, device)
    criterion = (nn.BCEWithLogitsLoss(pos_weight=pos_weight)
             if pos_weight is not None else nn.BCEWithLogitsLoss()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # 학습 (val AUROC 기준 early stopping & 체크포인트)
    best_auc = -1.0
    no_improve = 0
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"[Train] loss={tr_loss:.4f} acc={tr_acc:.2f}%")

            val_loss, val_acc, y_true, y_prob = validate(model, val_loader, criterion, device, return_probs=True)
            val_auc = roc_auc_score(y_true, y_prob)

            # 유댄 지수로 최적 threshold 계산
            best_thr = find_best_threshold(y_true, y_prob)

            # 0.5 기준 혼동행렬(참고)
            y_pred_05 = (y_prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05, labels=[0,1]).ravel()
            print(f"[Val] loss={val_loss:.4f} acc@0.5={val_acc:.2f}% auc={val_auc:.4f} thr*={best_thr:.3f} | CM@0.5 TN={tn} FP={fp} FN={fn} TP={tp}")

            # TensorBoard 기록
            step = epoch
            writer.add_scalar('Loss/train', tr_loss, step)
            writer.add_scalar('Loss/val',   val_loss, step)
            writer.add_scalar('Acc/train',  tr_acc, step)
            writer.add_scalar('Acc/val',    val_acc, step)
            writer.add_scalar('AUC/val',    val_auc, step)
            writer.add_scalar('LR',         optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('Threshold/youden', best_thr, step)
            writer.add_scalar('CM/TN', tn, step)
            writer.add_scalar('CM/FP', fp, step)
            writer.add_scalar('CM/FN', fn, step)
            writer.add_scalar('CM/TP', tp, step)

            # 스케줄러 스텝(기준: AUROC)
            scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

            # 체크포인트 & early stop
            improved = (val_auc > best_auc)
            if improved:
                best_auc = val_auc
                no_improve = 0
                state = {
                    'model': model.state_dict(),
                    'cfg': {
                        'image_size': args.image_size,
                        'best_threshold': float(best_thr),
                    },
                    'val_auc': float(val_auc),
                    'epoch': epoch
                }
                torch.save(state, best_ckpt_path)
                print(f"Saved best checkpoint (val_auc={val_auc:.4f}, thr*={best_thr:.3f}) -> {best_ckpt_path}")
            else:
                no_improve += 1
                print(f"no_improve={no_improve}/{args.patience} (best_auc={best_auc:.4f})")
                if no_improve >= args.patience:
                    print("Early stopping triggered.")
                    break

        # ------- 최종 테스트 -------
        print("\n[Load Best and Evaluate on TEST]")
        best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best['model'])

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x).squeeze(1)
                p = torch.sigmoid(logits)
                ps.append(p.detach().cpu())
                ys.append(y.detach().cpu())
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        test_auc = roc_auc_score(y_true, y_prob)
        print(f"[Test] AUROC={test_auc:.4f}")
        writer.add_scalar('AUC/test', test_auc, 0)

    finally:
        writer.close()
        print(f"TensorBoard logs saved to: {args.tb_logdir}")
        print(f"Model saved to: {args.model_dir}")

if __name__ == '__main__':
    main()