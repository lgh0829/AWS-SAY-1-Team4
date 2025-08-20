import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import models
from transformers import AutoModelForImageClassification
import numpy as np
import cv2
from collections import Counter
import kornia.augmentation as K

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

    # 입력 형태
    parser.add_argument('--input-size', type=int, default=224, help='H=W target size')
    parser.add_argument('--expect-channels', type=int, default=4,
                        help='이 스크립트는 4채널(RGB+mask)만 지원합니다.')
    parser.add_argument('--image-keys', type=str, default='image,img,x',
                        help='npz 내 이미지 후보 키(콤마구분, 우선순위 적용)')
    parser.add_argument('--mask-keys', type=str, default='mask,lung_mask,seg,msk',
                        help='npz 내 마스크 후보 키(콤마구분, 우선순위 적용)')

    # TensorBoard
    parser.add_argument('--tb-logdir', type=str,
                        default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR', './output'), 'tensorboard'))
    return parser.parse_args()


class NPZDataset(Dataset):
    """
    디렉토리 구조: root/{class_id or class_name}/*.npz
    npz 안에 다음 중 하나 이상의 키가 있어야 함:
      - 이미지: image | img | x (H,W) 또는 (H,W,3)
      - 마스크: mask | lung_mask | seg | msk (H,W) [선택]
    이 스크립트는 4채널(RGB+mask)만 지원합니다.
    """
    def __init__(self, root, expect_channels=4, image_keys=('image','img','x'),
                 mask_keys=('mask','lung_mask','seg','msk'), target_size=224, with_labels=True):
        if expect_channels != 4:
            raise NotImplementedError("This training script handles only 4-channel inputs. Set --expect-channels=4.")
        self.root = Path(root)
        self.files = sorted([str(p) for p in self.root.rglob('*.npz')])
        if len(self.files) == 0:
            raise RuntimeError(f'No NPZ files found under {root}')

        # 라벨: 상위 폴더명으로부터 추정 (ex: .../train/0/xxx.npz → 0)
        self.with_labels = with_labels
        if self.with_labels:
            self.labels = [self._label_from_path(fp) for fp in self.files]
        else:
            self.labels = [None] * len(self.files)
        # self.labels = [self._label_from_path(fp) for fp in self.files]
        self.expect_channels = expect_channels
        self.image_keys = image_keys
        self.mask_keys = mask_keys
        self.size = (target_size, target_size)

        # Augmentations
        # 기하 변환(이미지/마스크 동일 파라미터)
        self.geom_pair = K.AugmentationSequential(
            K.RandomRotation(15.0, p=0.5),
            K.RandomAffine(degrees=0.0, translate=(0.1, 0.1), p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["input", "mask"],   # 이미지/마스크 동기화
            same_on_batch=False,
        )
        # 포토메트릭(이미지만)
        self.photometric = K.ColorJitter(
            brightness=0.15,
            contrast=0.2,
            saturation=0.0,   # CXR이면 0 추천
            hue=0.0,
            p=0.5,
        )

        # 정규화 텐서
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _label_from_path(self, fp: str) -> int:
        p = Path(fp)
        rel = p.relative_to(self.root)
        cls_name = rel.parts[0]
        try:
            return int(cls_name)
        except ValueError:
            return abs(hash(cls_name)) % (10**6)

    def __len__(self):
        return len(self.files)

    def _ensure_hw3(self, arr):
        # arr: (H,W) or (H,W,C)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f'Unexpected image shape: {arr.shape}')
        return arr

    def _resize_hw(self, arr_hw_or_hwc):
        H, W = self.size
        if arr_hw_or_hwc.ndim == 2:  # (H,W)
            return cv2.resize(arr_hw_or_hwc, (W, H), interpolation=cv2.INTER_NEAREST)
        elif arr_hw_or_hwc.ndim == 3:  # (H,W,C)
            return cv2.resize(arr_hw_or_hwc, (W, H), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError(f'Unexpected array shape for resize: {arr_hw_or_hwc.shape}')

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as npz:
            # 이미지 키 선택
            img_key = None
            for k in self.image_keys:
                if k in npz:
                    img_key = k; break
            if img_key is None:
                raise KeyError(f"No image key found in {path}. Tried {self.image_keys}")

            img = npz[img_key]

            # 마스크 키 선택(선택적)
            msk_key = None
            for k in self.mask_keys:
                if k in npz:
                    msk_key = k; break
            mask = npz[msk_key] if msk_key else None

        # 이미지 전처리: (H,W) or (H,W,C) → (H,W,3)
        if img.ndim not in (2, 3):
            raise ValueError(f'Image must be 2D or 3D, got {img.shape} at {path}')
        img = img.astype(np.float32)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] != 3):
            img = self._ensure_hw3(img)

        # 리사이즈
        img = self._resize_hw(img)  # (H,W,3)

        # 마스크 처리(없으면 0으로 채움)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        else:
            mask = mask.astype(np.float32)
            if mask.ndim == 3:  # (H,W,1) → (H,W)
                mask = mask.squeeze(-1)
        mask = self._resize_hw(mask)  # (H,W)

        # 0~1 스케일로 변환 (Photometric은 정규화 전)
        img_01 = img / 255.0

        # (H,W,3)/(H,W) → (3,H,W)/(1,H,W)
        x_img = torch.from_numpy(img_01.transpose(2, 0, 1)).float()   # (3,H,W)
        x_msk = torch.from_numpy(np.expand_dims(mask, 0)).float()     # (1,H,W)

        # 기하 변환: 이미지/마스크 동일 파라미터
        x_img_b, x_msk_b = self.geom_pair(x_img.unsqueeze(0), x_msk.unsqueeze(0))
        x_img, x_msk = x_img_b.squeeze(0), x_msk_b.squeeze(0)         # (3,H,W), (1,H,W)

        # 포토메트릭: 이미지에만
        x_img = self.photometric(x_img.unsqueeze(0)).squeeze(0)       # (3,H,W)

        # 마스크 재이진화(기하변환 후 보간 영향 제거)
        x_msk = (x_msk > 0.5).float()

        # 정규화
        x_img = (x_img - self.mean) / self.std

        # 결합: (3+1,H,W) = (4,H,W)
        x = torch.cat([x_img, x_msk], dim=0)

        if self.with_labels:
            y = self.labels[idx]
            return x, y
        else:
            return x


def create_model(model_type, num_classes, device, in_channels=4):
    if model_type == "resnet50/microsoft":
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        return model.to(device)

    elif model_type == "resnet50/IMAGENET1K_V1":
        model = models.resnet50(weights='IMAGENET1K_V1')
        if in_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                    kernel_size=old_conv.kernel_size,
                                    stride=old_conv.stride,
                                    padding=old_conv.padding,
                                    bias=False)
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
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


def build_loaders(args, device):
    # 데이터셋 구성 (4채널 고정)
    common_kwargs = dict(
        expect_channels=args.expect_channels,
        image_keys=tuple([k.strip() for k in args.image_keys.split(',') if k.strip()]),
        mask_keys=tuple([k.strip() for k in args.mask_keys.split(',') if k.strip()]),
        target_size=args.input_size,
    )
    train_ds = NPZDataset(args.train, with_labels=True,  **common_kwargs)
    val_ds   = NPZDataset(args.val, with_labels=True,  **common_kwargs)
    test_ds = NPZDataset(args.test, with_labels=False, **common_kwargs)

    # 클래스 불균형 샘플러
    class_count = Counter(train_ds.labels)
    idx_to_label = np.array(train_ds.labels)
    weights_by_class = {c: 1.0 / cnt for c, cnt in class_count.items()}
    sample_weights = np.array([weights_by_class[y] for y in idx_to_label], dtype=np.float32)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"Class distribution: {class_count}")

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))
    return train_loader, val_loader, test_loader


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

    # 로더
    train_loader, val_loader, test_loader = build_loaders(args, device)

    # 모델 (입력 채널=4)
    in_ch = args.expect_channels
    model = create_model(args.model_type, args.num_classes, device, in_channels=in_ch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tb_logdir)
    writer.add_text('hparams', str(vars(args)))

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

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

    print(f"TensorBoard logs saved to: {args.tb_logdir}")
    print(f"Model saved to: {args.model_dir}")

if __name__ == '__main__':
    main()