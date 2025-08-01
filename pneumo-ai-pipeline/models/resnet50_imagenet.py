import os
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import kornia.augmentation as K
from pathlib import Path
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker 기본 인자
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--model-type', type=str, default='resnet50')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--augment',       type=lambda x: x.lower()=='true',
                        default=False,
                        help="pt 데이터셋에 증강을 적용할지 여부")
    return parser.parse_args()



def create_model(model_type, num_classes, device):
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        # 3) 가중치 초기화: 기존 RGB 복사 + mask 채널 He 초기화
        with torch.no_grad():
            w_old = old_conv.weight              # [64,3,7,7]
            w_new = torch.zeros_like(model.conv1.weight)  # [64,4,7,7]
            w_new[:, :3, :, :] = w_old            # RGB 복사
            init.kaiming_normal_(                # 마스크 채널은 He 초기화
                w_new[:, 3:4, :, :],
                mode='fan_out',
                nonlinearity='relu'
            )
            model.conv1.weight.copy_(w_new)

        # 4) FC 레이어 재설정
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model.to(device)


class PTTensorDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        # .pt 파일 리스트 (서브폴더명이 레이블)
        self.files   = list(Path(root_dir).rglob('*.pt'))
        self.augment = augment
        if augment:
            # 배치 단위로 처리하는 Kornia 증강기 선언
            self.augs = nn.Sequential(
                K.Resize((224,224)),
                K.RandomRotation(15.0, p=0.5),
                K.RandomAffine(degrees=0.0, translate=(0.1,0.1), p=0.5),
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) .pt에서 [4, H, W] 텐서와 레이블 로드
        tensor = torch.load(self.files[idx])           # float Tensor
        label  = int(self.files[idx].parent.name)      # 폴더명이 클래스

        # 2) 증강 (학습 시만)
        if self.augment:
            x = tensor.unsqueeze(0)    # [1,4,H,W]
            x = self.augs(x)           # [1,4,224,224]
            tensor = x.squeeze(0)      # [4,224,224]

        # 3) 채널별 정규화
        #   - RGB 채널: ImageNet mean/std
        #   - mask 채널: mean=0, std=1 (변경 없이)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=tensor.device)
        std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=tensor.device)
        tensor = (tensor - mean[:,None,None]) / std[:,None,None]

        return tensor, label



def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
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
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total


def evaluate_with_soft_triage(model, test_loader, device, threshold=0.7):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            max_conf, preds = probs.max(dim=1)

            for i in range(len(labels)):
                if max_conf[i] < threshold:
                    continue  # Uncertain: 정확도 계산에서 제외
                total += 1
                correct += int(preds[i] == labels[i])

    acc = 100. * correct / total if total > 0 else 0.0
    return running_loss / len(test_loader), acc

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

def main():
    args = parse_args()
    
    # 디바이스 설정
    device = get_device()
    
    # CUDA 사용 시 추가 설정
    if device.type == 'cuda':
        # 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # GPU 메모리 상태 출력
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # 데이터셋 로드
    train_dataset = PTTensorDataset(args.train, augment=args.augment)
    val_dataset   = PTTensorDataset(args.val,   augment=False)
    test_dataset  = PTTensorDataset(args.test,  augment=False)
    
    # 데이터 로더 설정 시 worker 수 최적화
    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_dataset, 
                          batch_size=args.batch_size,
                          num_workers=num_workers,
                          pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_dataset, 
                           batch_size=args.batch_size,
                           num_workers=num_workers,
                           pin_memory=device.type=='cuda')
    
    # 모델 설정
    model = create_model(args.model_type, args.num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.1, patience=3)
    
    # MLflow 실험 시작
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # 훈련
            train_loss, train_acc = train_epoch(model, train_loader, 
                                              criterion, optimizer, device)
            
            # 검증
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 로그 기록
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Early Stopping 체크
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 모델 저장
                model_path = os.path.join(args.model_dir, 'model.pth')
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step(val_acc)
            
        # 최종 모델 저장
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))

        # 최종 테스트
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy (all samples): {test_acc:.2f}%")
        mlflow.log_metric('test_accuracy_all', test_acc)
        test_loss_triage, test_acc_triage = evaluate_with_soft_triage(model, test_loader, device, threshold=0.65)
        print(f"Final Test Accuracy (excluding Uncertain): {test_acc_triage:.2f}%")
        mlflow.log_metric('test_accuracy_excluding_uncertain', test_acc_triage)

if __name__ == '__main__':
    main()