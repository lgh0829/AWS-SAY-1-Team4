import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import mlflow
import random
from PIL import Image

class Cutout(object):
    def __init__(self, size=32):
        self.size = size

    def __call__(self, img):
        # 이미지가 PIL 이미지인 경우 텐서로 변환
        if not torch.is_tensor(img):
            img = F.to_tensor(img)
            tensor_converted = True
        else:
            tensor_converted = False
            
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        img[:, y:y+self.size, x:x+self.size] = 0
        
        # 원래 PIL 이미지였다면 다시 PIL로 변환
        if tensor_converted:
            img = F.to_pil_image(img)
        return img

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
    
    return parser.parse_args()

def get_transforms():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 향상된 데이터 증강 기법 적용
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([Cutout(size=32)], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, data_transforms

def create_model(model_type, num_classes, device):
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

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

def create_balanced_sampler(dataset):
    """
    클래스 불균형을 처리하기 위한 WeightedRandomSampler 생성
    특히 클래스 1에 더 높은 가중치 부여
    """
    targets = [label for _, label in dataset.samples]
    
    # 클래스별 샘플 수 계산
    class_count = torch.bincount(torch.tensor(targets))
    print(f"클래스별 샘플 수: {class_count.tolist()}")
    
    # 클래스별 가중치 계산 (샘플 수가 적을수록 가중치 증가)
    class_weights = 1.0 / class_count
    
    # 클래스 1에 추가 가중치 부여 (더 자주 샘플링되도록)
    if len(class_weights) > 1:
        class_weights[1] *= 2.0  # 클래스 1의 가중치를 2배로 증가
    
    # 각 샘플의 가중치 설정
    sample_weights = [class_weights[t] for t in targets]
    weights = torch.DoubleTensor(sample_weights)
    
    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

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
    
    # 데이터 로더 설정
    train_transforms, data_transforms = get_transforms()
    
    # 데이터셋 생성
    train_dataset = datasets.ImageFolder(args.train, transform=train_transforms)
    val_dataset = datasets.ImageFolder(args.val, transform=data_transforms)
    test_dataset = datasets.ImageFolder(args.test, transform=data_transforms)
    
    # 클래스 불균형을 처리하기 위한 sampler 생성
    train_sampler = create_balanced_sampler(train_dataset)
    
    # 클래스 분포 출력
    print(f"클래스 이름: {train_dataset.classes}")
    print(f"클래스 인덱스 맵핑: {train_dataset.class_to_idx}")

    # 데이터 로더 설정 시 worker 수 최적화
    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            sampler=train_sampler,  # 커스텀 sampler 사용
                            num_workers=num_workers,
                            pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_dataset, 
                          batch_size=args.batch_size,
                          shuffle=False,  # 검증/테스트 시에는 셔플하지 않음
                          num_workers=num_workers,
                          pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_dataset, 
                           batch_size=args.batch_size,
                           shuffle=False,
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
        
        # 최종 테스트
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        mlflow.log_metric('test_accuracy', test_acc)

if __name__ == '__main__':
    main()