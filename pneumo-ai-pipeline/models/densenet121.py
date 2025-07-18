import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import torchxrayvision as xrv

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
        transforms.Grayscale(num_output_channels=1),  # ✅ 핵심
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])

    
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ✅ 핵심
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # ✅ 1채널 정규화
])

    
    return train_transforms, data_transforms

def create_model(model_type, num_classes, device):
    if model_type == 'densenet121':
        import torchxrayvision as xrv
        import os
        import torch.nn as nn
        import torch

        # ✅ 사전학습 weight 수동 다운로드
        model_dir = "/root/.torchxrayvision/models_data"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(
            model_dir,
            "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
        )

        if not os.path.exists(model_path):
            os.system(f"wget https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt -O {model_path}")


        model = torch.load(model_path, map_location=device)  # ✅ 그냥 모델 자체를 불러와!

        # ✅ 3. classifier 수정
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        model.n_outputs = num_classes
        model.pathologies = [f"class_{i}" for i in range(num_classes)]

        return model.to(device)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")

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
    
    train_dataset = datasets.ImageFolder(args.train, transform=train_transforms)
    val_dataset = datasets.ImageFolder(args.val, transform=data_transforms)
    test_dataset = datasets.ImageFolder(args.test, transform=data_transforms)
    
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
    
    # MLflow 사용 여부 확인
    use_mlflow = bool(os.environ.get('MLFLOW_TRACKING_URI')) and bool(os.environ.get('MLFLOW_EXPERIMENT_NAME'))
    
    if use_mlflow:
        mlflow.start_run()
        mlflow.log_params(vars(args))
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 훈련
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # MLflow 로그 기록
        if use_mlflow:
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
            if use_mlflow:
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
    
    if use_mlflow:
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.end_run()

if __name__ == '__main__':
    main()