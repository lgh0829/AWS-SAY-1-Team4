import os
import argparse
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import yaml
import os
import re
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path
import dotenv

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

def replace_env_vars(value):
    """환경 변수 치환 함수"""
    if isinstance(value, str):
        # ${VAR_NAME} 패턴 찾기
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, value)
        
        # 각 환경 변수 치환
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value:
                value = value.replace(f"${{{var_name}}}", env_value)
        return value
    return value

def process_yaml_dict(yaml_dict):
    """중첩된 딕셔너리에서 환경 변수 치환"""
    result = {}
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            result[key] = process_yaml_dict(value)
        elif isinstance(value, list):
            result[key] = [process_yaml_dict(item) if isinstance(item, dict) else replace_env_vars(item) for item in value]
        else:
            result[key] = replace_env_vars(value)
    return result

def load_config(config_path):
    """설정 파일 로드 및 환경 변수 치환"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 치환 처리
    config = process_yaml_dict(config)
    return config

# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),  # ResNet 입력 크기
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# train_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# train_dataset = datasets.ImageFolder('/Users/skku_aws19/Desktop/aws_project/pre-project/data/train_preprocessed', transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataset = datasets.ImageFolder('/Users/skku_aws19/Desktop/aws_project/pre-project/data/val_preprocessed', transform=data_transforms)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_dataset = datasets.ImageFolder('/Users/skku_aws19/Desktop/aws_project/pre-project/data/test_preprocessed', transform=data_transforms)
# test_loader = DataLoader(test_dataset,batch_size=32, shuffle=False)

# model = models.resnet50(weights='IMAGENET1K_V1')
# model.fc = nn.Linear(model.fc.in_features, 3)  # 3-class classification
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)


# # TensorBoard 설정
# log_dir = './runs/pneumonia_' + datetime.now().strftime('%Y%m%d-%H%M%S')
# writer = SummaryWriter(log_dir=log_dir)

# # 모델 구조 시각화 (1회만)
# example_input = torch.randn(1, 3, 224, 224).to(device)
# writer.add_graph(model, example_input)

# # 옵티마이저 정보 기록
# writer.add_text("Model", model._get_name())
# writer.add_text("Loss Function", criterion._get_name())
# writer.add_text("Device", str(device))
# opt_params = optimizer.__getstate__()['defaults']
# opt_text = '\n'.join([f"{k}: {v}" for k, v in opt_params.items()])
# writer.add_text("Optimizer Parameters", opt_text)

# # 하이퍼파라미터
# num_epochs = 20
# patience = 5
# best_val_acc = 0.0
# patience_counter = 0

# # 학습 루프
# for epoch in range(num_epochs):
#     print(f"\n📘 Epoch [{epoch+1}/{num_epochs}]")
    
#     model.train()
#     running_loss, running_correct, total = 0.0, 0, 0

#     train_loop = tqdm(train_loader, desc="Training", leave=False)
#     for images, labels in train_loop:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         running_correct += predicted.eq(labels).sum().item()

#         train_loop.set_postfix({
#             "Loss": f"{loss.item():.4f}",
#             "Acc": f"{100. * running_correct / total:.2f}%"
#         })

#     train_acc = 100. * running_correct / total
#     train_loss = running_loss / len(train_loader)

#     # ✅ Validation
#     model.eval()
#     val_loss, val_correct, val_total = 0.0, 0, 0

#     val_loop = tqdm(val_loader, desc="Validating", leave=False)
#     with torch.no_grad():
#         for images, labels in val_loop:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = outputs.max(1)
#             val_total += labels.size(0)
#             val_correct += predicted.eq(labels).sum().item()

#             val_loop.set_postfix({
#                 "Loss": f"{loss.item():.4f}",
#                 "Acc": f"{100. * val_correct / val_total:.2f}%"
#             })

#     val_acc = 100. * val_correct / val_total
#     val_loss /= len(val_loader)

#     print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

#     # ✅ TensorBoard에 기록
#     writer.add_scalar("Loss/train", train_loss, epoch)
#     writer.add_scalar("Loss/val", val_loss, epoch)
#     writer.add_scalar("Accuracy/train", train_acc, epoch)
#     writer.add_scalar("Accuracy/val", val_acc, epoch)

#     # ✅ Early Stopping & Save Best
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         patience_counter = 0
#         torch.save(model.state_dict(), "best_model_augmented.pth")
#         print(f"💾 모델 저장됨 (val acc: {val_acc:.2f}%)")
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print(f"🛑 Early stopping at epoch {epoch+1}")
#             break

#     # ✅ Scheduler 업데이트
#     scheduler.step(val_acc)

# writer.close()
# print("🏁 학습 완료")

def run_training():
    # 설정 파일 로드
    config = load_config('configs/train_config.yaml')
    
    # SageMaker 세션 설정
    session = sagemaker.Session()
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    
    # 현재 시간으로 작업 이름 생성
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base_job_name = f"{config['base_job_name']}-{timestamp}"
    
    # 데이터 입력 설정
    if config['upload_data']:
        s3_config = config['s3']
        data_channels = {
            'train': f"s3://{s3_config['bucket_name']}/{s3_config['prefix']}/{s3_config['train_prefix']}",
            'val': f"s3://{s3_config['bucket_name']}/{s3_config['prefix']}/{s3_config['val_prefix']}",
            'test': f"s3://{s3_config['bucket_name']}/{s3_config['prefix']}/{s3_config['test_prefix']}"
        }
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pneumo-ai-pipeline 디렉토리
        data_channels = {
            'train': os.path.join(base_dir, config['local']['data_dir'], config['local']['train_dir']),
            'val': os.path.join(base_dir, config['local']['data_dir'], config['local']['val_dir']),
            'test': os.path.join(base_dir, config['local']['data_dir'], config['local']['test_dir'])
        }
    
    # MLflow 환경 변수 설정
    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow']['tracking_uri']
    os.environ['MLFLOW_EXPERIMENT_NAME'] = config['mlflow']['experiment_name']
    
    # SageMaker PyTorch Estimator 생성
    estimator = PyTorch(
        entry_point=config['entry_point'],
        source_dir=config['source_dir'],
        role=role,
        instance_count=config['instance_count'],
        instance_type=config['instance_type'],
        framework_version=config['framework_version'],
        py_version=config['py_version'],
        hyperparameters=config['hyperparameters'],
        base_job_name=base_job_name,
        sagemaker_session=session,
        output_path=f"s3://{s3_config['bucket_name']}/{s3_config['prefix']}/output",
        environment={
            'MLFLOW_TRACKING_URI': config['mlflow']['tracking_uri'],
            'MLFLOW_EXPERIMENT_NAME': config['mlflow']['experiment_name']
        }
    )
    
    # 훈련 실행
    print("Starting training job...")
    estimator.fit(data_channels, wait=True)
    print("Training completed!")
    
    # 모델 아티팩트 경로 출력
    print(f"Model artifacts stored at: {estimator.model_data}")

if __name__ == '__main__':
    run_training()
