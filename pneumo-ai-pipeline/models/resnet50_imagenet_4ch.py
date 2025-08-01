import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import mlflow
from custom_dataset import FourChannelImageFolder  


def modify_resnet50_input_channels(model, in_channels):
    original_conv1 = model.conv1
    model.conv1 = torch.nn.Conv2d(in_channels, original_conv1.out_channels,
                                  kernel_size=original_conv1.kernel_size,
                                  stride=original_conv1.stride,
                                  padding=original_conv1.padding,
                                  bias=original_conv1.bias is not None)
    with torch.no_grad():
        model.conv1.weight[:, :3] = original_conv1.weight  # 기존 3채널 복사
        model.conv1.weight[:, 3:] = original_conv1.weight[:, :1]  # 4번째 채널 초기화
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--mask-dir', type=str, default=os.environ.get('SM_CHANNEL_MASK'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--model-type', type=str, default='resnet50')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()




def create_model(model_type, num_classes, device):
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model = modify_resnet50_input_channels(model, in_channels=4)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
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
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
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
                    continue
                total += 1
                correct += int(preds[i] == labels[i])
    acc = 100. * correct / total if total > 0 else 0.0
    return running_loss / len(test_loader), acc


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def main():
    args = parse_args()
    device = get_device()
    
    train_dataset = FourChannelImageFolder(args.train, args.mask_dir, train=True)
    val_dataset = FourChannelImageFolder(args.val, args.mask_dir, train=False)
    test_dataset = FourChannelImageFolder(args.test, args.mask_dir, train=False)

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=device.type=='cuda')

    model = create_model(args.model_type, args.num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        best_val_acc, patience_counter = 0.0, 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, step=epoch)
            print(f"Epoch {epoch+1}/{args.epochs}\nTrain Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%\nVal Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(args.model_dir, 'model.pth')
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            scheduler.step(val_acc)

        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy (all samples): {test_acc:.2f}%")
        mlflow.log_metric('test_accuracy_all', test_acc)
        test_loss_triage, test_acc_triage = evaluate_with_soft_triage(model, test_loader, device, threshold=0.65)
        print(f"Final Test Accuracy (excluding Uncertain): {test_acc_triage:.2f}%")
        mlflow.log_metric('test_accuracy_excluding_uncertain', test_acc_triage)


if __name__ == '__main__':
    main()
