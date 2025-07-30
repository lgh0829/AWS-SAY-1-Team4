# code/model_factory.py
import torch
import torch.nn as nn
import torchvision.models as models

class ModelFactory:
    """모델 생성을 담당하는 팩토리 클래스"""
    
    @staticmethod
    def create(config, device):
        """
        설정에 따라 모델 생성
        
        Args:
            config (dict): 모델 설정 정보
            device (torch.device): 사용할 장치
            
        Returns:
            torch.nn.Module: 생성된 모델
        """
        model_type = config.get('model_type', 'resnet50').lower()
        num_classes = config.get('num_classes', 3)
        pretrained_weight = config.get('pretrained_weight')
        
        print(f"ModelFactory: 모델 생성 - {model_type}, 클래스 수: {num_classes}")
        
        # 사전학습 가중치 처리
        weights = None
        if pretrained_weight:
            if pretrained_weight.upper() in ('IMAGENET1K_V1', 'IMAGENET1K_V2', 'DEFAULT'):
                weights = pretrained_weight.upper()
        
        # 모델 유형별 생성
        try:
            if model_type == 'resnet18':
                model = models.resnet18(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_type == 'resnet34':
                model = models.resnet34(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_type == 'resnet50':
                model = models.resnet50(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_type == 'densenet121':
                model = models.densenet121(weights=weights)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif model_type == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_type == 'mobilenet_v2':
                model = models.mobilenet_v2(weights=weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            else:
                print(f"지원되지 않는 모델 타입: {model_type}, ResNet50으로 대체")
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            return model.to(device)
        except Exception as e:
            print(f"모델 생성 중 오류: {e}")
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model.to(device)
