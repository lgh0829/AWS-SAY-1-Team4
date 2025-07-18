import os
from common.pneumo_utils.segmentation import LungSegmenter
from common.pneumo_utils.preprocessing import ImagePreprocessor
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchxrayvision as xrv

class ChestXrayPredictor:
    def __init__(self, model_path, device=None):
        # 디바이스 설정
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 전처리 도구 초기화
        self.preprocessor = ImagePreprocessor()
        self.segmenter = LungSegmenter(device=self.device)
        
        # 변환기 설정
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 클래스 이름
        self.class_names = ['normal', 'pneumonia', 'not_normal']
    
    def _load_model(self, model_path):
        """모델 로드 함수"""
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)  # 3-class classification
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
        
    
    def predict(self, image_path):
        """이미지 경로에서 예측 수행"""
        # 1. 이미지 전처리
        processed_img = self.preprocessor.process_image(image_path)
        
        # 2. 폐 영역 분할
        masked_img, _, _ = self.segmenter.segment_image(image_path)
        
        # 3. 변환 적용
        img_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        
        # 4. 예측 수행
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
        
        # 5. 결과 반환
        result = {
            'class_id': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': probs[predicted_class].item(),
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(self.class_names, probs)
            }
        }
        
        return result