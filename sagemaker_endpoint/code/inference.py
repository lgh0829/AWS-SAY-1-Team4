# code/inference.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
import io

# --- 1. 모델 로딩 함수 ---
def model_fn(model_dir):
    """
    SageMaker가 모델을 로드할 때 호출하는 함수입니다.
    model_dir 경로에는 model.tar.gz 압축이 풀린 파일들이 위치합니다.
    """
    print(f"모델 로딩 시작: {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet 모델 구조 정의
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3) # 클래스 수에 맞게 수정
    
    # 저장된 가중치 로드
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device).eval()
    print("모델 로딩 완료")
    return model

# --- 2. 입력 데이터 처리 함수 ---
def input_fn(request_body, request_content_type):
    """
    클라이언트로부터 받은 요청을 모델이 이해할 수 있는 형태로 변환합니다.
    """
    print(f"입력 데이터 처리 (Content-Type: {request_content_type})")
    if request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        try:
            img = Image.open(io.BytesIO(request_body)).convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"이미지 디코딩 오류: {e}")
    else:
        raise ValueError(f"지원하지 않는 Content-Type: {request_content_type}")

# --- 3. 추론 함수 (핵심 로직) ---

def predict_fn(input_data, model):
    """
    실제 추론과 Vanilla Gradient 시각화를 수행합니다.
    """
    print("추론 및 XAI 시각화 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    original_image_np = np.array(input_data)

    input_tensor = transform(input_data).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()
    
    model.zero_grad()
    output[0, prediction].backward()
    gradients = input_tensor.grad.data.squeeze().cpu().numpy()
    
    gradients = np.mean(gradients, axis=0)
    heatmap = np.abs(gradients)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    H, W = original_image_np.shape[:2]
    hm_resized = cv2.resize((heatmap * 255).astype(np.uint8), (W, H))
    hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    
    # ✨ 1. 색상 순서를 RGB에서 BGR로 변환합니다.
    img_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
    
    # ✨ 2. 색상이 변환된 'img_bgr' 변수를 사용하여 이미지를 합성합니다.
    overlay_img = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

    return overlay_img, {'prediction': prediction, 'confidence': confidence}

# --- 4. 출력 데이터 처리 함수 ---
def output_fn(prediction_output, response_content_type):
    """
    predict_fn의 결과를 클라이언트에게 보낼 형태로 변환합니다.
    여기서는 시각화 이미지를 반환합니다.
    """
    print(f"출력 데이터 처리 (Accept: {response_content_type})")
    overlay_img, metadata = prediction_output
    
    # 이미지를 PNG 형식으로 인코딩
    is_success, buffer = cv2.imencode(".png", overlay_img)
    if not is_success:
        raise ValueError("XAI 이미지 인코딩 실패")

    # 메타데이터를 커스텀 헤더에 추가하여 전송
    headers = {'Content-Type': 'image/png', 'X-Custom-Metadata': json.dumps(metadata)}
    
    return buffer.tobytes(), headers