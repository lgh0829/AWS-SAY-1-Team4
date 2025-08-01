# code/inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
import io
from pathlib import Path
import sys
from model_factory import ModelFactory
import logging
from lung_seg.pretrained_models import create_model

# --- 1. 모델 로딩 함수 ---
def model_fn(model_dir):
    """
    SageMaker가 모델을 로드할 때 호출하는 함수입니다.
    model_dir 경로에는 model.tar.gz 압축이 풀린 파일들이 위치합니다.
    """
    print(f"모델 로딩 시작: {model_dir}")
    
    # 로그 디렉토리 생성
    log_dir = "/opt/ml/tmp/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("inference")
    logger.info("모델 로딩 시작")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 장치: {device}")
    
    # 모델 파일 경로
    model_path = os.path.join(model_dir, 'model.pth')
    
    # 모델 메타데이터 로드 (존재하는 경우)
    model_info_path = os.path.join(model_dir, 'model_info.json')
    model_type = 'resnet50'  # 기본값
    num_classes = 3  # 기본값
    pretrained_weight = None  # 기본값
    
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                model_type = model_info.get('model_type', model_type)
                num_classes = model_info.get('num_classes', num_classes)
                pretrained_weight = model_info.get('pretrained_weight', pretrained_weight)
            print(f"모델 정보 로드: {model_type}, 클래스 수: {num_classes}, 사전학습: {pretrained_weight}")
        except Exception as e:
            print(f"모델 정보 로드 실패: {e}")
    
    # 모델 설정 구성
    config = {
        'model_type': model_type,
        'num_classes': num_classes,
        'pretrained_weight': pretrained_weight
    }
    
    # 모델 생성
    try:
        model = ModelFactory.create(config, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("모델 로딩 완료")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        # 폴백 - 기본 ResNet50 모델
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        print("폴백 모델 로딩 완료")
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

# --- 4. 추론 함수 (핵심 로직) ---
def predict_fn(input_data, model):
    """
    실제 추론과 Vanilla Gradient 시각화를 수행합니다.
    """
    print("추론 및 XAI 시각화 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 전처리 적용
    processed_image = preprocess_image(input_data)
    
    # 모델 입력용 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 원본 이미지 저장 (시각화용)
    original_image_np = np.array(processed_image)

    # 텐서 변환 및 그래디언트 계산 설정
    input_tensor = transform(processed_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 모델 추론
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()
    
    # 클래스별 확률
    class_probs = {i: probs[0, i].item() for i in range(probs.shape[1])}
    
    # 클래스 이름 매핑
    class_names = {0: 'normal', 1: 'pneumonia', 2: 'not_normal'}
    prediction_label = class_names.get(prediction, f'class_{prediction}')
    
    # # Vanilla Gradient 시각화
    # try:
    #     model.zero_grad()
    #     output[0, prediction].backward()
    #     gradients = input_tensor.grad.data.squeeze().cpu().numpy()
        
    #     gradients = np.mean(gradients, axis=0)
    #     heatmap = np.abs(gradients)
    #     if heatmap.max() > heatmap.min():
    #         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    #     H, W = original_image_np.shape[:2]
    #     hm_resized = cv2.resize((heatmap * 255).astype(np.uint8), (W, H))
    #     hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        
    #     # 색상 순서 변환 - 오류 수정
    #     # 이미지가 이미 RGB인지 확인 후 변환
    #     if len(original_image_np.shape) == 3 and original_image_np.shape[2] == 3:
    #         # RGB에서 BGR로 변환 필요
    #         img_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
    #     else:
    #         # 이미 BGR이거나 그레이스케일인 경우
    #         img_bgr = original_image_np
    #         if len(img_bgr.shape) == 2:  # 그레이스케일인 경우
    #             img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
    #     # 이미지 합성
    #     overlay_img = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
    # except Exception as e:
    #     print(f"히트맵 생성 오류: {e}")
    #     # 오류 시 원본 이미지 반환
    #     if len(original_image_np.shape) == 3 and original_image_np.shape[2] == 3:
    #         overlay_img = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
    #     else:
    #         overlay_img = original_image_np
    #         if len(overlay_img.shape) == 2:
    #             overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
    
    # 메타데이터 구성
    metadata = {
        'prediction': prediction,
        'prediction_label': prediction_label,
        'confidence': confidence,
        'class_probabilities': class_probs
    }

    return metadata#, overlay_img

# --- 5. 출력 데이터 처리 함수 ---
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
    headers = {
        'Content-Type': 'image/png',
        'X-Pneumonia-Prediction': metadata['prediction_label'],
        'X-Pneumonia-Confidence': str(metadata['confidence']),
        'X-Pneumonia-Metadata': json.dumps(metadata)
    }
    
    return buffer.tobytes(), headers