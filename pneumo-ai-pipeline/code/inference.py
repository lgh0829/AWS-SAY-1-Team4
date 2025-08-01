import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
import io
import yaml
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_factory import ModelFactory
from lung_seg.pretrained_models import create_model
from lung_seg.inference import inference as lung_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 모델 로딩 ---
def model_fn(model_dir):
    print(f"모델 로딩 시작: {model_dir}")
    print(f"사용 장치: {device}")
    model_path = os.path.join(model_dir, 'model.pth')
    config_path = os.path.join(model_dir, 'model_config.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"모델 설정 로드: {config}")
    except FileNotFoundError:
        print("설정 파일 없음: 기본 설정 사용")
        config = {
            'model': {
                'type': 'resnet50',
                'num_classes': 3,
                'pretrained_weight': 'IMAGENET1K_V1'
            }
        }

    try:
        model = ModelFactory.create(config, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("모델 로딩 완료")
        return model
    except Exception as e:
        raise RuntimeError(f"모델 로드 오류: {e}")

# --- 2. 입력 처리 ---
def input_fn(request_body, request_content_type):
    print(f"입력 처리 시작: {request_content_type}")
    if request_content_type in ['application/x-image', 'image/jpeg', 'image/png']:
        try:
            image = Image.open(io.BytesIO(request_body)).convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"이미지 디코딩 오류: {e}")
    else:
        raise ValueError(f"지원하지 않는 Content-Type: {request_content_type}")

# --- 3. 폐 분할 ---
def segment_lungs(image):
    print("폐 분할 시작")
    model_type = 'resnet34'
    try:
        lung_model = create_model(model_type).to(device)
        lung_model.eval()
    except Exception as e:
        raise ValueError(f"폐 분할 모델 로드 오류: {e}")

    if isinstance(image, Image.Image):
        image = np.array(image)
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("segment_lungs(): 입력 이미지가 None이거나 잘못된 형식입니다")

    try:
        _, mask = lung_inference(lung_model, image, thresh=0.2)
    except Exception as e:
        raise ValueError(f"폐 분할 오류: {e}")

    combined_mask = (mask[0] + mask[1]) > 0
    combined_mask = (combined_mask * 255).astype(np.uint8)

    original_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original_img = cv2.resize(original_img, (combined_mask.shape[1], combined_mask.shape[0]))
    binary_mask_3ch = np.stack([combined_mask]*3, axis=-1) // 255
    masked_img = original_img * binary_mask_3ch
    return masked_img

# --- 4. 추론 ---
def predict_fn(input_data, model):
    print("추론 시작")
    if isinstance(input_data, Image.Image):
        input_data = np.array(input_data)
    if input_data is None:
        raise ValueError("predict_fn(): 입력 데이터가 None입니다")

    original_image_np = input_data
    segmented = segment_lungs(input_data)
    transformed_image = Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

    input_tensor = transforms.ToTensor()(transformed_image).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except Exception as e:
        raise RuntimeError(f"모델 추론 오류: {e}")

    probs = F.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()
    class_probs = {i: probs[0, i].item() for i in range(probs.shape[1])}
    class_names = {0: 'normal', 1: 'pneumonia', 2: 'not_normal'}
    prediction_label = class_names.get(prediction, f'class_{prediction}')

    try:
        input_tensor.requires_grad_()
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
        if len(original_image_np.shape) == 3 and original_image_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_GRAY2BGR)
        overlay_img = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
    except Exception as e:
        print(f"히트맵 생성 오류: {e}")
        overlay_img = original_image_np
        if len(overlay_img.shape) == 2:
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)

    metadata = {
        'prediction': prediction,
        'prediction_label': prediction_label,
        'confidence': confidence,
        'class_probabilities': class_probs
    }

    return overlay_img, metadata

# --- 5. 출력 ---
def output_fn(prediction_output, response_content_type):
    print(f"출력 포맷 처리: {response_content_type}")
    overlay_img, metadata = prediction_output
    is_success, buffer = cv2.imencode(".png", overlay_img)
    if not is_success:
        raise ValueError("XAI 이미지 인코딩 실패")

    headers = {
        'Content-Type': 'image/png',
        'X-Pneumonia-Prediction': metadata['prediction_label'],
        'X-Pneumonia-Confidence': str(metadata['confidence']),
        'X-Pneumonia-Metadata': json.dumps(metadata)
    }
    return buffer.tobytes(), headers