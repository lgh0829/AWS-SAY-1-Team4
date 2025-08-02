import torch
import numpy as np
import albumentations as A
import cv2
import os
from pathlib import Path
# For local testing and SageMaker compatibility
try:
    from .models.unet import Resnet  # When imported as a package
except ImportError:
    from models.unet import Resnet   # When running directly
import json

# 이미지 전처리 정의
IMG_SIZE = 512
aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, p=1)], p=1)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로딩 함수
def model_fn(model_dir):
    # SageMaker가 모델을 저장하는 기본 경로 사용
    model_path = os.path.join(model_dir, "resnet34.pth")
    model = Resnet(seg_classes=2, backbone_arch="resnet34", model_path=model_path).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 입력 처리 함수
def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        image_bytes = bytes(data["image"])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# 추론 실행 함수
def predict_fn(input_data, model):
    # 1. 입력 이미지 정규화 및 타입 변환
    image = input_data
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0  # 0~1 스케일로 정규화

    # 2. 흑백 이미지일 경우 채널 확장
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image]*3, axis=2)

    # 3. 전처리 수행 (Resize, Tensor 변환)
    augs = aug(image=image)
    image_aug = augs["image"].transpose((2, 0, 1))  # HWC → CHW
    image_aug = torch.tensor(image_aug).unsqueeze(0).float().to(device)

    # 4. 모델 추론
    with torch.no_grad():
        outputs = model(image_aug)  # Shape: (1, 2, H, W)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()  # (2, H, W)

    # 5. 마스크 이진화
    left_lung_mask = (probs[0] > 0.2).astype(np.uint8)
    right_lung_mask = (probs[1] > 0.2).astype(np.uint8)

    # 6. 반환: 좌우 폐 마스크 개별 반환
    return {
        "left_lung": left_lung_mask.tolist(),
        "right_lung": right_lung_mask.tolist()
    }

# 출력 처리 함수
def output_fn(prediction, accept="application/json"):
    return json.dumps({"mask": [prediction["left_lung"], prediction["right_lung"]]}), accept