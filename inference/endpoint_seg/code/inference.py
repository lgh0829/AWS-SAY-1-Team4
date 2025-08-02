import torch
import numpy as np
import albumentations as A
import cv2
import os
from pathlib import Path
import boto3
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
        import base64
        data = json.loads(request_body)
        image_base64 = data["image"]
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        s3_bucket = data['s3_bucket']
        s3_prefix = data['s3_prefix']
        return image, s3_bucket, s3_prefix
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# 추론 실행 함수
def predict_fn(inputs, model):
    import uuid
    from PIL import Image

    image, s3_bucket, s3_prefix = inputs

    # 1. 입력 이미지 정규화
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    # 2. 채널 정리
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=2)

    # 3. 전처리 (resize → tensor 변환)
    augs = aug(image=image)
    image_aug = augs["image"].transpose((2, 0, 1))  # HWC → CHW
    image_aug = torch.tensor(image_aug).unsqueeze(0).float().to(device)

    # 4. 추론
    with torch.no_grad():
        outputs = model(image_aug)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    # 5. 마스크 이진화
    left_lung_mask = (probs[0] > 0.2).astype(np.uint8)
    right_lung_mask = (probs[1] > 0.2).astype(np.uint8)

    # 6. 저장용 준비
    uid = str(uuid.uuid4())
    mask_path = f"/tmp/{uid}_mask.npy"
    overlay_path = f"/tmp/{uid}_overlay.png"

    np.save(mask_path, np.stack([left_lung_mask, right_lung_mask], axis=0))  # (2, H, W)

    # 7. 오버레이 생성 (lung region만 보이도록 마스킹)
    mask_combined = np.clip(left_lung_mask + right_lung_mask, 0, 1)
    mask_resized = cv2.resize(mask_combined, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 원본 이미지를 흑색 배경 위에 마스크 영역만 살리는 방식
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    masked_image = cv2.bitwise_and(image_uint8, image_uint8, mask=mask_resized.astype(np.uint8))
    overlay = masked_image

    # mask_3ch = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2) * 127  # 0-255 범위로 조정
    # overlay = 0.6 * image + 0.4 * mask_3ch
    # overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)

    # 8. S3 업로드
    s3 = boto3.client("s3")
    s3_key_mask = f"{s3_prefix}/mask.npy"
    s3_key_overlay = f"{s3_prefix}/overlay.png"

    s3.upload_file(mask_path, s3_bucket, s3_key_mask)
    s3.upload_file(overlay_path, s3_bucket, s3_key_overlay)

    # 9. URI 반환
    return {
        "mask_s3_uri": f"s3://{s3_bucket}/{s3_key_mask}",
        "overlay_s3_uri": f"s3://{s3_bucket}/{s3_key_overlay}"
    }

# 출력 처리 함수
def output_fn(prediction, accept="application/json"):
    return json.dumps({
        "mask_s3_uri": prediction["mask_s3_uri"],
        "overlay_s3_uri": prediction["overlay_s3_uri"]
    }), accept