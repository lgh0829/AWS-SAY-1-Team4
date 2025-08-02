import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import boto3
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import uuid
import logging
import sys
import base64
from time import gmtime, strftime

# ---------------------------------------
# 로깅 설정 (SageMaker CloudWatch로 로그 전송)
# ---------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------------------------------
# 헬스체크 핑 핸들러
# ---------------------------------------
def ping():
    logger.info("Ping 요청 수신")
    return {"status": "ok"}, 200

# ---------------------------------------
# 1) 모델 로드
# ---------------------------------------
def model_fn(model_dir):
    logger.info("--- model_fn: 모델 로딩 시작 ---")
    try:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
        logger.info(f"디바이스 설정 완료: {device}")

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        logger.info("ResNet50 모델 구조 정의 완료")

        checkpoint_path = os.path.join(model_dir, "model_last.pth")
        logger.info(f"모델 가중치 파일 경로: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            logger.error(f"모델 가중치 파일을 찾을 수 없음: {checkpoint_path}")
            # model_dir 내부의 모든 파일/폴더 목록을 로깅하여 디버깅 돕기
            logger.error(f"{model_dir} 내부 파일 목록: {os.listdir(model_dir)}")
            raise FileNotFoundError(f"Model file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("torch.load 로 가중치 파일 로드 성공")

        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info("model.load_state_dict 로 모델에 가중치 적용 완료")

        model.to(device).eval()
        logger.info("모델을 evaluation 모드로 설정 및 디바이스로 이동 완료")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("이미지 전처리(transform) 정의 완료")

        logger.info("--- model_fn: 모델 로딩 성공 ---")
        return {"model": model, "transform": transform, "device": device}

    except Exception as e:
        # 에러 발생 시, 전체 traceback을 로깅
        logger.error("model_fn 에서 치명적 에러 발생", exc_info=True)
        raise e

# ---------------------------------------
# 2) 입력 파싱
# ---------------------------------------
def input_fn(request_body, content_type="application/json"):
    logger.info(f"--- input_fn: 입력 처리 시작 (Content-Type: {content_type}) ---")
    try:
        if content_type == "application/json":
            body = json.loads(request_body)
            
            # 1. 추론용 이미지 (primary_image) 디코딩
            primary_image_b64 = body["primary_image"]
            primary_image_bytes = base64.b64decode(primary_image_b64)
            primary_image = Image.open(io.BytesIO(primary_image_bytes)).convert("RGB")
            
            # 2. 시각화용 이미지 (background_image) 디코딩
            background_image_b64 = body["background_image"]
            background_image_bytes = base64.b64decode(background_image_b64)
            background_image = Image.open(io.BytesIO(background_image_bytes)).convert("RGB")
            
            # 원본 파일 이름 가져오기
            original_filename = body.get("original_filename", "unknown_file.jpg")
            
            logger.info("JSON 페이로드에서 primary, background, filename 파싱 성공")
            
            # 파일 이름도 함께 딕셔너리로 반환
            return {
                "primary": primary_image, 
                "background": background_image,
                "filename": original_filename
            }

        else:
            # 이제 application/json만 지원
            raise ValueError(f"지원하지 않는 Content-Type: {content_type}. 'application/json'을 사용하세요.")
    except Exception as e:
        logger.error("input_fn 에서 에러 발생", exc_info=True)
        raise e

# ---------------------------------------
# 3) gradient 계산
# ---------------------------------------

def compute_vanilla_gradients(image, model, transform, device, target_class=None):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()

    if target_class is None:
        target_class = prediction

    # Backward pass
    model.zero_grad()
    target_score = output[0, target_class]
    target_score.backward()

    gradients = input_tensor.grad.data.squeeze().cpu().numpy()
    if gradients.ndim == 3:
        gradients = np.mean(gradients, axis=0)
    elif gradients.ndim == 1:
        gradients = gradients.reshape(224, 224)

    return gradients

# ---------------------------------------
# Helper: 히트맵 시각화 및 S3 업로드
# ---------------------------------------
def visualize_and_upload_heatmap(raw_image, heatmap, bucket, key, alpha=0.4):
    raw_np = np.array(raw_image.convert("RGB"))
    H, W = raw_np.shape[:2]
    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_resized = cv2.resize(hm_uint8, (W, H))
    cmap = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(raw_np, 1 - alpha, cmap, alpha, 0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(overlay)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close(fig)

    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket, key)


# ---------------------------------------
# 3) 예측 수행
# ---------------------------------------
def predict_fn(input_object, model_context):
    logger.info("--- predict_fn: 예측 시작 ---")
    try:
        # input_fn에서 받은 딕셔너리 풀기
        primary_image = input_object["primary"]      # 추론/계산용 이미지
        background_image = input_object["background"] # 시각화 배경용 이미지
        original_filename = input_object["filename"] # 파일 이름

        model = model_context["model"]
        transform = model_context["transform"]
        device = model_context["device"]
        logger.info("모델과 컨텍스트, 두 종류의 이미지 로드 완료")

        # 분류 예측 (primary_image 사용)
        img_tensor = transform(primary_image).unsqueeze(0).to(device)
        logger.info("Primary 이미지 텐서 변환 및 디바이스로 이동 완료")
        with torch.no_grad():
            outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        class_idx = torch.argmax(probs).item()
        logger.info(f"모델 예측 완료. 결과 class_id: {class_idx}")

        classes = ["normal", "pneumonia", "not_normal"]
        result = {
            'class_id': class_idx,
            'class_name': classes[class_idx],
            'confidence': probs[class_idx].item(),
            'probabilities': {cls: float(p) for cls, p in zip(classes, probs)}
        }
        logger.info(f"최종 예측 결과 JSON 생성: {result}")

        # Vanilla Gradient 히트맵 계산 (primary_image 사용)
        logger.info("Vanilla Gradient 계산 시작")
        gradients = compute_vanilla_gradients(primary_image, model, transform, device, target_class=class_idx)
        # 히트맵 후처리
        logger.info("히트맵 후처리 시작")
        gradients = np.abs(gradients)  # 1. 절댓값 적용
        # 2. 원본 그래디언트에서 임계값 계산 (상위 10%)
        thresh = np.percentile(gradients, 90)
        # 3. 임계값보다 낮은 값들은 0으로 처리
        grad = np.where(gradients >= thresh, gradients, 0)
        # 4. 걸러진 값들을 대상으로 0-1 정규화 (시각적 대비 극대화)
        if grad.max() > 0:
            grad = grad / grad.max()
        # 5. 가우시안 블러로 부드럽게 처리
        grad = cv2.GaussianBlur(grad, (5, 5), 1.0)
        logger.info("히트맵 계산 및 후처리 완료")

        # S3 버킷/키 설정
        heatmap_bucket = os.getenv('HEATMAP_S3_BUCKET')
        prefix = os.getenv('HEATMAP_S3_PREFIX', 'default-prefix')

        # 파일 이름에서 확장자 제거
        file_name_without_ext, _ = os.path.splitext(original_filename)
        timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

        # 새로운 규칙으로 파일 이름(key) 생성
        heatmap_key = f"{prefix.rstrip('/')}/{file_name_without_ext}-inferenced-{timestamp}.png"
        
        logger.info(f"히트맵 저장 경로 설정: s3://{heatmap_bucket}/{heatmap_key}")
        
        if heatmap_bucket:
            logger.info("히트맵 시각화 및 S3 업로드 시작")
            # 히트맵 시각화 (background_image 사용)
            visualize_and_upload_heatmap(background_image, grad, heatmap_bucket, heatmap_key)
            result['heatmap_s3_path'] = f"s3://{heatmap_bucket}/{heatmap_key}"
            logger.info("히트맵 S3 업로드 성공")
        else:
            logger.warning("HEATMAP_S3_BUCKET 환경변수가 설정되지 않아 히트맵을 업로드하지 않습니다.")

        logger.info("--- predict_fn: 예측 성공 ---")
        return result
    except Exception as e:
        logger.error("predict_fn 에서 치명적 에러 발생", exc_info=True)
        raise e

# ---------------------------------------
# 4) 출력 포맷팅
# ---------------------------------------
def output_fn(prediction, response_content_type):
    logger.info(f"--- output_fn: 출력 포맷팅 시작 (Content-Type: {response_content_type}) ---")
    try:
        if response_content_type == "application/json":
            return json.dumps(prediction), response_content_type
        else:
            raise ValueError(f"지원하지 않는 응답 Content-Type: {response_content_type}")
    except Exception as e:
        logger.error("output_fn 에서 에러 발생", exc_info=True)
        raise e