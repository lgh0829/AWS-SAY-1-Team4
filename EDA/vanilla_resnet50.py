import argparse
import torch
import torchxrayvision as xrv
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision.models as models
import boto3
from time import gmtime, strftime
import csv
from datetime import datetime, timezone
import pytz
import io

# !!! 설정 !!!
# ---------------------------------
model_path = 'models/model.pth'  # 모델 가중치 파일 경로
#image_path = '87ddbd40-b218-4bfd-9a82-2ea252e08c1e.jpg'  # 테스트 이미지 로컬 경로
# ---------------------------------
bucket_name = 'say1-4team-bucket'  # S3 버킷 이름
s3_key = 'cxr-pneumonia-4/preprocessed-3/test/0/00000217_000.png' # S3에서 가져올 파일 경로
s3_raw_key = 'cxr-pneumonia-4/raw/test/0/00000217_000.png' # 원본 이미지 S3 경로
upload_key = f'cxr-pneumonia-4/upload-test/{s3_key}' # S3에 업로드할 경로
image_path = 'test_image.jpg'  # 로컬에 저장할 경로
raw_image_path = 'test_raw_image.jpg'  # 원본 이미지 로컬 저장 경로
# ---------------------------------




# --- 1) 모델 유틸 함수 ---
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(num_classes, device):
    # xrv.models.ResNet으로 CXR 사전학습된 가중치 사용
    model = models.resnet50(weights='IMAGENET1K_V1')
    # 프로젝트 클래스 수에 맞게 마지막 FC 레이어 교체
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.op_threshs = None
    # 학습된 가중치 로드
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)  # strict=False로 변경하여 일부 키가 일치하지 않아도 로드 가능
    model = model.to(device).eval()
    return model

# --- 2) VanillaGradientExplainer 클래스 ---
class VanillaGradientExplainer:
    def __init__(self, model, device):
        self.model = model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 입력 크기
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def compute_vanilla_gradients(self, image, target_class=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # 예측 결과 계산
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
        
        # 목표 클래스 설정
        if target_class is None:
            target_class = prediction
        
        # Backward pass - 목표 클래스에 대한 gradient 계산
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Gradient 추출 및 차원 처리
        gradients = input_tensor.grad.data.squeeze().cpu().numpy()
        
        # 다채널인 경우 (RGB 등) 평균 또는 첫 번째 채널만 사용
        if gradients.ndim == 3:
            gradients = np.mean(gradients, axis=0)  # 채널별 평균
        elif gradients.ndim == 1:
            # 1D인 경우 224x224로 reshape
            gradients = gradients.reshape(224, 224)
        
        return gradients, prediction, confidence

    def visualize(self, raw_image, heatmap, alpha=0.4, save_path=None, save_to_s3=False):
        # 원본 이미지를 numpy array로 변환
        raw_img_np = np.array(raw_image.convert('RGB'))
        # heatmap을 이미지 크기에 맞춰 리사이즈
        H, W = raw_img_np.shape[:2]
        hm_uint8 = (heatmap * 255).astype(np.uint8)
        hm_resized = cv2.resize(hm_uint8, (W, H))
        cmap = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(raw_img_np, 1 - alpha, cmap, alpha, 0)

        # save or show
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        ax.imshow(overlay)

        if save_to_s3:
            # 메모리에서 S3로 업로드
            s3 = boto3.client('s3')
            try:
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
                s3.upload_fileobj(buffer, bucket_name, upload_key)
                plt.close(fig)
                print(f"✅ S3에 파일 업로드 성공: s3://{bucket_name}/{upload_key}")
            except Exception as e:
                print(f"❌ S3에 파일 업로드 실패: {e}")
                raise
        else:
            plt.show()
        
        if save_path:
            # 로컬에 저장
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"Saved gradient visualization to '{save_path}'")
            
        else:
            plt.show()


# --- 3) 메인 스크립트 ---
def main():
    p = argparse.ArgumentParser()
    # p.add_argument('--model-path', required=True)
    # p.add_argument('--image-path', required=True)
    p.add_argument('--num-classes', type=int, default=3)
    p.add_argument('--output', type=str, default='grad.png')
    args = p.parse_args()

    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, image_path)
        print(f"✅ S3에서 segment 파일 다운로드 성공: {image_path}")
    except Exception as e:
        print(f"❌ S3에서 segment 파일 다운로드 실패: {e}")
        raise
    try:
        s3.download_file(bucket_name, s3_raw_key, raw_image_path)
        print(f"✅ S3에서 raw 파일 다운로드 성공: {raw_image_path}\n")
    except Exception as e:
        print(f"❌ S3에서 raw 파일 다운로드 실패: {e}")
        raise

    # 장치 설정 & 모델 로드
    device = get_device()
    model = create_model(args.num_classes, device)
    # model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    raw_image = Image.open(raw_image_path).convert('RGB')

    # Vanilla Gradient 계산 및 시각화
    explainer = VanillaGradientExplainer(model, device)
    gradients, prediction, confidence = explainer.compute_vanilla_gradients(image)
    # 절댓값 후 0-1 정규화
    gradients = np.abs(gradients)
    # 0-1 재정규화
    if gradients.max() > gradients.min():
        grad = (gradients - gradients.min()) / (gradients.max() - gradients.min())
    # 노이즈 제거를 위한 임계값 적용 (상위 5%만 유지)
    thresh = np.percentile(grad, 95)
    grad = np.where(grad >= thresh, grad, 0)
    # 부드러운 시각화를 위한 가우시안 블러 적용
    grad = cv2.GaussianBlur(grad, (5, 5), 1.0)

    # 시각화 저장
    explainer.visualize(raw_image, grad, save_path=None, save_to_s3=True)
    
    # 로컬 저장 확인 및 절대 경로 출력
    output_path = args.output
    if os.path.isfile(output_path):
        abs_path = os.path.abspath(output_path)
        print(f"✅ 로컬 파일 절대 경로 {abs_path}")
    else:
        print(f"❌ 로컬 파일 저장에 실패했습니다: {output_path}")
        
    # CSV 파일 작성
    csv_filename = 'prediction_results.csv'
    csv_s3_key = 'cxr-pneumonia-4/upload-test/prediction_results.csv'
    kst = pytz.timezone('Asia/Seoul')
    # 현재 시간 (UTC -> KST 변환)
    utc_now = datetime.now(timezone.utc)  # 권장되는 방식으로 UTC 시간 가져오기
    current_time = utc_now.astimezone(kst).strftime("%Y-%m-%d-%H-%M-%S")
    # current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    csv_data = [
        {
            'time_KST': current_time,
            'required_original': s3_raw_key,
            'required_segmented': s3_key,
            'predicted_class': prediction,
            'confidence': confidence,
            'uploaded_filename': upload_key
        }
    ]

    # 로컬에 CSV 파일 생성
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['time_KST', 'required_original', 'required_segmented', 'predicted_class', 'confidence', 'uploaded_filename'])
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"✅ CSV 파일 생성 완료: {csv_filename}")

    # S3에 CSV 파일 업로드
    try:
        with open(csv_filename, 'rb') as f:
            s3.upload_fileobj(f, bucket_name, csv_s3_key)
        print(f"✅ S3에 CSV 파일 업로드 성공: s3://{bucket_name}/{csv_s3_key}")
    except Exception as e:
        print(f"❌ S3에 CSV 파일 업로드 실패: {e}")
        raise
    
    # 예측 결과도 함께 출력
    if prediction == 0:
        print(f"\n>>> 예측 클래스: 정상, confidence: {confidence:.3f}")
    elif prediction == 1:
        print(f"\n>>> 예측 클래스: 폐렴, confidence: {confidence:.3f}")
    elif prediction == 2:
        print(f"\n>>> 예측 클래스: 기타, confidence: {confidence:.3f}")
    print(f"💾 시각화 결과 저장 파일명: {args.output}")
    print(f"📊 예측 결과 CSV 파일명: {csv_filename}\n")

if __name__ == "__main__":
    main()