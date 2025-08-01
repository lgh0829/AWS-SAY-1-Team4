import boto3
import json
import base64
import os

# -----------------------------------------------------
# 1. 설정
# -----------------------------------------------------

# (1) 엔드포인트 정보
endpoint_name = 'say1-4team-realtime-endpoint2025-07-31-07-56-05'
aws_region = 'ap-northeast-2' # 서울 리전

# (2) 이미지 경로
primary_image_path = 'test_image.jpg'  # 추론 및 그래디언트 계산에 사용할 이미지 (Lung Segmentation 이미지)
background_image_path = 'test_raw_image.jpg'  # 시각화 배경으로만 사용할 이미지 (원본 이미지)

# -----------------------------------------------------


# 2. 각 이미지를 읽고 base64로 인코딩
def image_to_base64(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded_str = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_str

try:
    primary_b64 = image_to_base64(primary_image_path)
    background_b64 = image_to_base64(background_image_path)
except FileNotFoundError as e:
    print(f"❌ 오류: 이미지 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
    exit()


# 3. SageMaker가 받을 JSON 페이로드 생성
payload = {
    "primary_image": primary_b64,
    "background_image": background_b64,
    "original_filename": os.path.basename(background_image_path) 
}
payload_json = json.dumps(payload)


# 4. 엔드포인트 호출
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=aws_region)
try:
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',  # ContentType을 json으로 변경
        Body=payload_json
    )


    # 5. 결과 파싱 및 출력
    result_str = response['Body'].read().decode('utf-8')
    result = json.loads(result_str)

    print("✅ 추론 성공!")
    print(json.dumps(result, indent=4, ensure_ascii=False))

except Exception as e:
    print(f"❌ 엔드포인트 호출 중 오류 발생: {e}")