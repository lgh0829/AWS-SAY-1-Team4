# invoke_endpoint_locally.py

import boto3
import json
import os

# --- ⚙️ 1. 테스트 설정 (이 부분을 본인 환경에 맞게 수정하세요) ---

# 로컬에 있는 테스트 이미지 파일의 경로
LOCAL_IMAGE_PATH = '87ddbd40-b218-4bfd-9a82-2ea252e08c1e.jpg'  # 👈 테스트할 이미지 파일명

# 호출할 SageMaker 엔드포인트의 이름
ENDPOINT_NAME = 'pre-4team-25-07-21-11-12-43' # 👈 본인이 배포한 엔드포인트 이름

# (선택 사항) 테스트 결과를 업로드할 S3 버킷 정보
# 이 스크립트는 로컬에만 저장하지만, 원할 경우 S3 업로드 로직을 추가할 수 있습니다.
# RESULT_S3_BUCKET = 'your-result-bucket' 

# -----------------------------------------------------------------

print(f"🚀 SageMa0ker 엔드포인트 로컬 호출 테스트를 시작합니다.")
print(f"   - 대상 엔드포인트: {ENDPOINT_NAME}")
print(f"   - 테스트 이미지: {LOCAL_IMAGE_PATH}")

try:
    # --- 2. 로컬 이미지 읽기 및 엔드포인트 호출 ---
    
    # boto3 클라이언트 생성
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # 이미지 파일을 바이너리(bytes) 모드로 읽기
    with open(LOCAL_IMAGE_PATH, 'rb') as f:
        image_data = f.read()
        
    # 파일 확장자에 따라 ContentType 결정
    content_type = 'image/jpeg' if LOCAL_IMAGE_PATH.lower().endswith(('.jpg', '.jpeg')) else 'image/png'

    print("\n1/3: SageMaker 엔드포인트를 호출합니다...")
    
    # SageMaker 엔드포인트 호출
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=content_type,
        Body=image_data,
        Accept='image/png'  # XAI 이미지를 PNG로 받음
    )
    print("✅ 엔드포인트 호출 성공")

    # --- 3. 결과 처리 및 로컬 저장 ---
    
    print("2/3: 응답 데이터를 처리합니다...")
    # 응답 본문에서 시각화 이미지(bytes)를 읽어오기
    xai_image_bytes = response['Body'].read()
    
    # 커스텀 헤더에서 메타데이터(예측 결과) 읽어오기
    metadata = json.loads(response['CustomAttributes'])
    print("✅ 응답 데이터 처리 성공")
    
    # 결과 파일명 설정
    result_filename = 'endpoint_test_result.png'
    
    print(f"3/3: 결과 이미지를 로컬에 저장합니다... (파일명: {result_filename})")
    # 시각화 이미지를 로컬 파일로 저장
    with open(result_filename, 'wb') as f:
        f.write(xai_image_bytes)

    # --- 4. 최종 결과 출력 ---
    
    print("\n🎉 테스트 성공!")
    print(f"결과가 '{result_filename}' 파일로 저장되었습니다.")
    print(f"모델 예측 결과: {json.dumps(metadata, indent=2)}")

except Exception as e:
    print(f"\n❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()