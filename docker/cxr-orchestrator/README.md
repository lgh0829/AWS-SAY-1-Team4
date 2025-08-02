# 디렉토리 구조
```
cxr-orchestrator/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── api/                     # 라우터 정의
│   │   └── webhook.py           # /api/webhook/on-store
│   ├── service/                 # 각 기능 모듈화
│   │   ├── orthanc.py           # Orthanc WADO-RS 통신
│   │   ├── s3.py                # S3 업로드/다운로드
│   │   ├── sagemaker.py         # endpoint 호출
│   │   ├── bedrock.py           # Bedrock report 저장 API 호출
│   │   └── db.py                # MySQL 연결 및 쿼리
│   ├── model/
│   │   └── schema.py            # Pydantic 모델
│   ├── core/
│       └── config.py            # 환경 변수 및 설정 로딩
│
├── tests/                       # 테스트 코드
│   └── test_webhook.py
│
├── requirements.txt             # 의존성 정의
├── Dockerfile                   # 도커 배포 설정
├── .env                         # 환경변수 (.gitignore에 추가)
└── README.md                    # 프로젝트 설명
```

|경로|설명|
|--|--|
|main.py|FastAPI 앱 객체 및 라우터 등록|
|api/webhook.py|Orthanc Webhook 수신 처리 (/api/webhook/on-store)|
|service/orthanc.py|DICOM/JPEG 다운로드 (WADO-RS)|
|service/s3.py|JPEG, DICOM, Mask, Report 결과 S3 업로드|
|service/sagemaker.py|Segmentation / Classification SageMaker 호출|
|service/bedrock.py|Report 자동완성 or 저장용 Bedrock 엔드포인트 호출|
|service/db.py|환자 ID, 결과 경로 등 MySQL 기록|
|model/schema.py|StudyPayload, AnalysisResult 등의 구조 정의|
|core/config.py|환경변수 .env 로딩 및 설정 관리|

# Docker 빌드 및 실행

```bash
# 1. Docker 이미지 빌드
docker build -t cxr-orchestrator .

# 2. Docker 컨테이너 실행
docker run -d -p 8000:8000 --name cxr-api cxr-orchestrator
```

# S3 버킷 저장 구조

```
s3://say1-4team-bucket/prod/
└── studies/
    └── {StudyInstanceUID}/
        ├── input/
        │   ├── original/                        # JPEG 이미지
        │   │   └── {InstanceUID}.jpg
        │   └── dicom/                           # DICOM 원본
        │       └── {InstanceUID}.dcm
        ├── segmentation/
        │   └── masks/
        │       └── {InstanceUID}_mask.png
        ├── classification/
        │   └── result.json                      # 클래스/확률/heatmap 위치 등
        └── metadata/
            └── log_{timestamp}.json             # 파이프라인 수행 기록, 오류 등
```

# 로컬 환경 테스트

## FastAPI 서버 올리기

```bash
# 디렉토리 이동
cd cxr-orchestrator

# 가상환경 & 의존성 설치
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 실행
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# 재실행
.venv/bin/uvicorn app.main:app --reload
```

## StudyInstanceUID 조회

```bash
curl http://localhost:8042/studies  # → Study UID 리스트 확인 가능
```