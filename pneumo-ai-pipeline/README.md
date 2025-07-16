# ML Pipeline 구조

```
my-dev-repo/
├── common/                          # 공통 라이브러리 폴더
│   ├── lung_utils/                  # 폐 관련 유틸리티 패키지
│   │   ├── __init__.py
│   │   ├── segmentation.py          # 폐 영역 분할 기능
│   │   ├── preprocessing.py         # 이미지 전처리 기능(CLAHE, blur 등)
│   │   ├── data_handling.py         # 데이터 로딩/저장/변환 기능
│   │   └── visualization.py         # 결과 시각화 도구
│   │
│   └── cloud_utils/                 # 클라우드 관련 유틸리티
│       ├── __init__.py
│       ├── s3_handler.py            # S3 연동 기능
│       └── sagemaker_utils.py       # SageMaker 연동 기능
│
├── training/                        # 학습 관련 코드
│   ├── scripts/                     # 실행 스크립트
│   │   ├── prepare_dataset.py       # 데이터셋 준비
│   │   ├── sagemaker_train.py       # SageMaker 학습 실행
│   │   └── mlflow_logger.py         # MLflow 연동
│   │
│   └── src/                         # 소스 코드
│       ├── train.py                 # 모델 훈련 코드
│       ├── evaluate.py              # 모델 평가 코드
│       └── data_loader.py           # 데이터 로더
│
├── inference/                       # 추론 관련 코드
│   ├── scripts/                     # 실행 스크립트
│   │   ├── deploy_model.py          # 모델 배포
│   │   └── batch_inference.py       # 배치 추론
│   │
│   └── src/                         # 소스 코드
│       ├── predict.py               # 개별 예측 로직
│       └── postprocessing.py        # 예측 후처리
│
├── configs/                         # 설정 파일
│   ├── config.yaml                  # 기본 설정
│   ├── train_config.yaml            # 훈련 관련 설정
│   └── inference_config.yaml        # 추론 관련 설정
│
├── requirements.txt                 # 프로젝트 의존성
└── setup.py                         # 패키지 설치용
```


## prepare_dataset.py

1. 작업 순서를 변경하여 원본 이미지에 폐 분할을 먼저 적용하도록 했습니다
2. 분할된 폐 이미지를 S3에 업로드하는 코드를 추가했습니다
3. 전처리는 원본 이미지가 아닌 분할된 이미지에 적용하도록 변경했습니다
4. 최종 전처리된 이미지를 S3에 업로드하는 코드를 추가했습니다
5. 이전의 선택적 플래그(segmentation, S3 업로드)는 이제 필수 워크플로우 단계이므로 제거했습니다
6. 출력 디렉토리가 확실히 존재하도록 os.makedirs()를 사용하여 디렉토리 생성을 추가했습니다