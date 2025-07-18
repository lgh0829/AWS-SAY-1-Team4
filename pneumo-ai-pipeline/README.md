# ML Pipeline 구조

```
penumo-ai-pipeline
├── common  # 공통 라이브러리
├── configs # 설정 파일
│   ├── inference_config.yaml
│   ├── prepare_config.yaml
│   └── train_config.yaml
├── models  # sagamaker training-job 입력 파일
│   ├── requirements.txt
│   ├── resnet34.py
│   └── resnet50.py
├── pipeline # 파이프라인 파일
│   ├── inference.py
│   ├── prepare_dataset.py
│   └── train.py
├── requirements-prepare.txt # 데이터셋 준비용 가상환경 의존성 관리
└── requirements-train.txt   # 훈련 시작용 가상환경 의존성 관리
```

## prepare_dataset.py

1. 원본 이미지에 폐 분할 적용
2. 분할된 이미지에 전처리 적용
3. 폐 분할, 전처리 이미지 S3 업로드

## train.py

SageMaker 통합:

SageMaker의 환경변수를 사용하여 데이터 경로와 모델 저장 경로를 설정
argparse를 통해 하이퍼파라미터 설정 가능
데이터 로딩:

SageMaker의 channel을 통해 전달된 데이터 경로 사용
train, validation, test 데이터셋 각각 설정
MLflow 통합:

Tensorboard 대신 MLflow로 지표 기록
실험 이름과 tracking URI는 환경변수에서 가져옴
모델 아티팩트 저장 및 로깅
하이퍼파라미터 설정:

모델 타입, 에포크 수, 배치 크기, 학습률 등을 인자로 받아 설정
early stopping patience 설정 가능

# 성능 평가 지표

## 일반

|지표|설명|
|---|---|
|Accuracy (정확도)|전체 중 맞게 예측한 비율|
|Precision (정밀도)|Positive 예측 중 실제 Positive인 비율 = TP / (TP + FP)|
|Recall (재현율, 민감도)|실제 Positive 중 모델이 Positive라고 예측한 비율 = TP / (TP + FN)|
|F1 Score|정밀도와 재현율의 조화 평균 = 2 × (Precision × Recall) / (Precision + Recall)|
|ROC-AUC|분류 임계값 변화에 따른 TPR vs FPR 곡선 아래 면적|
|PR-AUC|Precision vs Recall 곡선 아래 면적 (불균형 데이터에서 유용)|
|Specificity (특이도)|실제 Negative 중 Negative로 예측한 비율 = TN / (TN + FP)|

## Multi-Class Classification

|지표|설명|
|---|---|
|Macro F1 / Micro F1|클래스별 F1 평균 (Macro: 클래스별 평균 / Micro: 전체 평균)|
|Confusion Matrix|각 클래스 간 예측 결과를 행렬로 나타냄|
|Top-k Accuracy|정답이 상위 k개의 예측 결과 중 하나에 포함되는 비율|

## 의료/헬스케어 특화 지표
|지표|설명|
|---|---|
|Youden’s Index|TPR + TNR - 1 (임계값 선택에 사용)|
|Balanced Accuracy|(Sensitivity + Specificity) / 2|
|Dice Coefficient|주로 segmentation 모델에서 예측 mask와 정답 mask의 유사도 (IoU와 유사)|
