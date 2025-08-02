# AWS SageMaker Endpoint 배포 가이드

이 문서는 CXR(Chest X-ray) 영상 기반 폐 분할(Lung Segmentation) 및 폐렴 분류(Pneumonia Classification) 모델을 AWS SageMaker 엔드포인트로 배포하는 방법을 안내합니다.

## BYOM (Bring Your Own Model)

본 프로젝트는 SageMaker PyTorch 환경에서 사전 학습된 모델을 사용자 정의 코드와 함께 배포하는 BYOM 방식을 따릅니다. 자세한 내용은 [공식 문서](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#bring-your-own-model)를 참고하세요.

## Segmentation 모델 배포

### 모델 구성

`endpoint_seg` 디렉토리는 Amazon SageMaker에 추론용 PyTorch 모델을 BYOM(Bring Your Own Model) 방식으로 배포하기 위한 구조로 구성되어 있습니다.

```bash
endpoint_seg
├── code
│   ├── inference.py        # SageMaker용 진입점 스크립트
│   ├── models
│   │   ├── __init__.py
│   │   ├── resnet.py       # ResNet 모델 정의
│   │   └── unet.py         # UNet 모델 정의
│   └── requirements.txt    # 추론 시 필요한 패키지 명세
├── resnet34-333f7ec4.pth   # 사전학습된 ResNet34 모델 가중치
└── resnet34.pth            # 사전학습된 UNet+ResNet34 기반 폐 분할 모델 가중치
```

### 사전 학습 가중치

- [`resnet34.pth`](https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/resnet34.pth): `lungs_segmentation` 프로젝트 기반의 UNet 모델로, ResNet34를 encoder로 사용하여 폐 분할 작업에 최적화되어 있습니다.
- [`resnet34-333f7ec4.pth`](https://download.pytorch.org/models/resnet34-333f7ec4.pth): PyTorch에서 제공하는 ImageNet 기반의 ResNet34 사전학습 가중치로, encoder의 백본 모델로 활용됩니다.

### 엔드포인트 배포 예제

엔드포인트 배포를 위해 `config_deploy.yaml` 파일에서 `inference_type`을 `"segmentation"`으로 설정한 뒤, `endpoint-seg` 디렉토리를 압축하여 Amazon S3에 업로드하고, 이를 기반으로 SageMaker Endpoint를 생성할 수 있습니다.

다음은 설정 예시입니다:

```yaml
inference_type: "segmentation"         # 추론 타입 지정
base_job_name: "your-job-name"         # SageMaker 작업 이름

S3:
  prefix: "cxr-pneumonia-4/models/"    # 압축 파일 S3 저장 경로 prefix
  output_path: null                    # output_path 미지정

Sagemaker:
  instance_type: "ml.m5.large"         # 인스턴스 유형
  instance_count: 1                    # 인스턴스 개수
  framework_version: "1.13.1"          # PyTorch 프레임워크 버전
  py_version: "py39"                   # Python 버전

local:
  files_dir: "endpoint_seg"            # 압축할 로컬 디렉토리 이름
```

설정을 완료한 후 다음 명령어를 실행하여 SageMaker 엔드포인트를 생성합니다:

```bash
python deploy_endpoint.py --config config_deploy.yaml
```

### 엔드포인트 테스트

엔드포인트 테스트를 위해 배포가 완료된 엔드포인트 이름을 확인한 후, [call_endpoint_seg.py](test/call_endpoint_seg.py) 스크립트의 `endpoint_name` 변수에 해당 이름을 입력하고 실행합니다.

```bash
python test/call_endpoint_seg.py
```

## Classification 모델 배포

### 모델 구성

`endpoint_classification` 디렉토리는 CXR 영상을 기반으로 폐렴을 분류하는 모델을 SageMaker에 배포하기 위한 구조로 구성되어 있습니다.

```bash
endpoint_classification
├── inference.py        # SageMaker용 진입점 스크립트
└── requirements.txt    # 추론 시 필요한 패키지 명세
```

### 사전 학습 가중치

[pneumo-ai-pipeline](../pneumo-ai-pipeline/) 를 통해 Amazon SageMaker Training job으로 훈련한 가중치를 사용합니다.

### 엔드포인트 배포 예제
엔드포인트 배포를 위해 `config_deploy.yaml` 파일에서 `inference_type`을 `"classification"`으로 설정한 뒤, Amazon S3에 저장되어있는 가중치와 `endpoint_classification` 디렉토리에서 추론에 필요한 스크립트로 SageMaker Endpoint를 생성할 수 있습니다.

다음은 설정 예시입니다.
```yaml
inference_type: "classification"            # 추론 타입 지정
base_job_name: "your-job-name"              # SageMaker 작업

S3:
  prefix: null                              # 압축 파일 S3 저장 경로 미지정
  output_path: "output/your-model.tar.gz"   # S3에 저장된 모델 가중치 경로

Sagemaker:
  instance_type: "ml.m5.large"              # 인스턴스 유형
  instance_count: 1                         # 인스턴스 개수
  framework_version: "1.13.1"               # PyTorch 프레임워크 버전
  py_version: "py39"                        # Python 버전

local:
  files_dir: "endpoint_classification"      # 압축할 로컬 디렉토리 이름
```

설정을 완료한 후 다음 명령어를 실행하여 SageMaker 엔드포인트를 생성합니다:

```bash
python deploy_endpoint.py --config config_deploy.yaml
```

### 엔드포인트 테스트

엔드포인트 테스트를 위해 배포가 완료된 엔드포인트 이름을 확인한 후, [call_endpoint_classification_1.py](test/call_endpoint_classification_1.py) 스크립트의 `endpoint_name` 변수에 해당 이름을 입력하고 실행합니다.

```bash
python test/call_endpoint_classification_1.py
```

## 엔드포인트 삭제

[delete_endpoint.ipynb](delete_endpoint.ipynb) 노트북을 실행하여 현재 실행 중인 SageMaker 엔드포인트를 삭제할 수 있습니다.

---

이 문서는 AWS SageMaker를 이용한 BYOM 기반 CXR AI 모델 배포 워크플로우를 예시로 설명합니다.  
프로젝트 목적에 맞게 YAML 설정이나 모델 구조를 자유롭게 확장할 수 있습니다.