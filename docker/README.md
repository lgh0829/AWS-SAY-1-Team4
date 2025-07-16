# 디렉토리 구조

```
your-repo/
├── docker/
│   ├── install-docker.sh
│   ├── docker-compose.yml          # 핵심 구성
│   ├── .env                        # (선택) 환경변수 파일
│   ├── ohif/
│   │   └── config/
│   │       └── config.json         # OHIF 설정
│   ├── orthanc/
│   │   ├── orthanc.json            # (선택) Orthanc 설정
│   │   └── data/                   # 볼륨 mount용 폴더 (빈 채로 두세요)
│   └── README.md                   # Docker 실행 가이드
│
├── fastapi/                        # (선택) FastAPI 백엔드
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── .gitignore
└── README.md                       # 전체 프로젝트 설명
```

| 폴더/파일 경로              | 용도                                                |
|----------------------------|-----------------------------------------------------|
| `docker/`                  | Docker 관련 파일을 모두 모아 관리하는 핵심 폴더     |
| `docker/ohif/config/config.json` | OHIF의 DICOMweb 서버 주소 설정                     |
| `docker/orthanc/`          | Orthanc 설정 (플러그인 포함) 및 데이터 디렉토리     |
| `fastapi/`                 | 필요한 경우 커스텀 API 서비스 (예: AI inference 등) |
| `README.md` (루트)         | 프로젝트 개요 / 사용법 전체 요약                   |
| `docker/README.md`         | Docker 기반 실행법, 개발 환경 구성 설명             |

# docker 설치 가이드

docker가 설치되어 있지 않다면 [docker 설치 스크립트](install-docker.sh) 를 실행시켜줍니다.

1. 스크립트 파일에 실행 권한을 부여합니다. `chmod +x install-docker.sh`
2. 스크립트를 실행합니다. `your-path/docker/install-docker.sh`
3. docker 설치를 적용합니다. `newgrp docker`

# Git clone 가이드


이 프로젝트는 `docker/`, `EDA/`, `utils/` 등 서로 다른 리소스 환경에서 사용하는 디렉토리로 구성되어 있습니다.  
전체 리포를 클론하지 않고 필요한 디렉토리만 선택적으로 가져오려면 `sparse checkout` 기능을 사용할 수 있습니다.

`docker/` 디렉토리만 클론

```bash
git clone --filter=blob:none --sparse https://github.com/your-org/project.git
cd project
git sparse-checkout init --cone
git sparse-checkout set docker
cd docker
./install-docker.sh