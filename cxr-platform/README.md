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
├── cxr-orchestrator/                        # (선택) FastAPI 백엔드
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
````

# 실행 가이드
[docker compose](docker-compose.yml)로 OHIF Viewer와 Orthanc 서버 컨테이너를 실행합니다.

1. ohif-orthanc 디렉토리로 이동합니다. `cd your-path/docker/ohif-orthanc`
2. OHIF + Orthanc를 실행합니다. `docker-compose up -d`
3. `docker ps` 명령어를 입력했을 때 다음 처럼 jodogne/orthanc-plugins, ohif/viewer가 실행되고 있으면 성공적으로 실행된 것입니다.
```
CONTAINER ID   IMAGE                     ...         PORTS                                      NAMES
abcdef123456   jodogne/orthanc-plugins   ...   0.0.0.0:8042->8042/tcp, 0.0.0.0:4242->4242/tcp   orthanc
123456abcdef   ohif/viewer               ...   0.0.0.0:80->80/tcp                               ohif
```
4. 컨테이너에서 사용하고 있는 port에 대한 인바운드 규칙을 설정해줍니다.

| 유형       | 프로토콜 | 포트 범위 | 소스       | 설명               |
|------------|----------|-----------|------------|--------------------|
| HTTP       | TCP      | 80        | 0.0.0.0/0  | OHIF Viewer        |
| Custom TCP | TCP      | 8042      | 0.0.0.0/0  | Orthanc Web UI     |
| Custom TCP | TCP      | 4242      | 0.0.0.0/0  | DICOM 전송 포트    |

5. http://<your-public-ip>로 접속하면 OHIF Web Viewer가 로드됩니다.
6. DICOM 파일을 업로드하려면 http://<your-public-ip>:8042 Orthanc Web UI에서 우측 상단 'upload' 버튼을 참고하세요.

# Docker 종료
1. ohif-orthanc 디렉토리로 이동합니다. `cd your-path/docker/ohif-orthanc`
2. `docker-compose down` 명령어를 입력합니다.
3. `docker ps` 명령어를 입력해서 실행중인 컨테이너 유무를 확인합니다.

# Docker compose files

# Build

Using docker compose you can build the image with the following command:

```bash
docker-compose build
```

# Run

To run the container use the following command:

```bash
docker-compose up
```


# Routes

http://localhost/ -> OHIF
localhost/pacs -> Orthanc


See [here](../../../docs/docs/deployment/nginx--image-archive.md) for more information about this recipe.



```
root@cxr-orchestrator:/app# curl http://orthanc_dev:8042/studies?expand=true
[
   {
      "ID" : "aacd5698-67236ab1-cb89c24f-0cf3bf2f-046d0d07",
      "IsStable" : true,
      "Labels" : [],
      "LastUpdate" : "20250805T052758",
      "MainDicomTags" : {
         "AccessionNumber" : "",
         "ReferringPhysicianName" : "",
         "StudyDate" : "19010101",
         "StudyID" : "",
         "StudyInstanceUID" : "1.2.276.0.7230010.3.1.2.8323329.10118.1517874346.924222",
         "StudyTime" : "000000.00"
      },
      "ParentPatient" : "9aa3172f-bb97d71a-1f920986-55da17d1-5d9d8477",
      "PatientMainDicomTags" : {
         "PatientBirthDate" : "",
         "PatientID" : "0000a175-0e68-4ca4-b1af-167204a7e0bc",
         "PatientName" : "0000a175-0e68-4ca4-b1af-167204a7e0bc",
         "PatientSex" : "F"
      },
      "Series" : [ "7114ad61-f413f140-c09e427f-43ddd522-13c0eb9e" ],
      "Type" : "Study"
   },
   {
      "ID" : "5f11865f-12b55213-e72c836f-f8026305-2af3199c",
      "IsStable" : true,
      "Labels" : [],
      "LastUpdate" : "20250805T052758",
      "MainDicomTags" : {
         "AccessionNumber" : "",
         "ReferringPhysicianName" : "",
         "StudyDate" : "19010101",
         "StudyID" : "",
         "StudyInstanceUID" : "1.2.276.0.7230010.3.1.2.8323329.25090.1517874463.16029",
         "StudyTime" : "000000.00"
      },
      "ParentPatient" : "bbf606de-1fa68e9d-3b57a8d0-84a8df81-8e92eea0",
      "PatientMainDicomTags" : {
         "PatientBirthDate" : "",
         "PatientID" : "000fe35a-2649-43d4-b027-e67796d412e0",
         "PatientName" : "000fe35a-2649-43d4-b027-e67796d412e0",
         "PatientSex" : "M"
      },
      "Series" : [ "87156a74-43c2122c-721a4024-07e57ba6-fd711450" ],
      "Type" : "Study"
   },
   {
      "ID" : "704b3431-a26acfaf-8c62d921-78fa32f0-1e39abab",
      "IsStable" : true,
      "Labels" : [],
      "LastUpdate" : "20250805T052758",
      "MainDicomTags" : {
         "AccessionNumber" : "",
         "ReferringPhysicianName" : "",
         "StudyDate" : "19010101",
         "StudyID" : "",
         "StudyInstanceUID" : "1.2.276.0.7230010.3.1.2.8323329.4475.1517874307.936344",
         "StudyTime" : "000000.00"
      },
      "ParentPatient" : "670f9fd5-a76657d7-eba1a5ab-53703204-6fd4afeb",
      "PatientMainDicomTags" : {
         "PatientBirthDate" : "",
         "PatientID" : "000db696-cf54-4385-b10b-6b16fbb3f985",
         "PatientName" : "000db696-cf54-4385-b10b-6b16fbb3f985",
         "PatientSex" : "F"
      },
      "Series" : [ "d725636b-2481d65d-c2157335-8e456a5a-d3c7d7de" ],
      "Type" : "Study"
   },
   {
      "ID" : "c1d32b5b-af8d2994-709cb284-1729c06c-0e634d94",
      "IsStable" : true,
      "Labels" : [],
      "LastUpdate" : "20250808T012335",
      "MainDicomTags" : {
         "AccessionNumber" : "53189527",
         "ReferringPhysicianName" : "",
         "StudyDate" : "21800626",
         "StudyID" : "53189527",
         "StudyInstanceUID" : "2.25.6521350286201891923192108632889001416",
         "StudyTime" : "165500.312"
      },
      "ParentPatient" : "cdcd1e52-84d66be7-6696da54-04675dba-7498a393",
      "PatientMainDicomTags" : {
         "PatientBirthDate" : "",
         "PatientID" : "10000032",
         "PatientName" : "",
         "PatientSex" : ""
      },
      "Series" : [ "6b63fbdc-8a190a71-9809e7d0-79bb91c6-ae38b6b8" ],
      "Type" : "Study"
   }
]
```