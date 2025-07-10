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