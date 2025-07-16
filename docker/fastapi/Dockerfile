# 베이스 이미지로 Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY ./app ./app

# 컨테이너가 실행될 때 실행할 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]