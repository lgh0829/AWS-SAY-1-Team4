# 베이스 이미지 (CUDA 미사용 환경 기준)
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY app/ app/

# 포트 노출 (FastAPI 기본 포트)
EXPOSE 8000

# 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]