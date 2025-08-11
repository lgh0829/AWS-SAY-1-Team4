from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import webhook  # 라우터 등록

# FastAPI 애플리케이션 생성
# API 문서화 자동 지원 (/docs)
app = FastAPI(
    title="CXR AI Orchestrator",
    description="Handles Orthanc webhook, SageMaker pipeline, S3, and Bedrock report generation",
    version="1.0.0"
)

# CORS 설정 (필요시 수정)
# OHIF Viewer 또는 외부 호출 대응
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 실제 배포 시엔 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 헬스 체크용 기본 URL
@app.get("/")
async def root():
    return {"message": "CXR Orchestrator is running"}

# (로컬 개발용) uvicorn으로 직접 실행하는 경우
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)