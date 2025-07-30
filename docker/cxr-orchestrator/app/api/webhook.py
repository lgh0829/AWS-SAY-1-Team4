from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
import logging

# 필요한 서비스들 (아직 스텁)
from app.service import orthanc, s3, sagemaker, bedrock, db

router = APIRouter()

@router.post("/on-store", status_code=status.HTTP_202_ACCEPTED)
async def on_store_webhook(request: Request):
    try:
        payload = await request.json()
        study_uid = payload.get("StudyInstanceUID")
        logging.info(f"Received webhook for StudyInstanceUID: {study_uid}")

        # TODO: 파이프라인 호출 로직 연결
        # 예시 순서:
        # - orthanc.get_instance()
        # - s3.upload_file(...)
        # - sagemaker.invoke_segmentation(...)
        # - ...
        # - db.insert_result(...)

        return JSONResponse(content={"message": f"Webhook received for study {study_uid}"})
    
    except Exception as e:
        logging.error(f"Webhook 처리 중 에러 발생: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})