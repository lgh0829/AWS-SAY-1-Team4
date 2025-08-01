from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

# 필요한 서비스들 (아직 스텁)
from app.service import orthanc, s3, sagemaker, bedrock, db, dcm2jpg

router = APIRouter()

@router.post("/on-store", status_code=status.HTTP_202_ACCEPTED)
async def on_store_webhook(request: Request):
    try:
        payload = await request.json()
        study_uid = payload.get("StudyInstanceUID")
        logging.info(f"Received webhook for StudyInstanceUID: {study_uid}")

        # TODO: 파이프라인 호출 로직 연결
        # 예시 순서:
        # - s3.upload_file(...)
        # - sagemaker.invoke_segmentation(...)
        # - ...
        # - db.insert_result(...)

        # DICOM 인스턴스 정보 가져오기
        instance_uid = await orthanc.get_instance_info(study_uid)
        if not instance_uid:
            logging.error(f"No instance found for StudyInstanceUID: {study_uid}")
            return JSONResponse(status_code=404, content={"error": "Instance not found"})
        
        # DICOM to JPEG 변환 및 다운로드
        output_dir = Path("app/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        dcm_path = await orthanc.download_dicom(instance_uid, output_dir)
        jpg_path = output_dir / f"{dcm_path.stem}.jpg"
        dcm2jpg.convert_dcm2jpg(dcm_path, jpg_path)

        # s3에 JPEG 파일 업로드
        s3.upload_file(local_path=jpg_path, study_uid=study_uid, subfolder="raw", filename=f"{instance_uid}.jpg")

        # SageMaker endpoint 호출

        # 여기에 파이프라인 로직 추가 예정
        return JSONResponse(content={"message": f"Webhook received for study {study_uid}"})

    except Exception as e:
        logging.error(f"Webhook 처리 중 에러 발생: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})