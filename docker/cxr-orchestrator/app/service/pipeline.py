# app/service/pipeline.py

from pathlib import Path
from app.service import orthanc, s3, sagemaker, dcm2jpg, db

async def run_inference_pipeline(study_uid: str, output_dir: Path) -> dict:
    # 1. DICOM 정보 조회
    instance_uid = await orthanc.get_instance_info(study_uid)
    if not instance_uid:
        raise ValueError(f"No instance found for StudyInstanceUID: {study_uid}")

    # 2. DICOM 다운로드
    dcm_path = await orthanc.download_dicom(instance_uid, output_dir)

    # 3. JPEG 변환
    jpg_path = output_dir / f"{dcm_path.stem}.jpg"
    dcm2jpg.convert_dcm2jpg(dcm_path, jpg_path)

    # 4. S3 업로드
    s3.upload_file(local_path=jpg_path, study_uid=study_uid, subfolder="raw", filename=f"{instance_uid}.jpg")

    # 5. SageMaker 호출: segmentation
    result = sagemaker.invoke_segmentation(jpg_path, study_uid)

    # 6. DB 저장
    db.insert_result(study_uid, result)

    # 7. SageMaker 호출: classification
    classification_result = sagemaker.invoke_classification(jpg_path)

    # 8. DB 업데이트
    db.update_classification_result(study_uid, classification_result)

    return result