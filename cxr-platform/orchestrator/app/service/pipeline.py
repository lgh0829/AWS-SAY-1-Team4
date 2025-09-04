# app/service/pipeline.py

from pathlib import Path
import logging
from app.service import orthanc, s3, sagemaker, convert, db
import pydicom
import numpy as np
import cv2

async def run_inference_pipeline(orthanc_id: str, dicom_study_uid: str, output_dir: Path) -> dict:
    # 1. DICOM 정보 조회 - Orthanc ID 사용
    instance_uid = await orthanc.get_instance_info(orthanc_id)
    if not instance_uid:
        raise ValueError(f"No instance found for Orthanc ID: {orthanc_id}")

    # 2. DICOM 다운로드 - Orthanc ID 사용
    dcm_path = await orthanc.download_dicom(instance_uid, output_dir)

    # 3. JPEG 변환
    jpg_path = output_dir / f"{dcm_path.stem}.jpg"
    convert.convert_dcm2jpg(dcm_path, jpg_path)

    # 4. S3 업로드 - DICOM StudyInstanceUID 사용
    s3.upload_file(local_path=jpg_path, study_uid=dicom_study_uid, subfolder="raw", filename=f"{instance_uid}.jpg")

    # 5. SageMaker 호출: segmentation
    result = sagemaker.invoke_segmentation(jpg_path, dicom_study_uid)
    logging.info(f"SageMaker segmentation result: {result}")

    # # 6. DB 저장
    # db.insert_result(study_uid, result)

    # 7. segmentation 결과 다운로드
    overlay_path = s3.download_file(result.overlay, output_dir)

    # 8. SageMaker 호출: classification
    classification_result = sagemaker.invoke_classification(overlay_path, jpg_path, dicom_study_uid)
    logging.info(f"SageMaker classification result: {classification_result}")

    # # 9. DB 업데이트
    # db.update_classification_result(study_uid, classification_result)

    # 11. XAI 결과 저장 및 DICOM 변환
    xai_path = s3.download_file(classification_result.xai, output_dir)
    dcm_path = convert.convert_jpg2dcm(
        xai_path,
        output_dir / f"{instance_uid}_xai.dcm", 
        dicom_study_uid,
        original_dcm_path=dcm_path  # 원본 DICOM 파일 경로 전달
    )
    logging.info(f"XAI result downloaded to: {xai_path}")
    logging.info(f"Overlay DICOM saved to: {dcm_path}")

    # 10. orthanc 업데이트 (기존 study_uid에 추가)
    modified_dcm_path = output_dir / f"{instance_uid}_xai_modified.dcm"
    modified_dcm_path = convert.modify_dicom_for_existing_study(dcm_path, modified_dcm_path, dicom_study_uid)
    
    # DICOM 태그에 AI 결과임을 표시
    ds = pydicom.dcmread(str(modified_dcm_path))
    ds.add_new([0x0071, 0x0001], "CS", "PROCESSED_BY_AI")
    ds.save_as(modified_dcm_path)
    
    # Orthanc 업로드
    await orthanc.upload_dicom(modified_dcm_path)
    
    # 처리 완료 표시 (추가)
    await orthanc.mark_study_as_processed(orthanc_id)
    
    return {"message": f"{dicom_study_uid} uploaded successfully"}

def visualize_and_upload_heatmap(raw_image, heatmap, bucket, key, alpha=0.7):
    # RGB로 변환
    raw_np = np.array(raw_image.convert("RGB"))
    H, W = raw_np.shape[:2]
    
    # 히트맵 리사이즈 및 정규화
    hm_resized = cv2.resize(heatmap, (W, H))
    
    # 마스킹: 배경 제거 (임계값 이하는 0으로)
    threshold = 0.2  # 더 높은 값으로 설정하여 배경 제거
    hm_masked = np.copy(hm_resized)
    hm_masked[hm_masked < threshold] = 0
    
    # 히트맵 변환 (0-255 스케일)
    hm_uint8 = np.uint8(255 * hm_masked)
    
    # JET 컬러맵 적용
    heatmap_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    
    # 원본 이미지와 히트맵 합성 (마스킹된 부분만)
    mask = (hm_masked > 0).astype(np.float32)[:,:,np.newaxis]
    overlay = raw_np * (1 - alpha * mask) + heatmap_colored * (alpha * mask)
    overlay = np.uint8(overlay)
    
    # 이미지 저장 및 업로드
    # ...나머지 코드는 동일