# app/service/pipeline.py

from pathlib import Path
import logging
from app.service import orthanc, s3, sagemaker, convert, db
import pydicom

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

    # =========================
    # [KB/RAG add] 여기부터
    # =========================
    try:
        # 12. 분류 결과 정리
        #    - classification_result는 팀 유틸에 따라 dict/attr일 수 있으니 방어적으로 파싱
        cls_name = getattr(classification_result, "class_name", None) or classification_result.get("class_name")
        cls_conf = getattr(classification_result, "confidence", None) or classification_result.get("confidence")
        probs     = getattr(classification_result, "probabilities", None) or classification_result.get("probabilities", {})

        # 안전 가드
        if cls_name is None:
            raise RuntimeError("classification_result does not contain class_name")
        if cls_conf is None:
            cls_conf = 0.0

        # 13. 쿼리 텍스트 생성(간단 버전: 라벨 기반)
        #     - quadrant 등 고급 로직은 이미 kbretrieval.generate_report_with_sonnet에서
        #       qconf/quad_phrase를 반영할 수 있으므로, 1차 버전은 라벨 중심으로만.
        label_pred = "other" if cls_name == "not_normal" else cls_name
        keywords: List[str] = []  # 필요 시 설정/전략에 맞춰 키워드 추가
        query_text = "; ".join([p for p in [label_pred] + keywords if p])

        # 14. KB 메타데이터 필터(label 기반)
        metadata_filter: Optional[Dict[str, Any]] = None
        if label_pred in ("pneumonia", "normal"):
            metadata_filter = {"equals": {"key": "label", "value": label_pred}}

        # 15. KB 검색
        neighbors = kb_retrieve(
            query_text=query_text,
            kb_id=settings.BEDROCK_KB_ID,
            k=5,
            metadata_filter=metadata_filter
        )

        # 16. Sonnet 3.7로 리포트 생성(JSON 강제)
        #     - quadrant_pred, qconf는 현재 파이프라인에선 별도 산출 없으므로 None
        report_obj = generate_report_with_sonnet(
            label_pred=label_pred,
            quadrant_pred=None,
            qconf=None,
            query_text=query_text,
            neighbors=neighbors,
            model_id=settings.BEDROCK_KB_MODEL_ID
        )

        # 17. 결과 저장(Orthanc 메타 + S3)
        #     17-1) Orthanc 메타
        try:
            await orthanc.set_metadata(orthanc_id, "ai_report_json", json.dumps(report_obj))
            await orthanc.set_metadata(orthanc_id, "ai_retrieval_neighbors", json.dumps(neighbors))
        except Exception as e:
            logging.warning(f"Failed to write report JSON to Orthanc metadata: {e}")

        #     17-2) S3에도 백업 (선택)
        #     s3.upload_file() 유틸이 파일 기반이므로 임시 파일로 저장 후 업로드
        try:
            tmp_report = output_dir / f"{instance_uid}_report.json"
            with open(tmp_report, "w", encoding="utf-8") as f:
                json.dump(report_obj, f, ensure_ascii=False)
            s3.upload_file(local_path=tmp_report, study_uid=dicom_study_uid, subfolder="report", filename=f"{instance_uid}_report.json")
        except Exception as e:
            logging.warning(f"Failed to upload report to S3: {e}")

        rag_payload = {
            "retrieval": {
                "engine": "bedrock-kb",
                "kb_id": settings.BEDROCK_KB_ID,
                "query_text": query_text,
                "metadata_filter": metadata_filter,
                "neighbors": neighbors
            },
            "generation": report_obj
        }
    except Exception as e:
        logging.error(f"[KB/RAG] Failed: {e}", exc_info=True)
        # 실패하더라도 본 파이프라인의 기존 결과는 반환
        rag_payload = {"error": str(e)}
    # =========================
    # [KB/RAG add] 여기까지
    # =========================

    return {
        "message": f"{dicom_study_uid} uploaded successfully",
        "prediction": {
            "label_pred": cls_name,
            "confidence": cls_conf,
            "probabilities": probs,
            "xai_jpg": str(xai_path)
        },
        "segmentation": {
            "overlay_jpg": str(overlay_path),
            "raw_jpg": str(jpg_path)
        },
        **rag_payload  # [KB/RAG add] RAG 결과를 최상위에 병합
    }
