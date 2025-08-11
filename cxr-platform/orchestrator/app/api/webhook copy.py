# app/api/webhook.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
import logging
from pathlib import Path
import asyncio

from app.service import pipeline, orthanc
from app.model.schema import WebhookPayload  # payload: StudyInstanceUID, OrthancStudyID, Status

router = APIRouter()
log = logging.getLogger(__name__)

# ---------- ai_state helpers ----------
async def set_ai_state(orthanc_study_id: str, state: str) -> None:
    """
    Orthanc 메타데이터에 ai_state를 기록합니다.
    상태: queued | processing | done | error
    """
    try:
        # 권장: service.orthanc에 set_ai_state가 있다면 그것을 사용
        if hasattr(orthanc, "set_ai_state"):
            await orthanc.set_ai_state(orthanc_study_id, state)
            return

        # 레거시 호환: 기존에 사용하던 함수가 있다면 매핑
        if state == "processing" and hasattr(orthanc, "mark_study_as_processing"):
            await orthanc.mark_study_as_processing(orthanc_study_id); return
        if state == "done" and hasattr(orthanc, "mark_study_as_processed"):
            await orthanc.mark_study_as_processed(orthanc_study_id); return
        if state == "error" and hasattr(orthanc, "mark_study_as_error"):
            await orthanc.mark_study_as_error(orthanc_study_id); return

        # 최후수단: set_metadata가 있다면 직접 키/값 저장
        if hasattr(orthanc, "set_metadata"):
            await orthanc.set_metadata(orthanc_study_id, "ai_state", state)
            return

        log.warning("No suitable method to set ai_state on Orthanc was found.")
    except Exception as e:
        log.error(f"Failed to set ai_state={state} on {orthanc_study_id}: {e}", exc_info=True)
        # 상태 기록 실패는 치명적이진 않지만, 재시도 정책에 영향 줄 수 있음

async def run_pipeline_bg(dicom_study_uid: str, orthanc_study_id: str, output_dir: Path):
    """
    백그라운드에서 실제 추론 파이프라인을 수행.
    queued -> processing -> done / error 전이를 기록.
    """
    try:
        await set_ai_state(orthanc_study_id, "processing")

        # 필요 시 여기서 reconstruct 등 전/후처리를 수행 가능
        result = await pipeline.run_inference_pipeline(
            orthanc_id=orthanc_study_id,
            dicom_study_uid=dicom_study_uid,
            output_dir=output_dir
        )

        await set_ai_state(orthanc_study_id, "done")
        log.info(f"[pipeline] done: {dicom_study_uid} ({orthanc_study_id}) -> {result}")
    except Exception as e:
        await set_ai_state(orthanc_study_id, "error")
        log.error(f"[pipeline] error on {dicom_study_uid} ({orthanc_study_id}): {e}", exc_info=True)

# ---------- webhook endpoint ----------
@router.post("/on-store", response_class=PlainTextResponse, status_code=202)
async def on_store_webhook(payload: WebhookPayload):
    """
    Lua(webhook.lua)가 OnStableStudy에서 단 1회 호출.
    - Lua는 호출 전에 ai_state=queued로 Orthanc에 기록
    - 서버는 즉시 202(ACCEPTED)를 반환하고, 백그라운드에서 추론
    - 진행/완료 상태는 서버가 ai_state=processing/done/error로 기록
    """
    try:
        if not payload.StudyInstanceUID:
            raise HTTPException(status_code=400, detail="Missing StudyInstanceUID")
        if not payload.OrthancStudyID:
            raise HTTPException(status_code=400, detail="Missing OrthancStudyID")

        study_uid = payload.StudyInstanceUID
        orthanc_id = payload.OrthancStudyID

        log.info(f"[webhook] ACCEPTED StudyInstanceUID={study_uid}, OrthancStudyID={orthanc_id}")

        # 출력 폴더 준비
        output_dir = Path(__file__).parent / "temp"
        output_dir.mkdir(exist_ok=True)

        # 백그라운드 태스크로 파이프라인 실행 (FastAPI event loop에 스케줄)
        asyncio.create_task(run_pipeline_bg(study_uid, orthanc_id, output_dir))

        # 비동기 수락 응답 (Lua는 이 응답만 보고 종료)
        return PlainTextResponse("ACCEPTED", status_code=202)

    except HTTPException:
        raise
    except Exception as e:
        # 인입 자체 실패는 500으로 반환 (Lua는 실패 시 ai_state=error 로 전환)
        log.error(f"[webhook] failed to accept: {e}", exc_info=True)
        return PlainTextResponse(str(e), status_code=500)