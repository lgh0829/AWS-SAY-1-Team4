# app/api/webhook.py
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, JSONResponse
import logging
from pathlib import Path
import asyncio
import os
import json
from functools import lru_cache
import time

from app.service import pipeline, orthanc
from app.model.schema import WebhookPayload, WebhookResponse  # payload: StudyInstanceUID, OrthancStudyID, Status

router = APIRouter()
log = logging.getLogger(__name__)

# 처리 중/완료된 스터디 캐시 (5분 유효)
processed_studies = {}

def is_recently_processed(study_uid: str) -> bool:
    """메모리 캐시에서 최근 처리 여부 확인"""
    now = time.time()
    if study_uid in processed_studies:
        timestamp = processed_studies[study_uid]
        # 5분(300초) 이내 처리된 경우
        if now - timestamp < 300:
            return True
    return False

def mark_as_processed_in_cache(study_uid: str):
    """메모리 캐시에 처리 완료 표시"""
    processed_studies[study_uid] = time.time()

# ---------- ai_state read helper ----------
async def get_ai_state(orthanc_study_id: str) -> str | None:
    """
    Orthanc 메타데이터에서 ai_state를 읽어옵니다.
    없으면 None을 반환합니다.
    """
    try:
        # 우선 service.orthanc에 전용 함수가 있다면 사용
        if hasattr(orthanc, "get_ai_state"):
            return await orthanc.get_ai_state(orthanc_study_id)

        # 레거시: get_metadata가 있다면 키로 조회
        if hasattr(orthanc, "get_metadata"):
            v = await orthanc.get_metadata(orthanc_study_id, "ai_state")
            return v

        # 사용 가능한 리더가 없다면 None
        log.warning("No method to read ai_state from Orthanc was found.")
        return None
    except Exception as e:
        log.error(f"Failed to get ai_state on {orthanc_study_id}: {e}", exc_info=True)
        return None

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
        log.info(f"[pipeline] set ai_state=processing ({orthanc_study_id})")
        await set_ai_state(orthanc_study_id, "processing")

        # 필요 시 여기서 reconstruct 등 전/후처리를 수행 가능
        result = await pipeline.run_inference_pipeline(
            orthanc_id=orthanc_study_id,
            dicom_study_uid=dicom_study_uid,
            output_dir=output_dir
        )

        await set_ai_state(orthanc_study_id, "done")
        log.info(f"[pipeline] set ai_state=done ({orthanc_study_id})")
        log.info(f"[pipeline] done: {dicom_study_uid} ({orthanc_study_id}) -> {result}")
    except Exception as e:
        log.warning(f"[pipeline] set ai_state=error ({orthanc_study_id}) due to exception")
        await set_ai_state(orthanc_study_id, "error")
        log.error(f"[pipeline] error on {dicom_study_uid} ({orthanc_study_id}): {e}", exc_info=True)

# ---------- webhook endpoint ----------
@router.post("/on-store", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def on_store_webhook(payload: WebhookPayload):
    """
    Orthanc에서 DICOM 스터디가 저장되었을 때 호출되는 웹훅
    """
    study_uid = payload.StudyInstanceUID
    
    # 메모리 캐시 확인
    if is_recently_processed(study_uid):
        logging.info(f"Study recently processed (cached), skipping: {study_uid}")
        return {"message": f"Study already processed (cached): {study_uid}"}
    
    try:
        orthanc_id = payload.OrthancStudyID

        # 3) 중복 가드(이미 큐/실행/완료면 재실행 방지)
        current_state = await get_ai_state(orthanc_id)
        if current_state in {"queued", "processing", "done"}:
            log.info(f"[webhook] skip queueing; ai_state={current_state} (orthanc_id={orthanc_id})")
            return PlainTextResponse("ALREADY_QUEUED_OR_DONE", status_code=202)

        log.info(f"[webhook] ACCEPTED uid={study_uid}, orthanc_id={orthanc_id}")

        # 4) 출력 폴더 준비
        output_dir = Path(__file__).parent / "temp"
        output_dir.mkdir(exist_ok=True)

        # 5) 상태 없으면 queued로 세팅 (Lua 누락 대비)
        if not current_state:
            await set_ai_state(orthanc_id, "queued")
            log.info(f"[webhook] set ai_state=queued (orthanc_id={orthanc_id})")

        # 6) 비동기 파이프라인 실행
        asyncio.create_task(run_pipeline_bg(study_uid, orthanc_id, output_dir))

        # 성공적으로 처리 완료 후 캐시에 표시
        mark_as_processed_in_cache(study_uid)
        return {"message": f"Study {study_uid} processed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[webhook] failed to accept: {e}", exc_info=True)
        return PlainTextResponse(str(e), status_code=500)