import httpx
from pathlib import Path
from app.core.config import settings
import os
import logging

ORTHANC_AUTH = (settings.ORTHANC_USERNAME, settings.ORTHANC_PASSWORD)
ORTHANC_URL = settings.ORTHANC_URL


def _orthanc_url(path: str) -> str:
    return f"{settings.ORTHANC_URL.rstrip('/')}/{path.lstrip('/')}"

async def get_instance_info(study_uid: str) -> str | None:
    """
    주어진 StudyInstanceUID에서 첫 번째 InstanceUID를 반환.
    """
    study_url = _orthanc_url(f"/studies/{study_uid}")
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        resp = await client.get(study_url)
        if resp.status_code != 200:
            return None
        study_info = resp.json()

        for series_id in study_info.get("Series", []):
            series_url = _orthanc_url(f"/series/{series_id}")
            series_resp = await client.get(series_url)
            series_resp.raise_for_status()
            series_info = series_resp.json()

            for instance_id in series_info.get("Instances", []):
                return instance_id
    return None


async def download_dicom(instance_uid: str, output_dir: Path) -> Path:
    """
    특정 InstanceUID의 DICOM 원본을 다운로드
    """
    url = _orthanc_url(f"/instances/{instance_uid}/file")
    output_path = output_dir / f"{instance_uid}.dcm"

    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
    
    return output_path

async def download_jpeg(instance_uid: str, output_dir: Path) -> Path:
    """
    특정 InstanceUID의 JPEG 변환 이미지를 다운로드 (frame 1 기준)
    """
    url = _orthanc_url(f"/instances/{instance_uid}/frames/1/rendered")
    output_path = output_dir / f"{instance_uid}.jpg"

    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
    
    return output_path

async def upload_dicom(dicom_path: Path) -> str:
    """DICOM 파일을 Orthanc 서버에 업로드"""
    url = _orthanc_url("/instances")
    
    with open(dicom_path, 'rb') as f:
        dicom_data = f.read()
    
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        try:
            resp = await client.post(url, content=dicom_data)
            logging.info(f"DICOM upload response: status={resp.status_code}")
            
            # 응답 확인
            if resp.status_code == 200:
                if resp.text and resp.text.strip():
                    try:
                        result = resp.json()
                        return result.get('ID')  # Orthanc는 보통 'ID' 키로 결과를 반환
                    except ValueError:
                        logging.warning(f"Non-JSON response on successful upload: {resp.text[:100]}")
                        # 성공으로 처리하고 계속 진행
                        return "unknown-id"
                else:
                    logging.warning("Empty response on successful upload")
                    return "unknown-id"
            else:
                logging.error(f"DICOM upload failed: {resp.status_code}, {resp.text}")
                raise Exception(f"Failed to upload DICOM: {resp.status_code}")
        except Exception as e:
            logging.error(f"Error during DICOM upload: {str(e)}")
            raise

async def get_orthanc_study_id(study_uid: str) -> str | None:
    """
    study_uid (DICOM UID) → Orthanc internal ID 조회
    """
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(auth=ORTHANC_AUTH, timeout=timeout) as client:
        try:
            # Orthanc는 StudyInstanceUID를 쿼리할 수 없기 때문에
            # expand=true로 전체 목록을 불러온 후 필터링
            resp = await client.get(_orthanc_url("/studies?expand=true"))
            resp.raise_for_status()
            studies = resp.json()

            for study in studies:
                if study.get("MainDicomTags", {}).get("StudyInstanceUID") == study_uid:
                    return study.get("ID")
            return None
        except Exception as e:
            logging.error(f"Failed to get Orthanc ID for {study_uid}: {e}")
            return None

async def get_study_modality(study_uid: str) -> str:
    """
    스터디에 포함된 시리즈의 모달리티를 확인합니다.
    """
    study_url = _orthanc_url(f"/studies/{study_uid}")
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        resp = await client.get(study_url)
        if resp.status_code != 200:
            return "UNKNOWN"
            
        study_info = resp.json()
        
        # 모든 시리즈 확인
        for series_id in study_info.get("Series", []):
            series_url = _orthanc_url(f"/series/{series_id}")
            series_resp = await client.get(series_url)
            if series_resp.status_code != 200:
                continue
                
            series_info = series_resp.json()
            modality = series_info.get("MainDicomTags", {}).get("Modality")
            
            # OT 모달리티 발견 시 즉시 반환
            if modality == "OT":
                return "OT"
                
        # 기본값 반환 (첫 번째 시리즈의 모달리티 또는 UNKNOWN)
        if study_info.get("Series"):
            first_series = study_info["Series"][0]
            first_series_url = _orthanc_url(f"/series/{first_series}")
            first_series_resp = await client.get(first_series_url)
            if first_series_resp.status_code == 200:
                return first_series_resp.json().get("MainDicomTags", {}).get("Modality", "UNKNOWN")
                
        return "UNKNOWN"


async def is_study_processed(study_id: str) -> bool:
    """
    Orthanc 내부 메타데이터에서 처리 완료 여부를 확인합니다.
    """
    url = _orthanc_url(f"/studies/{study_id}/metadata/processed")
    
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        try:
            resp = await client.get(url)
            # 디버깅 추가
            logging.debug(f"Metadata check response: {resp.status_code}, text: {resp.text}")
            if resp.status_code == 200 and resp.text in ["true", '"true"']:
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to check if study is processed: {e}")
            return False

async def mark_study_as_processing(study_id: str) -> bool:
    """
    Orthanc 내부 메타데이터에 처리 중 표시를 남깁니다.
    """
    url = _orthanc_url(f"/studies/{study_id}/metadata/processing")
    
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        try:
            # 메타데이터에 "true" 문자열 저장
            resp = await client.put(
                url,
                content='"true"',
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            logging.info(f"Marked study {study_id} as processing")
            return True
        except Exception as e:
            logging.error(f"Failed to mark study as processing: {e}")
            return False

async def mark_study_as_processed(study_id: str) -> bool:
    """
    Orthanc 내부 메타데이터에 처리 완료 표시를 남깁니다.
    """
    url = _orthanc_url(f"/studies/{study_id}/metadata/processed")
    
    # 디버깅 로그 추가
    logging.debug(f"Marking study as processed: URL={url}")
    
    async with httpx.AsyncClient(auth=ORTHANC_AUTH) as client:
        try:
            # 메타데이터에 "true" 문자열 저장
            resp = await client.put(
                url,
                content='"true"',
                headers={"Content-Type": "application/json"}
            )
            
            # 결과 로깅
            logging.debug(f"Mark processed response: {resp.status_code}, body: {resp.text[:100]}")
            
            if resp.status_code == 404:
                # 404 오류 처리: Orthanc는 때때로 ID 형식에 민감할 수 있음
                # 별도 조회 시도
                orthanc_id = await get_orthanc_study_id_by_uid(study_id)
                if orthanc_id and orthanc_id != study_id:
                    logging.warning(f"Study ID mismatch: {study_id} vs {orthanc_id}, retrying with correct ID")
                    return await mark_study_as_processed(orthanc_id)
                    
                logging.error(f"Study not found in Orthanc: {study_id}")
                return False
                
            resp.raise_for_status()
            logging.info(f"Marked study {study_id} as processed")
            return True
        except Exception as e:
            logging.error(f"Failed to mark study as processed: {e}")
            return False


# example usage
import asyncio

async def example_usage():
    study_uid = "1.2.3"  # 실제 존재하는 StudyInstanceUID로 바꿔야 함
    output_dir = Path("app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_id = await get_instance_info(study_uid)
    print(f"Instance for study {study_uid}: {instance_id}")

    if instance_id:
        dcm_path = await download_dicom(instance_id, output_dir)
        print(f"Downloaded DICOM to {dcm_path}")

        jpeg_path = await download_jpeg(instance_id, output_dir)
        print(f"Downloaded JPEG to {jpeg_path}")


if __name__ == "__main__":
    asyncio.run(example_usage())