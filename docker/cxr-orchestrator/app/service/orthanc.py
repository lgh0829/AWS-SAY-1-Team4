import httpx
from pathlib import Path
from app.core.config import settings
import os
# import requests

ORTHANC_AUTH = (settings.ORTHANC_USERNAME, settings.ORTHANC_PASSWORD)


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