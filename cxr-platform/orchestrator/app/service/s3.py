import boto3
from pathlib import Path
from datetime import datetime
from app.core.config import settings

s3 = boto3.client("s3", region_name=settings.AWS_REGION)


def upload_file(local_path: Path, study_uid: str, subfolder: str, filename: str = None) -> str:
    """
    S3에 파일 업로드 (Prefix 포함)

    Args:
        local_path: 로컬 파일 경로
        study_uid: StudyInstanceUID
        subfolder: input/original, segmentation/masks 등
        filename: 저장할 파일명 (기본은 local_path의 이름)
    
    Returns:
        s3_uri: s3://bucket/prefix/path/to/file
    """
    if filename is None:
        filename = Path(local_path).name

    # 경로 구성: prod/studies/{StudyUID}/{subfolder}/{filename}
    key = f"{settings.S3_PREFIX.rstrip('/')}/studies/{study_uid}/{subfolder}/{filename}"

    s3.upload_file(str(local_path), settings.S3_BUCKET, key)
    return f"s3://{settings.S3_BUCKET}/{key}"


def upload_pipeline_log(study_uid: str, content: dict):
    """
    metadata/log_{timestamp}.json 저장
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    key = f"{settings.S3_PREFIX.rstrip('/')}/studies/{study_uid}/metadata/log_{timestamp}.json"

    s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=str(content).encode("utf-8"),
        ContentType="application/json"
    )
    return f"s3://{settings.S3_BUCKET}/{key}"

def download_file(s3_uri: str, local_path: Path):
    """
    S3에서 파일 다운로드

    Args:
        s3_uri: s3://bucket/prefix/path/to/file
        local_path: 로컬 저장 경로
    """
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    
    # local_path가 디렉토리일 경우, S3 key의 basename을 사용하여 저장 경로 설정
    if local_path.is_dir():
        filename = Path(key).name
        local_path = local_path / filename
    
    s3.download_file(bucket, key, str(local_path))

    return local_path

# example usage:
if __name__ == "__main__":
    # 로컬 파일 경로 예시
    local_image_path = Path("local/image.jpg")
    local_dcm_path = Path("local/image.dcm")
    local_mask_path = Path("mask.png")
    local_result_path = Path("result.json")

    # StudyInstanceUID 예시
    study_uid = "1.2.3"

    # 파일 업로드 예시
    jpeg_uri = upload_file(local_image_path, study_uid, "input/original", "abc.jpg")
    dcm_uri = upload_file(local_dcm_path, study_uid, "input/dicom", "abc.dcm")
    mask_uri = upload_file(local_mask_path, study_uid, "segmentation/masks", "abc_mask.png")
    result_uri = upload_file(local_result_path, study_uid, "classification", "result.json")

    # 로그 업로드 예시
    upload_pipeline_log(study_uid, {"status": "success", "timestamp": datetime.utcnow().isoformat()})

    print(f"JPEG URI: {jpeg_uri}")
    print(f"DICOM URI: {dcm_uri}")
    print(f"Mask URI: {mask_uri}")
    print(f"Result URI: {result_uri}")