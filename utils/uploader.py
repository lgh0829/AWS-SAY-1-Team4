import pandas as pd
import boto3
import botocore.session
import certifi
import os
from tqdm import tqdm

def upload_images(
    csv_path: str,
    local_image_dir: str,
    s3_uris: dict,  # {'train': 's3://bucket/train', ...}
    set_name: str   # e.g., 'nih'
):
    session = botocore.session.get_session()
    session.set_config_variable('ca_bundle', certifi.where())
    s3 = boto3.client('s3', config=botocore.config.Config(), verify=certifi.where())
    df = pd.read_csv(csv_path)

    # 필수 컬럼 확인
    required_cols = {'file_name', 'split', 'set_name'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 파일에는 {required_cols} 컬럼이 포함되어 있어야 합니다.")

    # set_name이 일치하는 항목만 필터링
    df = df[df['set_name'] == set_name]

    if df.empty:
        print(f"⚠️ '{set_name}'에 해당하는 레코드가 없습니다.")
        return

    # 하위 디렉토리 전체를 인덱싱 (파일명 기준)
    local_file_map = {}
    for root, _, files in os.walk(local_image_dir):
        for fname in files:
            local_file_map[fname] = os.path.join(root, fname)

    # 업로드 루프
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Uploading images for set '{set_name}'"):
        file_name = row['file_name']
        split = row['split']

        local_path = local_file_map.get(file_name)
        if not local_path or not os.path.isfile(local_path):
            print(f"❌ 파일 없음: {file_name}")
            continue

        if split not in s3_uris:
            print(f"❌ 잘못된 split 값: {split}")
            continue

        s3_uri = s3_uris[split]
        bucket, *key_parts = s3_uri.replace("s3://", "").split("/", 1)
        s3_prefix = key_parts[0] if key_parts else ""

        # 안전하게 S3 key 구성
        s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"

        # ✅ 진단용 출력
        print(f"[DEBUG] local_path: {local_path}")
        print(f"[DEBUG] bucket: {bucket}")
        print(f"[DEBUG] s3_key: {s3_key}")

        try:
            s3.upload_file(local_path, bucket, s3_key)
            print(f"✅ 업로드 완료: {file_name} → s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"🚨 업로드 실패: {file_name} → Failed to upload {local_path} to {bucket}/{s3_key}: {e}")