import os
import sys
from pathlib import Path

# 현재 파일의 상위 디렉토리(프로젝트 루트)로 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from common.pneumo_utils.segmentation import LungSegmenter
from common.pneumo_utils.preprocessing import ImagePreprocessor
from common.cloud_utils.s3_handler import S3Handler
import dotenv
import re

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

def replace_env_vars(value):
    """환경 변수 치환 함수"""
    if isinstance(value, str):
        # ${VAR_NAME} 패턴 찾기
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, value)
        
        # 각 환경 변수 치환
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value:
                value = value.replace(f"${{{var_name}}}", env_value)
        return value
    return value

def process_yaml_dict(yaml_dict):
    """중첩된 딕셔너리에서 환경 변수 치환"""
    result = {}
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            result[key] = process_yaml_dict(value)
        elif isinstance(value, list):
            result[key] = [process_yaml_dict(item) if isinstance(item, dict) else replace_env_vars(item) for item in value]
        else:
            result[key] = replace_env_vars(value)
    return result

def load_config(config_path):
    """설정 파일 로드 및 환경 변수 치환"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 환경 변수 치환 처리
    config = process_yaml_dict(config)
    return config

def prepare_training_data(config_path):
    """학습 데이터셋 전처리 및 준비"""
    # 설정 로드
    config = load_config(config_path)

    # 디렉토리 경로 설정
    input_dir = config['directories']['input_dir']
    segmented_dir = config['directories']['segmented_dir']
    preprocessed_dir = config['directories']['preprocessed_dir']
    
    # S3 설정
    bucket_name = config['s3'].get('bucket_name')
    s3_prefix = config['s3'].get('prefix')
    s3_input_prefix = config['s3'].get('input_prefix', '')
    s3_segmented_prefix = config['s3'].get('segmented_prefix', '')
    s3_preprocessed_prefix = config['s3'].get('preprocessed_prefix', '')

    # 디렉토리 생성
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # 1. S3에서 이미지 파일 다운로드 (클래스별 구조 유지)
    if bucket_name and s3_input_prefix:
        full_s3_prefix = f"{s3_prefix}/{s3_input_prefix}" if s3_prefix else s3_input_prefix
        print(f"S3에서 이미지 다운로드 시작: s3://{bucket_name}/{full_s3_prefix}")
        s3 = S3Handler(bucket_name)
        
        try:
            # S3에서 폴더 목록 조회 (0, 1, 2 등의 클래스 폴더)
            paginator = s3.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=full_s3_prefix,
                Delimiter='/'
            )
            
            # 클래스별 폴더 구조 확인
            class_prefixes = []
            for page in pages:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        class_prefixes.append(prefix['Prefix'])
            
            # 클래스별 폴더가 없는 경우 전체 다운로드
            if not class_prefixes:
                downloaded_files = s3.download_directory(full_s3_prefix, input_dir)
                print(f"S3에서 다운로드 완료: {len(downloaded_files)}개 파일")
            else:
                # 클래스별 폴더별로 다운로드
                total_downloaded = 0
                for class_prefix in class_prefixes:
                    class_name = os.path.basename(class_prefix.rstrip('/'))
                    class_dir = os.path.join(input_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    downloaded_files = s3.download_directory(class_prefix, class_dir)
                    total_downloaded += len(downloaded_files)
                    print(f"클래스 {class_name} 다운로드 완료: {len(downloaded_files)}개 파일")
                
                print(f"S3에서 다운로드 완료: 총 {total_downloaded}개 파일")
        except Exception as e:
            print(f"S3 다운로드 중 오류 발생: {str(e)}")
    
    # 2. 이미지 파일 목록 수집 (클래스별 구조 포함)
    image_paths = []
    # 입력 디렉토리 내 모든 이미지 파일 수집 (하위 디렉토리 포함)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    print(f"처리할 이미지 총 개수: {len(image_paths)}개")

    # 3. 폐 분할 적용 (클래스별 구조 유지)
    segmenter = LungSegmenter()
    total_images = len(image_paths)

    print(f"폐 분할 시작: 총 {total_images}개 이미지")

    # 클래스별 구조를 유지하면서 이미지 분할
    for i, img_path in enumerate(image_paths, 1):
        try:
            # 진행 상황 출력
            if i % 10 == 0 or i == total_images:
                print(f"폐 분할 진행 중: {i}/{total_images} ({i/total_images*100:.1f}%)")
            
            # 원본 경로에서 상대 경로 계산
            rel_path = os.path.relpath(img_path, input_dir)
            # 클래스 구조를 유지하기 위한 대상 경로 계산
            output_dir = os.path.dirname(os.path.join(segmented_dir, rel_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # 출력 파일 경로
            output_path = os.path.join(segmented_dir, rel_path)
            
            # 개별 이미지 분할
            segmenter.segment_image(img_path, output_path)
        except Exception as e:
            print(f"이미지 분할 중 오류 발생 ({img_path}): {str(e)}")

    print(f"폐 분할 완료")
    
    # 4. 분할된 폐 이미지를 S3에 업로드 (클래스별 구조 유지)
    if bucket_name and s3_prefix:
        s3 = S3Handler(bucket_name)
        full_segmented_prefix = f"{s3_prefix}/{s3_segmented_prefix}" if s3_prefix else s3_segmented_prefix
        print(f"분할된 이미지 S3 업로드 시작: s3://{bucket_name}/{full_segmented_prefix}")
        
        # 클래스 구조를 유지하면서 업로드
        uploaded_files = s3.upload_directory(segmented_dir, full_segmented_prefix)
        print(f"분할된 폐 이미지 S3 업로드 완료: {len(uploaded_files)}개 파일")
    
    # 5. 이미지 전처리기 생성 및 분할된 이미지에 적용 (설정 적용)
    preprocessor = ImagePreprocessor(config.get('preprocessing'))
    
    # 6. 분할된 이미지들에 대해 전처리 적용 (클래스별 구조 유지)
    segmented_paths = []
    for root, _, files in os.walk(segmented_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                segmented_paths.append(os.path.join(root, file))

    total_images = len(segmented_paths)
    print(f"전처리 시작: 총 {total_images}개 이미지")

    # 클래스별 구조를 유지하면서 이미지 전처리
    for i, img_path in enumerate(segmented_paths, 1):
        try:
            # 진행 상황 출력
            if i % 10 == 0 or i == total_images:
                print(f"전처리 진행 중: {i}/{total_images} ({i/total_images*100:.1f}%)")
            
            # 원본 경로에서 상대 경로 계산
            rel_path = os.path.relpath(img_path, segmented_dir)
            # 클래스 구조를 유지하기 위한 대상 경로 계산
            output_dir = os.path.dirname(os.path.join(preprocessed_dir, rel_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # 출력 파일 경로
            output_path = os.path.join(preprocessed_dir, rel_path)
            
            # 개별 이미지 전처리
            preprocessor.preprocess_image(img_path, output_path)
        except Exception as e:
            print(f"이미지 전처리 중 오류 발생 ({img_path}): {str(e)}")

    print(f"전처리 완료")
    
    # 7. 전처리된 이미지를 S3에 업로드 (클래스별 구조 유지)
    if bucket_name and s3_prefix:
        s3 = S3Handler(bucket_name)
        full_preprocessed_prefix = f"{s3_prefix}/{s3_preprocessed_prefix}" if s3_prefix else s3_preprocessed_prefix
        print(f"전처리된 이미지 S3 업로드 시작: s3://{bucket_name}/{full_preprocessed_prefix}")
        
        # 클래스 구조를 유지하면서 업로드
        uploaded_files = s3.upload_directory(preprocessed_dir, full_preprocessed_prefix)
        print(f"전처리된 이미지 S3 업로드 완료: {len(uploaded_files)}개 파일")
    
    print(f"데이터셋 준비 완료")
    return preprocessed_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 데이터셋 준비")
    parser.add_argument("--config", required=True, help="설정 파일 경로")
    
    args = parser.parse_args()
    
    prepare_training_data(args.config)