import boto3
import os
import tempfile
from pathlib import Path

class S3Handler:
    """AWS S3 상호작용을 위한 클래스"""
    
    def __init__(self, bucket_name=None):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
    
    def set_bucket(self, bucket_name):
        """S3 버킷 설정"""
        self.bucket_name = bucket_name
    
    def download_file(self, s3_key, local_path=None):
        """S3에서 파일 다운로드"""
        if not self.bucket_name:
            raise ValueError("S3 버킷이 설정되지 않았습니다.")
            
        if not local_path:
            # 임시 파일 생성
            suffix = Path(s3_key).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                local_path = temp.name
        
        # 파일 다운로드
        self.s3.download_file(self.bucket_name, s3_key, local_path)
        return local_path
    
    def upload_file(self, local_path, s3_key):
        """S3에 파일 업로드"""
        if not self.bucket_name:
            raise ValueError("S3 버킷이 설정되지 않았습니다.")
        
        # 파일 업로드
        self.s3.upload_file(local_path, self.bucket_name, s3_key)
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def download_directory(self, s3_prefix, local_dir):
        """S3 디렉토리의 모든 파일 다운로드"""
        if not self.bucket_name:
            raise ValueError("S3 버킷이 설정되지 않았습니다.")
        
        # 대상 디렉토리 생성
        os.makedirs(local_dir, exist_ok=True)
        
        # 디렉토리 내의 모든 파일 나열
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
        
        downloaded_files = []
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # 폴더 자체인 경우 스킵
                if key.endswith('/'):
                    continue
                    
                # 로컬 경로 생성
                relative_path = key[len(s3_prefix):].lstrip('/')
                local_path = os.path.join(local_dir, relative_path)
                
                # 중간 디렉토리 생성
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # 파일 다운로드
                self.s3.download_file(self.bucket_name, key, local_path)
                downloaded_files.append(local_path)
        
        return downloaded_files
    
    def upload_directory(self, local_dir, s3_prefix):
        """로컬 디렉토리의 모든 파일을 S3에 업로드"""
        if not self.bucket_name:
            raise ValueError("S3 버킷이 설정되지 않았습니다.")
        
        uploaded_keys = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # S3 키 계산
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')
                
                # 업로드
                self.s3.upload_file(local_path, self.bucket_name, s3_key)
                uploaded_keys.append(s3_key)
                
        return uploaded_keys