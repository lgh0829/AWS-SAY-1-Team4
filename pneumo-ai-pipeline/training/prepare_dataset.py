import os
import argparse
from common.pneumo_utils.segmentation import LungSegmenter
from common.pneumo_utils.preprocessing import ImagePreprocessor
from common.cloud_utils.s3_handler import S3Handler

def prepare_training_data(input_dir, output_dir, apply_segmentation=True, 
                          upload_to_s3=False, bucket_name=None, s3_prefix=None):
    """학습 데이터셋 전처리 및 준비"""
    
    # 1. 이미지 전처리기 생성
    preprocessor = ImagePreprocessor()
    
    # 2. 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    
    # 3. 이미지 전처리 수행
    print(f"전처리 시작: {len(image_paths)}개 이미지")
    processed_dir = os.path.join(output_dir, "preprocessed")
    preprocessor.batch_process(image_paths, processed_dir)
    
    # 4. 폐 분할 적용 (선택 사항)
    if apply_segmentation:
        segmenter = LungSegmenter()
        segmented_dir = os.path.join(output_dir, "segmented")
        
        # 전처리된 이미지들에 대해 분할 적용
        processed_paths = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)]
        segmenter.process_batch(processed_paths, segmented_dir)
        
        # 최종 결과 디렉토리 설정
        final_dir = segmented_dir
    else:
        final_dir = processed_dir
    
    # 5. S3 업로드 (선택 사항)
    if upload_to_s3 and bucket_name and s3_prefix:
        s3 = S3Handler(bucket_name)
        s3.upload_directory(final_dir, s3_prefix)
        print(f"S3 업로드 완료: s3://{bucket_name}/{s3_prefix}")
    
    print(f"데이터셋 준비 완료: {final_dir}")
    return final_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 데이터셋 준비")
    parser.add_argument("--input_dir", required=True, help="원본 이미지 디렉토리")
    parser.add_argument("--output_dir", required=True, help="결과 저장 디렉토리")
    parser.add_argument("--no-segmentation", action="store_false", dest="apply_segmentation",
                       help="폐 분할을 건너뛰기")
    parser.add_argument("--upload-s3", action="store_true", dest="upload_to_s3",
                       help="결과를 S3에 업로드")
    parser.add_argument("--bucket", help="S3 버킷 이름")
    parser.add_argument("--prefix", help="S3 프리픽스")
    
    args = parser.parse_args()
    
    prepare_training_data(
        args.input_dir, 
        args.output_dir,
        args.apply_segmentation,
        args.upload_to_s3,
        args.bucket,
        args.prefix
    )