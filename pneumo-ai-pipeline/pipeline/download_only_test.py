from pathlib import Path
import sys
import dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.cloud_utils.s3_handler import S3Handler

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

s3_prefix = 'cxr-pneumonia-3/preprocessed/test'
local_dir = Path(__file__).parent / 'data'

s3_handle = S3Handler(bucket_name=os.getenv('S3_BUCKET_NAME'))
s3_handle.download_directory(f'{s3_prefix}/0', f'{local_dir}/0')
s3_handle.download_directory(f'{s3_prefix}/1', f'{local_dir}/1')
s3_handle.download_directory(f'{s3_prefix}/2', f'{local_dir}/2')
