import pydicom
from PIL import Image
import numpy as np
from pathlib import Path

def convert_dcm2jpg(dcm_path: Path, jpeg_path: Path):
    """
    DICOM(.dcm) 파일을 JPEG(.jpg) 이미지로 변환합니다.
    """
    ds = pydicom.dcmread(str(dcm_path))
    pixel_array = ds.pixel_array

    # 정규화 (0~255 범위) 및 uint8 타입 변환
    pixel_array = pixel_array.astype(float)
    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
    pixel_array = np.uint8(pixel_array)

    img = Image.fromarray(pixel_array)
    img.save(jpeg_path)
