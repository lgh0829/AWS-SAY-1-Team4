import pydicom
from PIL import Image
import numpy as np
from pathlib import Path
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.dataset import FileDataset, FileMetaDataset

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

    return jpeg_path

def convert_jpg2dcm(jpeg_path: Path, dcm_path: Path, study_uid: str = None, original_dcm_path: Path = None):
    """
    JPEG(.jpg) 이미지를 DICOM(.dcm) 파일로 변환합니다.
    study_uid가 제공되면 기존 스터디에 시리즈를 추가합니다.
    original_dcm_path가 제공되면 원본 DICOM의 환자 정보를 복사합니다.
    """
    # 컬러 이미지로 유지 (흑백 변환 제거)
    img = Image.open(jpeg_path)
    pixel_array = np.array(img)
    
    is_color = len(pixel_array.shape) > 2 and pixel_array.shape[2] >= 3
    
    # 원본 DICOM에서 환자 정보를 복사할 경우
    if original_dcm_path and Path(original_dcm_path).exists():
        source_ds = pydicom.dcmread(str(original_dcm_path))
        
        # 파일 메타 정보 생성
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        # DICOM 데이터셋 생성
        ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # 환자 정보 복사
        for tag in ['PatientID', 'PatientName', 'PatientBirthDate', 'PatientSex']:
            if hasattr(source_ds, tag):
                setattr(ds, tag, getattr(source_ds, tag))
                
        # 스터디 정보 복사 (StudyDescription, StudyDate, StudyTime 등)
        for tag in ['StudyDescription', 'StudyDate', 'StudyTime', 'AccessionNumber']:
            if hasattr(source_ds, tag):
                setattr(ds, tag, getattr(source_ds, tag))
    else:
        # 새 DICOM 생성
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        
        # DICOM 데이터셋 생성
        ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # 픽셀 데이터 설정
    if is_color:
        rgb = pixel_array[..., :3]  # RGB 컴포넌트만 사용
        ds.PixelData = rgb.tobytes()
        ds.Rows, ds.Columns = rgb.shape[:2]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0  # 인터리브 포맷
    else:
        ds.PixelData = pixel_array.tobytes()
        ds.Rows, ds.Columns = pixel_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
    
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    
    # 필수 DICOM 태그 추가
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # 스터디 정보 설정
    if study_uid:
        # 기존 스터디에 시리즈 추가
        ds.StudyInstanceUID = study_uid
    else:
        # 새 스터디 생성
        ds.StudyInstanceUID = generate_uid()
        
    # 새 시리즈 ID 생성
    ds.SeriesInstanceUID = generate_uid()
    
    # 모달리티 설정 (OT: 기타)
    ds.Modality = 'OT'  # 오버레이/히트맵은 OT(Other) 모달리티로 설정
    
    # 설명 추가
    ds.SeriesDescription = "AI Analysis Result - Heatmap"
    
    # 사용자 정의 태그 추가 - AI 처리 여부 표시
    ds.add_new([0x0071, 0x0001], "CS", "PROCESSED_BY_AI")
    
    # 날짜/시간 정보 추가
    import datetime
    now = datetime.datetime.now()
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S')
    
    # 인코딩 방식 설정
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # DICOM 파일 저장
    ds.save_as(dcm_path)

    return dcm_path

def modify_dicom_for_existing_study(original_path: Path, output_path: Path, study_uid: str) -> None:
    ds = pydicom.dcmread(original_path)

    # 기존 StudyInstanceUID 재사용
    ds.StudyInstanceUID = study_uid

    # SeriesInstanceUID 새로 생성하거나 기존 값 복사
    ds.SeriesInstanceUID = generate_uid()

    # InstanceUID는 반드시 유일하게
    ds.SOPInstanceUID = generate_uid()

    # Optionally 업데이트 시간 정보도 갱신
    ds.InstanceCreationDate = ''
    ds.InstanceCreationTime = ''

    ds.Modality = 'OT'  # Original Type, e.g., Overlay

    # 파일 메타 정보 설정
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID if hasattr(ds, 'SOPClassUID') else '1.2.840.10008.5.1.4.1.1.7'
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds.file_meta = file_meta
    
    # 인코딩 방식 설정
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # 파일 저장
    ds.save_as(output_path)

    return output_path

