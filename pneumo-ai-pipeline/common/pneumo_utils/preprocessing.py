import cv2
import numpy as np
from PIL import Image, ImageFilter

class ImagePreprocessor:
    """이미지 전처리를 위한 클래스"""
    
    def __init__(self, clahe_clip=2.0, clahe_grid=(8, 8), 
                 blur_radius=1.0, sharpen_radius=2, 
                 sharpen_percent=150, sharpen_threshold=3):
        # CLAHE 파라미터
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        
        # 블러 파라미터
        self.blur_radius = blur_radius
        
        # 샤프닝 파라미터
        self.sharpen_radius = sharpen_radius
        self.sharpen_percent = sharpen_percent
        self.sharpen_threshold = sharpen_threshold
    
    def convert_to_grayscale(self, image_path):
        """이미지를 그레이스케일로 변환"""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("L")
        else:
            img = image_path.convert("L")
        return img
    
    def apply_clahe(self, image):
        """CLAHE 적용"""
        np_img = np.array(image)
        equalized = self.clahe.apply(np_img)
        return Image.fromarray(equalized)
    
    def apply_gaussian_blur(self, image):
        """가우시안 블러 적용"""
        return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
    
    def apply_sharpening(self, image):
        """언샤프 마스킹으로 샤프닝 적용"""
        return image.filter(ImageFilter.UnsharpMask(
            radius=self.sharpen_radius, 
            percent=self.sharpen_percent, 
            threshold=self.sharpen_threshold
        ))
    
    def convert_to_rgb(self, grayscale_image):
        """그레이스케일 이미지를 RGB로 변환"""
        return Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
    
    def process_image(self, image_path):
        """전체 전처리 파이프라인 실행"""
        # 그레이스케일 변환
        gray_img = self.convert_to_grayscale(image_path)
        
        # CLAHE 적용
        clahe_img = self.apply_clahe(gray_img)
        
        # 가우시안 블러 적용
        blurred_img = self.apply_gaussian_blur(clahe_img)
        
        # 샤프닝 적용
        sharpened_img = self.apply_sharpening(blurred_img)
        
        # RGB 변환
        rgb_img = self.convert_to_rgb(sharpened_img)
        
        return rgb_img
    
    def batch_process(self, image_paths, output_dir=None):
        """다중 이미지 처리"""
        results = []
        for img_path in image_paths:
            try:
                processed_img = self.process_image(img_path)
                results.append(processed_img)
                
                # 결과 저장 (선택사항)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_dir, filename)
                    processed_img.save(output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return results