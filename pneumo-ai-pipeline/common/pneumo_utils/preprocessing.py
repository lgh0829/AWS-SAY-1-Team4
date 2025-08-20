from PIL import Image, ImageFilter
import cv2
import numpy as np

class ImagePreprocessor:
    """이미지 전처리를 위한 클래스"""
    
    def __init__(self, preprocessing_config=None):
        """
        preprocessing_config: 전처리 설정 딕셔너리
            {
                'steps': {'convert_grayscale': True, 'apply_clahe': True, ...},
                'params': {'clahe': {...}, 'gaussian_blur': {...}, ...}
            }
        """
        # 기본 설정
        self.steps = {
            'convert_grayscale': True,
            'apply_clahe': True,
            'apply_gaussian_blur': True,
            'apply_min_max_stretch': True,
            'apply_sharpening': False,
            'convert_to_rgb': False
        }
        
        self.params = {
            'clahe': {
                'clip_limit': 2.0,
                'grid_size': (8, 8)
            },
            'gaussian_blur': {
                'radius': 1.0
            },
            'min_max_stretch': {
                'lower_percentile': 1,
                'upper_percentile': 99
            },
            'sharpening': {
                'radius': 2,
                'percent': 150,
                'threshold': 3
            }
        }
        
        # 설정 업데이트
        if preprocessing_config:
            if 'steps' in preprocessing_config:
                self.steps.update(preprocessing_config['steps'])
            if 'params' in preprocessing_config:
                if 'clahe' in preprocessing_config['params']:
                    self.params['clahe'].update(preprocessing_config['params']['clahe'])
                if 'gaussian_blur' in preprocessing_config['params']:
                    self.params['gaussian_blur'].update(preprocessing_config['params']['gaussian_blur'])
                if 'min_max_stretch' in preprocessing_config['params']:
                    self.params['min_max_stretch'].update(preprocessing_config['params']['min_max_stretch'])
                if 'sharpening' in preprocessing_config['params']:
                    self.params['sharpening'].update(preprocessing_config['params']['sharpening'])
        
        # CLAHE 초기화
        if self.steps['apply_clahe']:
            clip_limit = self.params['clahe']['clip_limit']
            grid_size = tuple(self.params['clahe']['grid_size']) if isinstance(self.params['clahe']['grid_size'], list) else self.params['clahe']['grid_size']
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
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
        radius = self.params['gaussian_blur']['radius']
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def apply_min_max_stretch(self, image):
        """Min-Max Stretching 적용 (히스토그램 스트레칭)"""
        np_img = np.array(image)
        
        # 양끝단 1%씩 잘라내서 극단값 처리
        lower = self.params['min_max_stretch']['lower_percentile']
        upper = self.params['min_max_stretch']['upper_percentile']
        min_val, max_val = np.percentile(np_img, [lower, upper])
        
        # 히스토그램 스트레칭 적용
        stretched = (np_img - min_val) / (max_val - min_val + 1e-5) * 255.0
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
        
        return Image.fromarray(stretched)
    
    def apply_sharpening(self, image):
        """언샤프 마스킹으로 샤프닝 적용"""
        radius = self.params['sharpening']['radius']
        percent = self.params['sharpening']['percent']
        threshold = self.params['sharpening']['threshold']
        return image.filter(ImageFilter.UnsharpMask(
            radius=radius, 
            percent=percent, 
            threshold=threshold
        ))
    
    def convert_to_rgb(self, grayscale_image):
        """그레이스케일 이미지를 RGB로 변환"""
        return Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
    
    def process_image(self, image_path):
        """전체 전처리 파이프라인 실행"""
        # 처리할 이미지 로드
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        # 그레이스케일 변환
        if self.steps['convert_grayscale']:
            if isinstance(image_path, str):
                img = self.convert_to_grayscale(image_path)
            else:
                img = img.convert("L")
        
        # CLAHE 적용
        if self.steps['apply_clahe']:
            img = self.apply_clahe(img)
        
        # 가우시안 블러 적용
        if self.steps['apply_gaussian_blur']:
            img = self.apply_gaussian_blur(img)
        
        # Min-Max Stretching 적용
        if self.steps['apply_min_max_stretch']:
            img = self.apply_min_max_stretch(img)
        
        # 샤프닝 적용
        if self.steps['apply_sharpening']:
            img = self.apply_sharpening(img)
        
        # RGB 변환
        if self.steps['convert_to_rgb']:
            img = self.convert_to_rgb(img)
        
        return img
    
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
                print(f"이미지 처리 오류 {img_path}: {e}")
                
        return results

    def preprocess_image(self, image_path, output_path=None):
        """단일 이미지 전처리 및 저장

        Args:
            image_path (str): 입력 이미지 경로
            output_path (str, optional): 출력 이미지 저장 경로

        Returns:
            numpy.ndarray: 전처리된 이미지
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # 그레이스케일 변환
            if self.steps.get('convert_grayscale', False):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 3채널로 변환

            # CLAHE 적용
            if self.steps.get('apply_clahe', False):
                clahe_params = self.params.get('clahe', {})
                clip_limit = clahe_params.get('clip_limit', 2.0)
                grid_size = tuple(clahe_params.get('grid_size', (8, 8)))
                
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                cl = clahe.apply(l)
                lab = cv2.merge((cl, a, b))
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 가우시안 블러 적용
            if self.steps.get('apply_gaussian_blur', False):
                blur_params = self.params.get('gaussian_blur', {})
                radius = int(blur_params.get('radius', 1))
                image = cv2.GaussianBlur(image, (2*radius+1, 2*radius+1), 0)

            # Min-Max Stretching
            if self.steps.get('apply_min_max_stretch', False):
                stretch_params = self.params.get('min_max_stretch', {})
                p_low = stretch_params.get('lower_percentile', 1)
                p_high = stretch_params.get('upper_percentile', 99)
                
                for i in range(3):  # BGR 각 채널에 대해 처리
                    channel = image[:,:,i]
                    low = np.percentile(channel, p_low)
                    high = np.percentile(channel, p_high)
                    image[:,:,i] = np.clip((channel - low) * 255.0 / (high - low), 0, 255)

            # Sharpening
            if self.steps.get('apply_sharpening', False):
                sharp_params = self.params.get('sharpening', {})
                radius = sharp_params.get('radius', 2)
                percent = sharp_params.get('percent', 150)
                threshold = sharp_params.get('threshold', 3)
                
                blurred = cv2.GaussianBlur(image, (2*radius+1, 2*radius+1), 0)
                sharpened = cv2.addWeighted(image, percent/100, blurred, 1-percent/100, 0)
                lowContrastMask = abs(image - blurred) < threshold
                image = np.where(lowContrastMask, image, sharpened)

            # 결과 저장
            if output_path:
                cv2.imwrite(output_path, image)

            return image

        except Exception as e:
            print(f"이미지 전처리 중 오류 발생 ({image_path}): {str(e)}")
            raise