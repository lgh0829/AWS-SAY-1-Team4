import cv2
import numpy as np
import torch
from inference import inference
from lungs_segmentation.pre_trained_models import create_model
import os

class LungSegmenter:
    def __init__(self, model_type='resnet34', device=None):
        # 디바이스 자동 선택
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = create_model(model_type).to(device=self.device)
        self.model.eval()
    
    def segment_image(self, image_path):
        """단일 이미지에서 폐 영역 분할"""
        _, mask = inference(self.model, image_path)
        
        # 좌우 폐 마스크 결합
        combined_mask = (mask[0] + mask[1]) > 0
        combined_mask = (combined_mask * 255).astype(np.uint8)
        
        # 원본 이미지 로드
        original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_img = cv2.resize(original_img, (combined_mask.shape[1], combined_mask.shape[0]))
        
        # 마스크 적용
        binary_mask_3ch = np.stack([combined_mask]*3, axis=-1)
        binary_mask_3ch = binary_mask_3ch // 255
        masked_img = original_img * binary_mask_3ch
        
        return masked_img, combined_mask, original_img
    
    def process_batch(self, image_paths, output_dir=None):
        """다중 이미지 처리"""
        results = []
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            try:
                masked_img, mask, original = self.segment_image(img_path)
                results.append({
                    'filename': filename,
                    'masked_image': masked_img,
                    'mask': mask,
                    'original': original
                })
                
                # 결과 저장 (선택사항)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"mask_{os.path.splitext(filename)[0]}.png")
                    cv2.imwrite(output_path, masked_img)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        return results