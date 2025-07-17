import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches

class VanillaGradientExplainer:
    """
    흉부 X-ray 폐렴 진단을 위한 Vanilla Gradient 기반 XAI 구현
    (NIH 제공 bounding box 지원)
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 전처리 transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 입력 크기
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    
    def load_bounding_boxes(self, csv_path, image_filename):
        import pandas as pd

        # CSV 읽기
        df = pd.read_csv(csv_path)

        # ────────── 필터 1: 폐렴 병변(Target==1)만
        if 'Target' in df.columns:
            df = df[df['Target'] == 1]

        # ────────── 필터 2: 좌표컬럼에 NaN 있는 행 제거
        df = df.dropna(subset=['x','y','width','height'])

        # ────────── 필터 3: patientId(혹은 filename)로 해당 영상만 골라내기
        key = 'patientId' if 'patientId' in df.columns else 'filename'
        df = df[df[key] == image_filename]

       # 4) 'Target' 컬럼을 'class' 로 복사 (정수형)
        df = df.rename(columns={'Target':'class'})
        df['class'] = df['class'].astype(int)

        # 디버그: 몇 개 행이 남았는지 출력
        print(f"[DEBUG] '{image_filename}' 에 대해 {len(df)}개 bbox 로드됨")

        # 기록할 박스 리스트로 변환
        boxes = df[['x','y','width','height','class']].to_dict('records')
        return boxes
    
    def normalize_bbox_coordinates(self, bboxes, original_size, target_size=(224, 224)):
        """
        Bounding box 좌표를 모델 입력 크기에 맞게 정규화
        
        Args:
            bboxes: 원본 bounding box 리스트
            original_size: 원본 이미지 크기 (width, height)
            target_size: 목표 이미지 크기 (width, height)
            
        Returns:
            normalized_bboxes: 정규화된 bounding box 리스트
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        normalized_bboxes = []
        for bbox in bboxes:
            norm_bbox = bbox.copy()
            norm_bbox['x'] = bbox['x'] * scale_x
            norm_bbox['y'] = bbox['y'] * scale_y
            norm_bbox['width'] = bbox['width'] * scale_x
            norm_bbox['height'] = bbox['height'] * scale_y
            normalized_bboxes.append(norm_bbox)
            
        return normalized_bboxes
    
    def compute_vanilla_gradients(self, image, target_class=None):
        """
        Vanilla Gradient 계산
        
        Args:
            image: 입력 X-ray 이미지 (PIL Image 또는 numpy array)
            target_class: 목표 클래스 (None이면 예측된 클래스 사용)
        
        Returns:
            gradients: 각 픽셀의 gradient 값 (2D array)
            prediction: 모델 예측 결과
            confidence: 예측 신뢰도
        """
        
        # 이미지 전처리
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # 예측 결과 계산
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
        
        # 목표 클래스 설정
        if target_class is None:
            target_class = prediction
        
        # Backward pass - 목표 클래스에 대한 gradient 계산
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Gradient 추출 및 차원 처리
        gradients = input_tensor.grad.data.squeeze().cpu().numpy()
        
        # 다채널인 경우 (RGB 등) 평균 또는 첫 번째 채널만 사용
        if gradients.ndim == 3:
            gradients = np.mean(gradients, axis=0)  # 채널별 평균
        elif gradients.ndim == 1:
            # 1D인 경우 224x224로 reshape
            gradients = gradients.reshape(224, 224)
        
        return gradients, prediction, confidence
    
    def postprocess_gradients(self, gradients, method='absolute'):
        """
        Gradient 후처리
        
        Args:
            gradients: 원본 gradient 값
            method: 후처리 방법 ('absolute', 'positive', 'negative', 'raw')
        
        Returns:
            processed_gradients: 후처리된 gradient 값
        """
        
        if method == 'absolute':
            # 절댓값 - 양수/음수 영향 모두 고려
            processed = np.abs(gradients)
        elif method == 'positive':
            # 양수만 - 예측을 강화하는 픽셀
            processed = np.maximum(gradients, 0)
        elif method == 'negative':
            # 음수만 - 예측을 약화시키는 픽셀
            processed = np.minimum(gradients, 0)
            processed = np.abs(processed)
        else:  # raw
            processed = gradients
        
        # 정규화 (0-1 범위)
        if processed.max() > processed.min():
            processed = (processed - processed.min()) / (processed.max() - processed.min())
        
        return processed
    
    def apply_threshold_filter(self, gradients, threshold_percentile=95):
        """
        임계값 필터링 - 상위 퍼센타일만 표시
        
        Args:
            gradients: 입력 gradient 값
            threshold_percentile: 임계값 퍼센타일 (예: 95 = 상위 5%)
        
        Returns:
            filtered_gradients: 필터링된 gradient 값
        """
        
        threshold = np.percentile(gradients, threshold_percentile)
        filtered = np.where(gradients >= threshold, gradients, 0)
        
        return filtered
    
    def smooth_gradients(self, gradients, sigma=1.0):
        """
        가우시안 블러를 통한 노이즈 제거
        
        Args:
            gradients: 입력 gradient 값
            sigma: 가우시안 블러 강도
        
        Returns:
            smoothed_gradients: 스무딩된 gradient 값
        """
        
        # 가우시안 블러 적용
        smoothed = cv2.GaussianBlur(gradients, (5, 5), sigma)
        
        return smoothed

    def batch_visualize_gradients_with_bbox(self, original_image, heatmap, bboxes, alpha=0.4):

        # 1) 이미지 → numpy, 크기(H,W) 얻기
        if not isinstance(original_image, np.ndarray):
            img_np = np.array(original_image)
        else:
            img_np = original_image
        H, W = img_np.shape[:2]

        # 2) heatmap을 원본 크기로 리사이즈 후 컬러맵 씌우기
        hm_resized = cv2.resize((heatmap * 255).astype(np.uint8), (W, H))
        cmap = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)

        # 그레이스케일 배경이라면 BGR로 변환
        if img_np.ndim == 2:
            bg = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            bg = img_np.copy()

        # 히트맵 오버레이 합성
        overlay = cv2.addWeighted(bg, 1 - alpha, cmap, alpha, 0)

        # 3) 박스 좌표 복원 (normalized → pixel)
        pixel_boxes = []
        for box in bboxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']

            # 0~1 사이면 normalized로 간주
            if 0.0 <= x <= 1.0 and 0.0 <= w <= 1.0:
                x_px = int(x * W)
                y_px = int(y * H)
                w_px = int(w * W)
                h_px = int(h * H)
            else:
                # 이미 픽셀 단위라면 그대로 int 처리
                x_px, y_px, w_px, h_px = map(int, (x, y, w, h))

            pixel_boxes.append((x_px, y_px, w_px, h_px))

        # 4) 시각화
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.axis('off')

        # cyan 색 박스 그리기
        for (x0, y0, ww, hh) in pixel_boxes:
            rect = patches.Rectangle(
                (x0, y0), ww, hh,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)

        # plt.show() # 이 줄을 제거합니다.
        return fig, ax # Figure와 Axes 객체를 반환하도록 변경합니다.


    def visualize_gradients_with_bbox(self, original_image, heatmap, bboxes, alpha=0.4):
        """
        original_image: PIL Image 또는 numpy array (HxW 또는 HxWx3)
        heatmap: 0~1 정규화된 2D numpy 배열
        bboxes: [{'x':…, 'y':…, 'width':…, 'height':…}, …]
                값이 0~1(normalized) 또는 픽셀 단위 둘 다 지원
        alpha: 히트맵 투명도
        """
        import numpy as np, cv2, matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # 1) 이미지 → numpy, 크기(H,W) 얻기
        if not isinstance(original_image, np.ndarray):
            img_np = np.array(original_image)
        else:
            img_np = original_image
        H, W = img_np.shape[:2]

        # 2) heatmap을 원본 크기로 리사이즈 후 컬러맵 씌우기
        hm_resized = cv2.resize((heatmap * 255).astype(np.uint8), (W, H))
        cmap = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)

        # 그레이스케일 배경이라면 BGR로 변환
        if img_np.ndim == 2:
            bg = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            bg = img_np.copy()

        # 히트맵 오버레이 합성
        overlay = cv2.addWeighted(bg, 1 - alpha, cmap, alpha, 0)

        # 3) 박스 좌표 복원 (normalized → pixel)
        pixel_boxes = []
        for box in bboxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']

            # 0~1 사이면 normalized로 간주
            if 0.0 <= x <= 1.0 and 0.0 <= w <= 1.0:
                x_px = int(x * W)
                y_px = int(y * H)
                w_px = int(w * W)
                h_px = int(h * H)
            else:
                # 이미 픽셀 단위라면 그대로 int 처리
                x_px, y_px, w_px, h_px = map(int, (x, y, w, h))

            pixel_boxes.append((x_px, y_px, w_px, h_px))

        # 4) 시각화
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.axis('off')

        # cyan 색 박스 그리기
        for (x0, y0, ww, hh) in pixel_boxes:
            rect = patches.Rectangle(
                (x0, y0), ww, hh,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)

        plt.show()

    def calculate_bbox_gradient_overlap(self, gradients, bboxes, threshold_percentile=95):
        """
        Bounding box와 gradient의 겹침 정도 계산
        
        Args:
            gradients: gradient 값 (2D array)
            bboxes: bounding box 리스트
            threshold_percentile: 고강도 gradient 임계값
            
        Returns:
            overlap_metrics: 겹침 정도 메트릭
        """
        
        # 고강도 gradient 영역 추출
        threshold = np.percentile(gradients, threshold_percentile)
        high_gradient_mask = gradients >= threshold
        
        overlap_metrics = []
        
        for i, bbox in enumerate(bboxes):
            # Bounding box 영역 마스크 생성
            bbox_mask = np.zeros_like(gradients, dtype=bool)
            
            x_start = max(0, int(bbox['x']))
            y_start = max(0, int(bbox['y']))
            x_end = min(gradients.shape[1], int(bbox['x'] + bbox['width']))
            y_end = min(gradients.shape[0], int(bbox['y'] + bbox['height']))
            
            bbox_mask[y_start:y_end, x_start:x_end] = True
            
            # 겹침 계산
            intersection = np.logical_and(high_gradient_mask, bbox_mask)
            union = np.logical_or(high_gradient_mask, bbox_mask)
            
            # 메트릭 계산
            intersection_area = np.sum(intersection)
            bbox_area = np.sum(bbox_mask)
            high_grad_area = np.sum(high_gradient_mask)
            
            iou = intersection_area / np.sum(union) if np.sum(union) > 0 else 0
            precision = intersection_area / high_grad_area if high_grad_area > 0 else 0
            recall = intersection_area / bbox_area if bbox_area > 0 else 0
            
            # Bounding box 내부 평균 gradient 강도
            bbox_gradient_intensity = np.mean(gradients[bbox_mask]) if bbox_area > 0 else 0
            
            overlap_metrics.append({
                'bbox_id': i,
                'bbox_class': bbox['class'],
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'bbox_gradient_intensity': bbox_gradient_intensity,
                'intersection_area': intersection_area,
                'bbox_area': bbox_area
            })
        
        return overlap_metrics
    
    def comprehensive_analysis_with_bbox(self, image, csv_path=None, image_filename=None, target_class=None):
        """
        Bounding box를 포함한 종합적인 Vanilla Gradient 분석
        
        Args:
            image: 입력 X-ray 이미지
            csv_path: CSV 파일 경로 (선택사항)
            image_filename: 이미지 파일명 (선택사항)
            target_class: 목표 클래스
        
        Returns:
            analysis_results: 분석 결과 딕셔너리
        """
        
        # 기본 gradient 계산
        gradients, prediction, confidence = self.compute_vanilla_gradients(image, target_class)
        
        # 다양한 후처리 방법 적용
        grad_absolute = self.postprocess_gradients(gradients, 'absolute')
        grad_positive = self.postprocess_gradients(gradients, 'positive')
        grad_negative = self.postprocess_gradients(gradients, 'negative')
        
        # 임계값 필터링 적용
        grad_filtered = self.apply_threshold_filter(grad_absolute, 95)
        
        # 스무딩 적용
        grad_smoothed = self.smooth_gradients(grad_absolute, sigma=1.0)
        
        # Bounding box 로드
        bboxes = []
        overlap_metrics = []
        
        if csv_path and image_filename:
            bboxes = self.load_bounding_boxes(csv_path, image_filename)
            
            # 이미지 크기 정보 (실제 사용시 적절히 수정)
            if isinstance(image, Image.Image):
                original_size = image.size  # (width, height)
            else:
                original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Bounding box 좌표 정규화
            normalized_bboxes = self.normalize_bbox_coordinates(bboxes, original_size)
            
            # 겹침 정도 계산
            overlap_metrics = self.calculate_bbox_gradient_overlap(grad_absolute, normalized_bboxes)
        
        # 통계 정보 계산
        stats = {
            'mean_gradient': np.mean(grad_absolute),
            'std_gradient': np.std(grad_absolute),
            'max_gradient': np.max(grad_absolute),
            'min_gradient': np.min(grad_absolute),
            'active_pixels_ratio': np.sum(grad_filtered > 0) / grad_filtered.size,
            'num_bboxes': len(bboxes)
        }
        
        analysis_results = {
            'prediction': prediction,
            'confidence': confidence,
            'gradients': {
                'raw': gradients,
                'absolute': grad_absolute,
                'positive': grad_positive,
                'negative': grad_negative,
                'filtered': grad_filtered,
                'smoothed': grad_smoothed
            },
            'bounding_boxes': bboxes,
            'overlap_metrics': overlap_metrics,
            'statistics': stats
        }
        
        return analysis_results
    
    def print_overlap_analysis(self, overlap_metrics):
        """
        겹침 분석 결과 출력
        """
        if not overlap_metrics:
            print("겹침 분석 결과가 없습니다.")
            return
        
        print("=== XAI와 Ground Truth Bounding Box 겹침 분석 ===")
        print(f"{'ID':<3} {'Class':<12} {'IoU':<6} {'Precision':<9} {'Recall':<6} {'Intensity':<9}")
        print("-" * 60)
        
        for metric in overlap_metrics:
            print(f"{metric['bbox_id']:<3} {metric['bbox_class']:<12} "
                  f"{metric['iou']:.3f}  {metric['precision']:.3f}     "
                  f"{metric['recall']:.3f}  {metric['bbox_gradient_intensity']:.3f}")
        
        # 평균 메트릭 계산
        avg_iou = np.mean([m['iou'] for m in overlap_metrics])
        avg_precision = np.mean([m['precision'] for m in overlap_metrics])
        avg_recall = np.mean([m['recall'] for m in overlap_metrics])
        
        print("-" * 60)
        print(f"평균 IoU: {avg_iou:.3f}")
        print(f"평균 Precision: {avg_precision:.3f}")
        print(f"평균 Recall: {avg_recall:.3f}")


# # 사용 예제
# def example_usage_with_bbox():
#     """
#     Bounding box를 포함한 사용 예제
#     """
    
#     # 예제 CNN 모델 (실제로는 학습된 폐렴 분류 모델 사용)
#     class SimpleCNN(nn.Module):
#         def __init__(self, num_classes=2):  # 정상/폐렴
#             super(SimpleCNN, self).__init__()
#             self.features = nn.Sequential(
#                 nn.Conv2d(1, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(64, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(128, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool2d((7, 7))
#             )
#             self.classifier = nn.Sequential(
#                 nn.Flatten(),
#                 nn.Linear(256 * 7 * 7, 512),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(512, num_classes)
#             )
        
#         def forward(self, x):
#             x = self.features(x)
#             x = self.classifier(x)
#             return x
    
#     # 모델 초기화
#     model = SimpleCNN(num_classes=2)
    
#     # XAI 분석기 초기화
#     explainer = VanillaGradientExplainer(model)
    
#     # 실제 사용 예제 (주석 처리된 부분)
#     """
#     # 이미지 로드
#     image = Image.open('path/to/chest_xray.jpg').convert('L')
    
#     # CSV 파일과 함께 분석
#     analysis_results = explainer.comprehensive_analysis_with_bbox(
#         image, 
#         csv_path='path/to/bounding_boxes.csv',
#         image_filename='chest_xray.jpg'
#     )
    
#     # 시각화
#     explainer.visualize_gradients_with_bbox(
#         image, 
#         analysis_results['gradients']['absolute'],
#         analysis_results['bounding_boxes']
#     )
    
#     # 겹침 분석 결과 출력
#     explainer.print_overlap_analysis(analysis_results['overlap_metrics'])
#     """
    
#     print("Enhanced XAI 분석기가 준비되었습니다!")
#     print("사용 방법:")
#     print("1. 학습된 모델 로드")
#     print("2. X-ray 이미지 로드") 
#     print("3. bounding box CSV 파일 준비")
#     print("4. comprehensive_analysis_with_bbox() 호출")
#     print("5. visualize_gradients_with_bbox()로 시각화")

# if __name__ == "__main__":
#     example_usage_with_bbox()