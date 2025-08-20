from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Callable, Optional, Dict, Any
from PIL import Image

class ImageProcessor:
    """
    전처리(그레이스케일/CLAHE/블러/스트레치/샤프닝/옵션, RGB 변환)
    """

    def preprocess(
        self,
        image: np.ndarray,
        steps: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        save_format: str = 'jpg'
    ) -> np.ndarray:
        """
        - 동작: 전처리(그레이스케일/CLAHE/블러/스트레치/샤프닝)
        - 출력: RGB (H,W,3) 배열을 반환. (입력은 BGR/GRAY 모두 허용)
        """
        if steps is None:
            steps = {
                'convert_grayscale': True,
                'apply_clahe': True,
                'apply_gaussian_blur': True,
                'apply_min_max_stretch': True,
                'apply_sharpening': False,
            }
        
        if params is None:
            params = {
                'clahe': {'clip_limit': 2.0, 'grid_size': (8, 8)},
                'gaussian_blur': {'radius': 1.0},
                'min_max_stretch': {'lower_percentile': 1, 'upper_percentile': 99},
                'sharpening': {'radius': 2, 'percent': 150, 'threshold': 3}
            }
        
        # 전처리 전 복사
        img = image.copy()

        # 1) 그레이스케일 → 3채널(모델 파이프라인 유지 용이)
        if steps.get('convert_grayscale', False):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 2) CLAHE (L 채널에만)
        if steps.get('apply_clahe', False):
            clip_limit = float(params.get('clahe', {}).get('clip_limit', 2.0))
            grid_size  = params.get('clahe', {}).get('grid_size', (8, 8))
            # list로 들어오면 tuple로 변환
            if isinstance(grid_size, list):
                grid_size = tuple(grid_size)
            # CLAHE 인스턴스 생성
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

            # LAB으로 변환 후 L 채널에만 CLAHE 적용
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            L_eq = clahe.apply(L)
            lab_eq = cv2.merge((L_eq, A, B))
            img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 3) 가우시안 블러
        if steps.get('apply_gaussian_blur', False):
            radius = int(round(float(params.get('gaussian_blur', {}).get('radius', 1))))
            radius = max(radius, 0)
            if radius > 0:
                k = 2 * radius + 1  # 홀수 커널
                img = cv2.GaussianBlur(img, (k, k), 0)

        # 4) Min-Max Stretching (채널별)
        if steps.get('apply_min_max_stretch', False):
            p_low  = float(params.get('min_max_stretch', {}).get('lower_percentile', 1))
            p_high = float(params.get('min_max_stretch', {}).get('upper_percentile', 99))
            out = img.astype(np.float32)
            for c in range(3):
                ch = out[..., c]
                lo = np.percentile(ch, p_low)
                hi = np.percentile(ch, p_high)
                if hi > lo:
                    ch = (ch - lo) * 255.0 / (hi - lo + 1e-6)
                    out[..., c] = np.clip(ch, 0, 255)
            img = out.astype(np.uint8)

        # 5) 샤프닝(언샤프 마스크)
        if steps.get('apply_sharpening', False):
            r   = int(params.get('sharpening', {}).get('radius', 2))
            pct = float(params.get('sharpening', {}).get('percent', 150))
            thr = float(params.get('sharpening', {}).get('threshold', 3))
            if r > 0:
                k = 2 * r + 1
                blurred = cv2.GaussianBlur(img, (k, k), 0)
                sharpened = cv2.addWeighted(img, pct / 100.0, blurred, 1.0 - pct / 100.0, 0)
                low_contrast = (np.abs(img.astype(np.int16) - blurred.astype(np.int16)) < thr)
                img = np.where(low_contrast, img, sharpened).astype(np.uint8)

        # 6) BGR → RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 저장 옵션 처리
        if output_path and save_format != "none":
            if save_format == "npz":
                np.savez_compressed(output_path, image=img)
            elif save_format in ("png", "jpg"):
                Image.fromarray(img).save(output_path, format=save_format.upper())

        return img  # RGB (H,W,3)
    
    def _cv2_interp(self, name: str, is_mask: bool = False):
        """OpenCV 보간 방식 반환"""
        if is_mask:
            return cv2.INTER_NEAREST
        name = (name or "").lower()
        if name in ("area", "inter_area"):
            return cv2.INTER_AREA
        if name in ("linear", "bilinear", "inter_linear"):
            return cv2.INTER_LINEAR
        if name in ("cubic", "bicubic", "inter_cubic"):
            return cv2.INTER_CUBIC
        return cv2.INTER_AREA

class ImagePackager:
    """
    - 입력:
        preprocessed_img_path: 전처리(clahe/blur/stretch 등) 완료된 이미지 경로
        mask_path            : 세그멘테이션(폐/병변 등) 완료된 마스크 경로
      (둘 다 원본 해상도 기준이며, 이 함수에서는 세그 수행/전처리를 하지 않음)

    - 동작:
        1) RGB 이미지/마스크 로드
        2) 원본 해상도에서 letterbox 메타를 기반으로
           분류 입력 해상도(224)로 동일 규칙 리사이즈
        3) (224,224,3) RGB + (224,224,1) mask → (224,224,4) 패킹
        4) 필요 시 .npz로 저장

    - 출력:
        dict {
            "image_224": (224,224,3) RGB np.uint8,
            "mask_224" : (224,224)    np.uint8 (0/255),
            "packed"   : (224,224,4)  np.uint8
        }
    """

    # --------------- helpers ---------------
    @staticmethod
    def _cv2_interp(name: str, is_mask: bool = False):
        if is_mask:
            return cv2.INTER_NEAREST
        name = (name or "").lower()
        if name in ("area", "inter_area"):   return cv2.INTER_AREA
        if name in ("linear", "bilinear"):   return cv2.INTER_LINEAR
        if name in ("cubic", "bicubic"):     return cv2.INTER_CUBIC
        return cv2.INTER_AREA

    def _letterbox(self, img: np.ndarray, target: int, pad_value: int, interp: str, is_mask: bool=False):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid image size.")
        scale = float(target) / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=self._cv2_interp(interp, is_mask=is_mask))
        pad_h, pad_w = target - nh, target - nw
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        border_val = (pad_value,)*3 if (img.ndim == 3) else pad_value
        squared = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_val)
        meta = {"scale": scale, "pad": (top, bottom, left, right), "orig_hw": (h, w)}
        return squared, meta

    def _undo_letterbox(self, square: np.ndarray, meta: Dict[str, Any], is_mask: bool, interp_img="area", interp_mask="nearest"):
        top, bottom, left, right = meta["pad"]
        crop = square[top:square.shape[0]-bottom, left:square.shape[1]-right]
        h, w = meta["orig_hw"]
        restored = cv2.resize(
            crop, (w, h),
            interpolation=self._cv2_interp(interp_mask if is_mask else interp_img, is_mask=is_mask)
        )
        return restored

    # --------------- main ---------------
    def build_npz_from_preprocessed(
        self,
        preprocessed_img_path: str,                 # 전처리 완료 이미지
        mask_path: str,                             # 세그멘테이션 완료 마스크
        save_npz_path: Optional[str] = None,        # 저장 경로 (없으면 저장 안 함)
        meta_info: Optional[Dict[str, Any]] = None, # 함께 저장할 메타(선택)
        *,
        # 분류/패킹 관련 (224 사이즈, 종횡비 보존 리사이즈 + 패딩)
        cls_input_size: int = 224,
        cls_pad_value: int = 0,
        cls_interp_image: str = "area",
        # 마스크 리사이즈
        mask_interp: str = "nearest",
        # 중간결과 저장(디버깅용)
        save_intermediate: bool = False,
    ) -> Dict[str, Any]:

        # 0) 이미지/마스크 로드
        bgr = cv2.imread(preprocessed_img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(preprocessed_img_path)
        # 전처리 이미지는 보통 BGR로 저장되어 있으므로 RGB로 변환
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 마스크 이진화 안전장치 (0/255)
        mask = (mask > 0).astype(np.uint8) * 255

        # 1) 분류 입력 사이즈로 letterbox (이미지)
        rgb_224, rgb_meta = self._letterbox(
            rgb, target=cls_input_size, pad_value=cls_pad_value,
            interp=cls_interp_image, is_mask=False
        )  # (224,224,3)

        # 2) 마스크도 동일 규칙(원본 해상도 기준)으로 224로 맞춤
        #    - 먼저 mask를 '그냥' 224로 letterbox (원본 기준)
        mask_224, _ = self._letterbox(
            mask, target=cls_input_size, pad_value=0,
            interp=mask_interp, is_mask=True
        )  # (224,224)

        # 3) 최종 후처리
        mask_224 = (mask_224 > 127).astype(np.uint8) * 255
        mask_224_1c = np.expand_dims(mask_224, axis=-1)

        # 4) 4채널 패킹
        packed_4ch = np.concatenate([rgb_224.astype(np.uint8), mask_224_1c], axis=-1)  # (224,224,4)

        if save_intermediate:
            base, _ = os.path.splitext(preprocessed_img_path)
            # 저장은 시각화를 위해 BGR로 변환
            cv2.imwrite(f"{base}.cls_in.png", cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{base}.mask_224.png", mask_224)

        # 5) npz 저장(옵션)
        if save_npz_path:
            payload = {"image": packed_4ch}
            if meta_info:
                payload.update(meta_info)
            np.savez_compressed(save_npz_path, **payload)

        return {
            "image_224": rgb_224,   # (224,224,3) RGB
            "mask_224":  mask_224,  # (224,224)    0/255
            "packed":    packed_4ch # (224,224,4)
        }