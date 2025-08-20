import os
import sys
from pathlib import Path
import argparse
import re
import yaml
import dotenv
import cv2
import numpy as np

# 프로젝트 루트로 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.pneumo_utils.segmentation import LungSegmenter
from common.pneumo_utils.processing_npz import ImageProcessor, ImagePackager
from common.cloud_utils.s3_handler import S3Handler

dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')

# -----------------------------
# YAML helpers (env var expand)
# -----------------------------

def replace_env_vars(value):
    if isinstance(value, str):
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value:
                value = value.replace(f"${{{var_name}}}", env_value)
        return value
    return value

def process_yaml_dict(yaml_dict):
    result = {}
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            result[key] = process_yaml_dict(value)
        elif isinstance(value, list):
            result[key] = [process_yaml_dict(item) if isinstance(item, dict) else replace_env_vars(item) for item in value]
        else:
            result[key] = replace_env_vars(value)
    return result

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = process_yaml_dict(config)
    return config

# -----------------------------
# IO helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def iter_image_files(root_dir: str):
    root = Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def rel_to(base: str, path: Path) -> Path:
    return Path(os.path.relpath(path, base))


# -----------------------------
# Core pipeline
# -----------------------------
def prepare_training_data(config_path: str):
    cfg = load_config(config_path)

    steps = cfg.get("steps", {})
    s3_cfg = cfg.get("s3", {})
    dirs = cfg.get("directories", {})
    seg_cfg = cfg.get("segmentation", {})
    prep_cfg = cfg.get("preprocessing", {})
    cls_cfg = cfg.get("classification", {})
    npz_cfg = cfg.get("npz", {})

    # Local directories
    data_dir = Path(dirs["data_dir"])
    input_dir = data_dir / dirs["input_dir"]
    segmented_dir = data_dir / dirs["segmented_dir"]
    preprocessed_dir = data_dir / dirs["preprocessed_dir"]
    packing_dir = data_dir / dirs["packing_dir"]

    for d in [input_dir, segmented_dir, preprocessed_dir, packing_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # S3 handler
    bucket = s3_cfg.get("bucket_name")
    s3_prefix_root = s3_cfg.get("prefix", "")
    s3_input_prefix = s3_cfg.get("input_prefix", "")
    s3_segmented_prefix = s3_cfg.get("segmented_prefix", "")
    s3_preprocessed_prefix = s3_cfg.get("preprocessed_prefix", "")
    s3_packing_prefix = s3_cfg.get("packing_prefix", "")

    s3 = S3Handler(bucket) if bucket else None

    # -----------------------------
    # 1) Download from S3
    # -----------------------------
    if steps.get("download_from_s3", False):
        if not s3:
            raise ValueError("S3 bucket_name is required for download.")
        remote_input = "/".join([p for p in [s3_prefix_root, s3_input_prefix] if p])
        print(f"[S3] Downloading data: s3://{bucket}/{remote_input} -> {input_dir}")
        s3.download_directory(remote_input, str(input_dir))
        print("[S3] Download complete.")
    else:
        print("[Skip] S3 download step")
    
    

    # -----------------------------
    # 2) Segmentation on original images
    #    - Save masked image AND binary mask
    # -----------------------------
    if steps.get("segment_lungs", False):
        print("[Seg] Starting segmentation on original images...")
        segmenter = LungSegmenter()
        # Two subfolders inside segmented_dir for clarity
        seg_mask_root = segmented_dir / "masks"
        seg_img_root = segmented_dir / "masked"
        seg_mask_root.mkdir(parents=True, exist_ok=True)
        seg_img_root.mkdir(parents=True, exist_ok=True)

        total = 0
        for src_path in iter_image_files(str(input_dir)):
            rel = rel_to(str(input_dir), src_path)
            # Output paths
            out_mask_path = seg_mask_root / rel.with_suffix(".png")
            out_img_path = seg_img_root / rel  # keep original suffix for masked image
            ensure_parent(out_mask_path)
            ensure_parent(out_img_path)

            try:
                masked_img, combined_mask, _ = segmenter.segment_image(str(src_path))
                # Save both masked image and mask
                cv2.imwrite(str(out_img_path), masked_img)
                cv2.imwrite(str(out_mask_path), combined_mask)
                total += 1
                if total % 100 == 0:
                    print(f"[Seg] Processed {total} images...")
            except Exception as e:
                print(f"[Seg][ERR] {src_path}: {e}")

        print(f"[Seg] Done. Total segmented: {total}")
    else:
        print("[Skip] Segmentation step")

    # -----------------------------
    # 3) Preprocess on original images
    # -----------------------------
    if steps.get("preprocess_images", False):
        print("[Prep] Starting preprocessing on original images...")
        proc = ImageProcessor()
        total = 0
        # # ######### **** 약간 변경 **** #########
        # for src_path in iter_image_files(str(segmented_dir / "masked")):
        for src_path in iter_image_files(str(input_dir)):
            rel = rel_to(str(input_dir), src_path)
            out_pre_path = preprocessed_dir / rel.with_suffix(".png")  # keep same suffix
            ensure_parent(out_pre_path)

            # Load BGR and preprocess via class API (expects numpy image)
            bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[Prep][WARN] Failed to load: {src_path}")
                continue
            try:
                _ = proc.preprocess(
                    image=bgr,
                    steps=prep_cfg.get("steps"),
                    params=prep_cfg.get("params"),
                    output_path=str(out_pre_path),
                    save_format=out_pre_path.suffix.lower().lstrip(".") or "jpg",
                )
                total += 1
                if total % 200 == 0:
                    print(f"[Prep] Processed {total} images...")
            except Exception as e:
                print(f"[Prep][ERR] {src_path}: {e}")

        print(f"[Prep] Done. Total preprocessed: {total}")
    else:
        print("[Skip] Preprocessing step")

    # -----------------------------
    # 4) Package to NPZ (224x224x4)
    #     - uses preprocessed image + segmentation mask
    # -----------------------------
    if steps.get("packing_conversion", False):
        print("[Pack] Building NPZ from preprocessed + masks...")
        packager = ImagePackager()
        # Masks are under segmented_dir / "masks"
        seg_mask_root = segmented_dir / "masks"
        total = 0
        for pre_img_path in iter_image_files(str(preprocessed_dir)):
            rel = rel_to(str(preprocessed_dir), pre_img_path)
            # Corresponding mask path mirrors original folder structure
            mask_path = seg_mask_root / rel.with_suffix(".png")
            if not mask_path.exists():
                # Try alternative: maybe mask saved with same suffix
                alt = seg_mask_root / rel
                if alt.exists():
                    mask_path = alt
                else:
                    print(f"[Pack][WARN] Mask not found for {pre_img_path} -> {mask_path}")
                    continue

            # Output .npz path under packing_dir (keep folder structure)
            npz_out = (packing_dir / rel).with_suffix(".npz")
            ensure_parent(npz_out)

            # Optional meta fields (best-effort)
            stem = pre_img_path.stem
            meta = {}
            if npz_cfg.get("save_meta", True):
                # fill a subset if present in filename
                meta["study_id"] = stem

            try:
                packager.build_npz_from_preprocessed(
                    preprocessed_img_path=str(pre_img_path),
                    mask_path=str(mask_path),
                    save_npz_path=str(npz_out),
                    meta_info=meta,
                    cls_input_size=int(cls_cfg.get("input_size", 224)),
                    cls_pad_value=int(cls_cfg.get("pad_value", 0)),
                    cls_interp_image=str(cls_cfg.get("interpolation_image", "area")),
                    mask_interp="nearest",
                    save_intermediate=False,
                )
                total += 1
                if total % 200 == 0:
                    print(f"[Pack] Packed {total} npz...")
            except Exception as e:
                print(f"[Pack][ERR] {pre_img_path}: {e}")

        print(f"[Pack] Done. Total npz: {total}")
    else:
        print("[Skip] NPZ packing step")

    # -----------------------------
    # 5) Upload to S3
    # -----------------------------
    if steps.get("upload_to_s3", False):
        if not s3:
            raise ValueError("S3 bucket_name is required for upload.")
        remote_packing = "/".join([p for p in [s3_prefix_root, s3_packing_prefix] if p])
        print(f"[S3] Uploading NPZ dir: {packing_dir} -> s3://{bucket}/{remote_packing}")
        s3.upload_directory(str(packing_dir), remote_packing)
        print("[S3] Upload complete.")
    else:
        print("[Skip] S3 upload step")

    # ######### **** 약간 변경 **** #########
    # # -----------------------------
    # # 5) Upload to S3
    # # -----------------------------
    # if steps.get("upload_to_s3", False):
    #     if not s3:
    #         raise ValueError("S3 bucket_name is required for upload.")
    #     remote_preprocessed = "/".join([p for p in [s3_prefix_root, s3_preprocessed_prefix] if p])
    #     print(f"[S3] Uploading NPZ dir: {preprocessed_dir} -> s3://{bucket}/{remote_preprocessed}")
    #     s3.upload_directory(str(preprocessed_dir), remote_preprocessed)
    #     print("[S3] Upload complete.")
    # else:
    #     print("[Skip] S3 upload step")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 데이터셋 준비 (seg + preprocess + npz pack)")
    parser.add_argument("--config", required=True, help="설정 파일 경로")
    args = parser.parse_args()
    prepare_training_data(args.config)