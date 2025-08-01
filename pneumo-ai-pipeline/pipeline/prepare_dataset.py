import os
import sys
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
import argparse
import yaml
import re
import dotenv
import boto3

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.pneumo_utils.preprocessing import ImagePreprocessor
from common.cloud_utils.s3_handler import S3Handler

# 환경 변수 로드
dotenv.load_dotenv()
dotenv.load_dotenv(Path(__file__).parent / '.env')


def replace_env_vars(value):
    pattern = r"\${([A-Za-z0-9_]+)}"
    if isinstance(value, str):
        for var in re.findall(pattern, value):
            env_val = os.environ.get(var)
            if env_val:
                value = value.replace(f"${{{var}}}", env_val)
    return value


def process_yaml(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict): result[k] = process_yaml(v)
        elif isinstance(v, list): result[k] = [process_yaml(item) if isinstance(item, dict) else replace_env_vars(item) for item in v]
        else: result[k] = replace_env_vars(v)
    return result


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return process_yaml(cfg)


def find_mask(mask_root, rel_base):
    """
    mask_root 아래 rel_base(.txt 또는 .png) 파일 탐색
    """
    mask_root = os.path.normpath(mask_root)
    for ext in ['.txt', '.png']:
        candidate = os.path.normpath(os.path.join(mask_root, rel_base + ext))
        if os.path.exists(candidate):
            return candidate
    return None


def load_mask(mask_path, size):
    """
    Load mask (.txt 또는 이미지), resize to size=(W,H) via NEAREST, binarize, return (1,H,W) uint8 tensor.
    """
    if not mask_path:
        print("⚠️ mask_path is None")
        return None
    try:
        if mask_path.endswith('.txt'):
            arr = np.loadtxt(mask_path, dtype=np.uint8)
        else:
            arr = np.array(Image.open(mask_path).convert('L'))
        W, H = size
        resized = cv2.resize(arr, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        binarized = (resized > 0).astype(np.uint8)
        tensor = torch.tensor(binarized, dtype=torch.uint8).unsqueeze(0)
        print(f"✅ mask loaded: {mask_path} -> {tensor.shape}")
        return tensor
    except Exception as e:
        print(f"❌ load_mask error: {e}")
        return None


def preprocess_image(img, preprocessor):
    print(f"🌀 Preprocessing image size={img.size}")
    proc = preprocessor.process_image(img)
    arr = np.array(proc) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def prepare_combined_tensor(img_path, cfg, preproc, do_segment, do_preprocess):
    # root 디렉토리
    data_root = cfg['directories']['data_dir']
    in_root = os.path.join(data_root, cfg['directories']['input_dir'])
    seg_root = os.path.join(data_root, cfg['directories']['segmented_dir'])

    # img_path 상대경로 (in_root 기준)
    rel = os.path.relpath(img_path, in_root)
    rel_base = os.path.splitext(rel)[0]  # e.g. 'test/0/00024175_003'
    print(f"\n📂 Processing {rel}")

    img = Image.open(img_path).convert('L')
    # 이미지 전처리
    if do_preprocess:
        img_tensor = preprocess_image(img, preproc)
    else:
        arr = np.array(img) / 255.0
        img_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        print("🔹 Raw image used")

    _, H, W = img_tensor.shape

    # 마스크 로드 시도
    mask_path = None
    if do_segment:
        mask_path = find_mask(seg_root, rel_base)
        mask_tensor = load_mask(mask_path, size=(W, H))
        if mask_tensor is None:
            print("⚠️ Mask not found or load failed, running auto-segmentation")
            try:
                from common.pneumo_utils.segmentation import LungSegmenter
                seg = LungSegmenter()
                _, raw_mask, _ = seg.segment_image(img_path)
                raw = (raw_mask > 0).astype(np.uint8)
                resized = cv2.resize(raw, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.tensor(resized, dtype=torch.uint8).unsqueeze(0)
                print(f"✅ Auto-seg mask: {mask_tensor.shape}")
            except Exception as e:
                print(f"❌ segmentation error: {e}")
                mask_tensor = torch.zeros((1, H, W), dtype=torch.uint8)
    else:
        mask_tensor = torch.zeros((1, H, W), dtype=torch.uint8)
        print("🔹 No segmentation, zero mask used")

    img_rgb = img_tensor.expand(3, -1, -1)
    combined = torch.cat([img_rgb, mask_tensor.float()], dim=0)
    print(f"✅ Combined tensor: {combined.shape}")
    return combined


def prepare_training_data(cfg_path):
    cfg = load_config(cfg_path)
    steps = cfg.get('steps', {})
    do_segment = steps.get('segment_lungs', False)
    do_preprocess = steps.get('preprocess_images', False)
    upload_s3 = steps.get('upload_to_s3', False)

    out_root = 'pt_output'
    os.makedirs(out_root, exist_ok=True)
    preproc = ImagePreprocessor(cfg['preprocessing']) if do_preprocess else None
    s3 = boto3.client('s3') if upload_s3 else None
    bucket = cfg['s3']['bucket_name']; prefix = cfg['s3']['prefix']

    in_root = os.path.join(cfg['directories']['data_dir'], cfg['directories']['input_dir'])
    all_imgs = [os.path.join(r, f)
                for r, _, files in os.walk(in_root)
                for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]
    print(f"🔎 Found {len(all_imgs)} images; segment={do_segment}, preprocess={do_preprocess}, upload={upload_s3}")

    for idx, img_path in enumerate(all_imgs, 1):
        rel = os.path.relpath(img_path, in_root)
        rel_base = os.path.splitext(rel)[0]
        out_path = os.path.join(out_root, rel_base + '.pt')
        # 하위 디렉토리 생성
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(all_imgs)}] {rel} -> {out_path}")
        try:
            tensor4 = prepare_combined_tensor(img_path, cfg, preproc, do_segment, do_preprocess)
            torch.save(tensor4, out_path)
            print(f"💾 Saved: {out_path}")
            if upload_s3:
                s3.upload_file(out_path, bucket, f"{prefix}/processed_pt/{rel_base}.pt")
                print(f"☁️ Uploaded: {prefix}/processed_pt/{rel_base}.pt")
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    print("\n🎉 All done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4-channel dataset preparer preserving folder structure')
    parser.add_argument('--config', required=True, help='Config file path')
    args = parser.parse_args()
    prepare_training_data(args.config)


