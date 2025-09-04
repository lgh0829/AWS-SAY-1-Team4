#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import tarfile
from pathlib import Path
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator

# ===== 프로젝트 유틸 경로 추가 및 .env 로드 =====
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import dotenv
    dotenv.load_dotenv()  # CWD 기준
    dotenv.load_dotenv(Path(__file__).parent / ".env")  # 스크립트 폴더 기준
except Exception:
    pass

from common.cloud_utils.s3_handler import S3Handler  # ← 주신 코드와 동일 방식 사용


# =========================
# YAML 로딩 + 환경변수 치환
# =========================
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
            result[key] = [
                process_yaml_dict(item) if isinstance(item, dict) else replace_env_vars(item)
                for item in value
            ]
        else:
            result[key] = replace_env_vars(value)
    return result

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = process_yaml_dict(cfg)

    # 치환 실패 검출(남은 ${...})
    unresolved = []
    def scan(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                scan(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(x, str) and re.search(r"\$\{[A-Za-z0-9_]+\}", x):
            unresolved.append((prefix, x))
    scan(cfg)

    if unresolved:
        msg = "\n".join([f"  - {k}: {v}" for k, v in unresolved])
        raise ValueError(
            "환경변수 치환이 끝나지 않은 값이 있습니다 (.env 또는 쉘 환경변수를 확인하세요):\n" + msg
        )
    return cfg


# =========================
# TensorBoard event helpers
# =========================
def pick_first_existing(tags_available, candidates):
    for t in candidates:
        if t in tags_available:
            return t
    return None

def load_history_from_events(event_path: str):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])

    # 폭넓은 태그 후보(대소문자/네임변형 지원)
    candidates = {
        "train_loss": ["Loss/train", "loss/train", "training_loss", "Train/Loss", "train/loss"],
        "val_loss":   ["Loss/val",   "loss/val",   "validation_loss", "Val/Loss", "val/loss"],
        "train_acc":  ["Acc/train",  "Accuracy/train", "acc/train", "training_accuracy",
                       "Train/Acc", "train/acc"],
        "val_acc":    ["Acc/val",    "Accuracy/val",   "acc/val",   "validation_accuracy",
                       "Val/Acc", "val/acc"],
    }
    chosen = {k: pick_first_existing(scalars, v) for k, v in candidates.items()}

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    for key, tag in chosen.items():
        if not tag:
            continue
        events = ea.Scalars(tag)
        history[key] = [e.value for e in events]

    return history, scalars

def find_auc_series(tags_available):
    """AUC/ROC_AUC로 로그한 스칼라 자동 탐지"""
    auc_map = {}
    regexes = [
        r"^(AUC|ROC_AUC)/(?P<cls>.+)$",
        r"^(?P<cls>.+)/(AUC|ROC_AUC)$",
        r"^auc[_/](?P<cls>.+)$",
        r"^(?P<cls>.+)_auc$",
    ]
    for tag in tags_available:
        for rx in regexes:
            m = re.match(rx, tag, flags=re.IGNORECASE)
            if m:
                cls = m.group("cls").replace(" ", "")
                auc_map.setdefault(cls, tag)
                break
    return auc_map

def load_auc_series(event_path: str):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    auc_map = find_auc_series(scalars)

    series = {}
    for cls, tag in auc_map.items():
        ev = ea.Scalars(tag)
        xs = [e.step for e in ev]
        ys = [e.value for e in ev]
        series[cls] = (xs, ys)
    return series

def plot_history(history, out_png):
    plt.figure(figsize=(12, 4))
    # Loss
    plt.subplot(1, 2, 1)
    if history["train_loss"]: plt.plot(history["train_loss"], label="Training Loss")
    if history["val_loss"]:   plt.plot(history["val_loss"],   label="Validation Loss")
    plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    if history["train_loss"] or history["val_loss"]: plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    if history["train_acc"]: plt.plot(history["train_acc"], label="Training Accuracy")
    if history["val_acc"]:   plt.plot(history["val_acc"],   label="Validation Accuracy")
    plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    if history["train_acc"] or history["val_acc"]: plt.legend()
    plt.grid(True)

    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"[SAVE] training curves -> {out_png}")

def plot_auc_series(auc_series, out_png):
    if not auc_series:
        print("[INFO] No AUC-like scalar tags in event file. Skip AUC plot.")
        return False

    has_steps = any(len(v[0]) > 1 for v in auc_series.values())
    if has_steps:
        plt.figure(figsize=(7, 5))
        for cls, (xs, ys) in auc_series.items():
            plt.plot(xs, ys, label=cls)
        plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.ylim(0.0, 1.0)
        plt.title("ROC AUC over Epochs"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVE] AUC curves -> {out_png}")
        return True

    # 단일값만 있으면 bar
    plt.figure(figsize=(7, 5))
    classes = list(auc_series.keys())
    values = [auc_series[c][1][-1] if auc_series[c][1] else float("nan") for c in classes]
    plt.bar(classes, values); plt.ylim(0.0, 1.0); plt.ylabel("ROC AUC")
    plt.title("ROC AUC"); plt.grid(axis="y")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"[SAVE] AUC bars -> {out_png}")
    return True


# =========================
# S3 → output.tar.gz → 이벤트 파일
# =========================
def event_from_config(cfg: dict, workdir: Path) -> Path:
    """
    주신 로더와 동일한 규칙:
    s3_key = f"{prefix}/output/{job_name}/output/output.tar.gz"
    """
    bucket = cfg["s3"]["bucket_name"]
    prefix = cfg["s3"]["prefix"].rstrip("/")
    job    = cfg["s3"]["job_name"]

    s3_key = f"{prefix}/output/{job}/output/output.tar.gz"
    print(f"[INFO] downloading from s3://{bucket}/{s3_key}")

    workdir.mkdir(parents=True, exist_ok=True)
    local_tar_path = workdir / "output.tar.gz"

    # S3Handler 사용(질문 스크립트와 동일)
    s3 = S3Handler(bucket)
    s3.download_file(s3_key, local_tar_path)

    extract_dir = workdir / "tb_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # 첫 이벤트 파일 찾기
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return Path(root) / f

    raise FileNotFoundError("No events.out.tfevents* found after extracting output.tar.gz")


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Plot training history and ROC AUC from a TensorBoard event file or from config(S3)."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event", help="Path to events.out.tfevents.* (or directory containing one)")
    g.add_argument("--config", help="Path to evaluation_config.yaml (uses S3Handler to fetch output.tar.gz)")

    ap.add_argument("--outdir", default="./plots", help="Output directory for PNGs")
    ap.add_argument("--history-name", default="training_history.png")
    ap.add_argument("--auc-name", default="roc_auc.png")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 이벤트 파일 확보
    if args.config:
        cfg = load_config(args.config)
        event_path = event_from_config(cfg, outdir)
    else:
        event_path = args.event
        if os.path.isdir(event_path):
            found = None
            for root, _, files in os.walk(event_path):
                for f in files:
                    if f.startswith("events.out.tfevents"):
                        found = os.path.join(root, f); break
                if found: break
            if not found:
                raise FileNotFoundError("No events.out.tfevents* under the directory.")
            event_path = found
        if not os.path.exists(event_path):
            raise FileNotFoundError(f"Event file not found: {event_path}")
        event_path = Path(event_path)

    # 2) 히스토리 플롯
    history, available = load_history_from_events(str(event_path))
    print("[TB] available scalar tags:", available)
    plot_history(history, outdir / args.history_name)

    # 3) AUC 플롯(로그되어 있을 때만)
    auc_series = load_auc_series(str(event_path))
    plotted = plot_auc_series(auc_series, outdir / args.auc_name)
    if not plotted:
        print("[INFO] AUC scalars not found. "
              "필요하다면 학습 중 예: writer.add_scalar('AUC/pneumonia', val_auc_p, epoch) 형태로 로깅하세요.")

if __name__ == "__main__":
    main()