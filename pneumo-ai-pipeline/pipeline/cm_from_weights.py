
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_transforms(image_size):
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return eval_tf

def imagefolder_keep_classes(root, transform, keep=('0','1')):
    ds = datasets.ImageFolder(root, transform=transform)
    keep_ids = {ds.class_to_idx[k] for k in keep if k in ds.class_to_idx}
    keep_idx = [i for i, y in enumerate(ds.targets) if y in keep_ids]
    return Subset(ds, keep_idx)

def build_model(device):
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 1)
    return model.to(device)

def load_checkpoint(weights_path, model, device):
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)  # support plain state_dict or saved dict
    model.load_state_dict(state_dict)
    # best_threshold might be stored under cfg
    best_thr = 0.5
    if isinstance(ckpt, dict) and 'cfg' in ckpt and isinstance(ckpt['cfg'], dict):
        best_thr = float(ckpt['cfg'].get('best_threshold', 0.5))
    return best_thr

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(probs)
        ys.append(y.detach().cpu().numpy())
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps)

    # Metrics
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    report = classification_report(y_true, y_pred, target_names=['0','1'], digits=4, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(acc),
        'auroc': float(auroc),
        'threshold': float(threshold),
        'report': report
    }, (y_true, y_prob, y_pred)

def main():
    ap = argparse.ArgumentParser(description="Compute confusion matrix from weights on an ImageFolder split")
    ap.add_argument('--weights', type=str, required=True, help='Path to best_model.pt (or state_dict)')
    ap.add_argument('--data-dir', type=str, required=True, help='Path to split folder (e.g., ./val or ./test)')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--use-best-threshold', action='store_true', help='Use best_threshold from checkpoint if available')
    ap.add_argument('--threshold', type=float, default=0.5, help='Manual threshold when not using best-threshold')
    ap.add_argument('--save-csv', type=str, default='', help='Optional: path to save raw predictions CSV')
    args = ap.parse_args()

    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tf = get_transforms(args.image_size)
    ds = imagefolder_keep_classes(args.data_dir, tf, keep=('0','1'))
    pin = (device.type == 'cuda')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin)

    model = build_model(device)
    best_thr_in_ckpt = load_checkpoint(args.weights, model, device)
    thr = best_thr_in_ckpt if args.use_best_threshold else args.threshold

    metrics, (y_true, y_prob, y_pred) = evaluate(model, loader, device, threshold=thr)

    print("=== Confusion Matrix (threshold={:.3f}) ===".format(metrics['threshold']))
    print("TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}".format(**metrics))
    print("\nAccuracy: {:.4f}".format(metrics['accuracy']))
    print("AUROC: {}".format("nan" if np.isnan(metrics['auroc']) else f"{metrics['auroc']:.4f}"))
    print("\n=== Classification Report ===")
    print(metrics['report'])

    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "y_true", "y_prob", "y_pred"])
            for i, (yt, yp, yd) in enumerate(zip(y_true, y_prob, y_pred)):
                w.writerow([i, int(yt), float(yp), int(yd)])
        print(f"Saved raw predictions to: {args.save_csv}")

if __name__ == "__main__":
    main()
