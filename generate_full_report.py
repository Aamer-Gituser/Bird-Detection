"""
generate_full_report.py
------------------------------------------------------------
ONE SCRIPT → generates a FULL PDF report + separate images for:

A) YOLO SOLO:
   - reads Ultralytics results.csv (loss, mAP, precision, recall curves)
   - reads best.pt evaluation summary from results.csv last row

B) RESNET SOLO:
   - evaluates on crop_dataset-2/test (or crop_dataset/test)
   - computes: TP TN FP FN precision recall f1 accuracy
   - saves Confusion Matrix PNG

C) COMBINED (YOLO + ResNet):
   - runs YOLO predictions on YOLO dataset test images
   - filters YOLO 'bird' detections using ResNet classifier
   - evaluates detection metrics vs ground-truth labels (IoU=0.50):
        precision, recall, TP FP FN
   - also reports how many detections were removed by classifier

OUTPUT:
  reports/final_report/
      report.pdf
      yolo_curves.png
      resnet_confusion_matrix.png
      resnet_curves.png  (if history CSV exists)
      combined_summary.json
      resnet_metrics.json
      yolo_last_row.json

How to run (PowerShell):
  python generate_full_report.py --yolo-run runs_yolo/yolov8m_bird_no_bird_v22 \
    --yolo-dataset data/yolo_dataset-2 \
    --resnet-weights classifier/weights/resnet50_best_v2.pth \
    --crop-test classifier/crop_dataset-2/test
------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from ultralytics import YOLO

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def plot_df_curves(df: pd.DataFrame, cols: list[str], title: str, out_png: Path):
    plt.figure(figsize=(10, 5))
    for c in cols:
        if c in df.columns:
            plt.plot(df[c].values, label=c)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_png: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(classes)), classes, rotation=20)
    plt.yticks(range(len(classes)), classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def yolo_txt_to_xyxy(line: str, img_w: int, img_h: int):
    """
    YOLO label format: cls cx cy w h (normalized)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])

    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return cls, np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


# ---------------------------
# ResNet model loader
# ---------------------------
def load_resnet50_binary(weights_path: Path, device: str):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    state = torch.load(str(weights_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    elif isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        raise RuntimeError("Unknown ResNet checkpoint format.")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def resnet_predict(model, bgr_img: np.ndarray, device: str, image_size=224):
    """
    returns (label_idx, prob_of_pred)
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    x = tfm(rgb).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    return idx, conf


# ---------------------------
# YOLO SOLO: read results.csv
# ---------------------------
def yolo_solo_report(yolo_run_dir: Path, out_dir: Path):
    results_csv = yolo_run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"YOLO results.csv not found at: {results_csv}")

    df = pd.read_csv(results_csv)

    # Save last row as JSON (summary)
    last_row = {k: safe_float(v) for k, v in df.iloc[-1].to_dict().items()}
    write_json(out_dir / "yolo_last_row.json", {"run_dir": str(yolo_run_dir), "last_row": last_row})

    # Plot curves (Ultralytics column names vary; we try common ones)
    yolo_cols = [
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95",
    ]
    plot_df_curves(df, yolo_cols, "YOLO Training / Validation Curves", out_dir / "yolo_curves.png")

    # Return key metrics if present
    keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"]
    summary = {}
    for k in keys:
        if k in df.columns:
            summary[k] = safe_float(df.iloc[-1][k])
    return summary


# ---------------------------
# ResNet SOLO: eval + confusion matrix
# ---------------------------
def resnet_solo_report(crop_test_dir: Path, resnet_weights: Path, out_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_resnet50_binary(resnet_weights, device=device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(str(crop_test_dir), transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().tolist())

    classes = ds.classes
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, out_dir / "resnet_confusion_matrix.png")

    # Compute TP TN FP FN for bird (if exists)
    bird_idx = classes.index("bird") if "bird" in classes else 0

    TP = int(cm[bird_idx, bird_idx])
    FP = int(cm[:, bird_idx].sum() - TP)
    FN = int(cm[bird_idx, :].sum() - TP)
    TN = int(cm.sum() - (TP + FP + FN))

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    report_txt = classification_report(y_true, y_pred, target_names=classes, digits=4)

    metrics = {
        "device": device,
        "classes": classes,
        "num_samples": len(y_true),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision_bird": float(precision),
        "recall_bird": float(recall),
        "f1_bird": float(f1),
        "accuracy_overall": float(acc),
        "weights": str(resnet_weights),
        "test_dir": str(crop_test_dir),
        "classification_report": report_txt,
    }
    write_json(out_dir / "resnet_metrics.json", metrics)

    # Optional: ResNet training curves if user saved resnet_history.csv
    hist_csv_candidates = [
        crop_test_dir.parent / "resnet_history.csv",
        Path("classifier/resnet_history.csv"),
        Path("classifier/weights/resnet_history.csv"),
    ]
    hist_csv = next((p for p in hist_csv_candidates if p.exists()), None)
    if hist_csv:
        dfh = pd.read_csv(hist_csv)
        cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
        plot_df_curves(dfh, cols, "ResNet Training Curves", out_dir / "resnet_curves.png")

    return metrics


# ---------------------------
# Combined evaluation (YOLO + ResNet) on YOLO test split
# ---------------------------
def combined_report(
    yolo_weights: Path,
    yolo_dataset_dir: Path,
    resnet_weights: Path,
    out_dir: Path,
    conf_thres=0.25,
    iou_thres=0.50,
):
    """
    Assumes YOLO dataset structure:
      yolo_dataset_dir/
        test/images/*.jpg
        test/labels/*.txt
    (same as Roboflow export)
    """
    test_images_dir = yolo_dataset_dir / "test" / "images"
    test_labels_dir = yolo_dataset_dir / "test" / "labels"

    if not test_images_dir.exists() or not test_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected test/images and test/labels inside: {yolo_dataset_dir}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf_model = load_resnet50_binary(resnet_weights, device=device)
    yolo = YOLO(str(yolo_weights))

    img_paths = sorted([p for p in test_images_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    if not img_paths:
        raise RuntimeError(f"No images found in: {test_images_dir}")

    total_gt_bird = 0
    total_pred_bird_yolo = 0
    total_pred_bird_kept = 0
    removed_by_clf = 0

    TP = 0
    FP = 0
    FN = 0

    t0 = time.time()

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load GT boxes (bird class assumed 0 in your dataset)
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_bird_boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = yolo_txt_to_xyxy(line, w, h)
                if parsed is None:
                    continue
                cls, box = parsed
                # IMPORTANT: bird=0 (as per your names ['bird','no-bird'])
                if cls == 0:
                    gt_bird_boxes.append(box)

        total_gt_bird += len(gt_bird_boxes)

        # YOLO predictions
        res = yolo(img, conf=conf_thres, verbose=False)[0]
        pred_bird_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, c, s in zip(xyxy, cls, confs):
                cname = yolo.names.get(int(c), str(int(c)))
                if cname != "bird":
                    continue
                total_pred_bird_yolo += 1

                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                pred_idx, pred_conf = resnet_predict(clf_model, crop, device=device, image_size=224)

                # ResNet classes: assume 0=bird, 1=no-bird (as per your mapping)
                if pred_idx == 0:
                    pred_bird_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                    total_pred_bird_kept += 1
                else:
                    removed_by_clf += 1

        # Match kept preds to GT by IoU (one-to-one)
        matched_gt = set()
        for pb in pred_bird_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_bird_boxes):
                if j in matched_gt:
                    continue
                iou = iou_xyxy(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thres and best_j >= 0:
                TP += 1
                matched_gt.add(best_j)
            else:
                FP += 1

        FN += (len(gt_bird_boxes) - len(matched_gt))

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    runtime = time.time() - t0
    summary = {
        "dataset_test_images": str(test_images_dir),
        "yolo_weights": str(yolo_weights),
        "resnet_weights": str(resnet_weights),
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "total_gt_bird_boxes": int(total_gt_bird),
        "total_pred_bird_yolo": int(total_pred_bird_yolo),
        "total_pred_bird_kept_after_clf": int(total_pred_bird_kept),
        "removed_by_classifier": int(removed_by_clf),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "runtime_sec": float(runtime),
    }
    write_json(out_dir / "combined_summary.json", summary)
    return summary


# ---------------------------
# PDF Generator
# ---------------------------
def build_pdf(out_dir: Path, yolo_summary: dict, resnet_metrics: dict, combined_summary: dict):
    pdf_path = out_dir / "report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    def title(text, y):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, y, text)

    def para(lines, y, font="Helvetica", size=10, line_h=12):
        c.setFont(font, size)
        for ln in lines:
            c.drawString(2 * cm, y, ln)
            y -= line_h
        return y

    y = H - 2 * cm
    title("Bird Detection Project - Final Model Report", y)
    y -= 1.0 * cm

    y = para([
        "This report was generated automatically from your trained models and datasets.",
        "Sections included: YOLO Solo, ResNet Solo, Combined YOLO+ResNet Evaluation.",
    ], y)

    # --- YOLO SOLO
    y -= 0.6 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "1) YOLOv8 SOLO (Detector)")
    y -= 0.6 * cm

    yolo_lines = ["Key metrics (from last row of results.csv):"]
    for k, v in yolo_summary.items():
        yolo_lines.append(f"- {k}: {v}")
    y = para(yolo_lines, y)

    yolo_png = out_dir / "yolo_curves.png"
    if yolo_png.exists():
        y -= 0.3 * cm
        c.drawImage(str(yolo_png), 2 * cm, y - 9 * cm, width=W - 4 * cm, height=9 * cm, preserveAspectRatio=True, anchor='sw')
        y -= 9.4 * cm

    c.showPage()

    # --- RESNET SOLO
    y = H - 2 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "2) ResNet-50 SOLO (Patch Classifier)")
    y -= 0.7 * cm

    rn_lines = [
        f"Accuracy (overall): {resnet_metrics['accuracy_overall']:.4f}",
        f"Precision (bird): {resnet_metrics['precision_bird']:.4f}",
        f"Recall (bird): {resnet_metrics['recall_bird']:.4f}",
        f"F1 (bird): {resnet_metrics['f1_bird']:.4f}",
        f"TP={resnet_metrics['TP']}  TN={resnet_metrics['TN']}  FP={resnet_metrics['FP']}  FN={resnet_metrics['FN']}",
        f"Test samples: {resnet_metrics['num_samples']}",
        f"Weights: {resnet_metrics['weights']}",
    ]
    y = para(rn_lines, y)

    cm_png = out_dir / "resnet_confusion_matrix.png"
    if cm_png.exists():
        y -= 0.3 * cm
        c.drawImage(str(cm_png), 2 * cm, y - 8 * cm, width=10 * cm, height=8 * cm, preserveAspectRatio=True, anchor='sw')

    curves_png = out_dir / "resnet_curves.png"
    if curves_png.exists():
        c.drawImage(str(curves_png), 13 * cm, y - 8 * cm, width=6.5 * cm, height=8 * cm, preserveAspectRatio=True, anchor='sw')

    c.showPage()

    # --- COMBINED
    y = H - 2 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "3) Combined Pipeline (YOLO + ResNet)")
    y -= 0.7 * cm

    cb_lines = [
        "Evaluation done on YOLO test split (IoU=0.50).",
        f"YOLO bird detections (raw): {combined_summary['total_pred_bird_yolo']}",
        f"Kept after ResNet filter: {combined_summary['total_pred_bird_kept_after_clf']}",
        f"Removed by classifier: {combined_summary['removed_by_classifier']}",
        f"GT bird boxes: {combined_summary['total_gt_bird_boxes']}",
        f"TP={combined_summary['TP']}  FP={combined_summary['FP']}  FN={combined_summary['FN']}",
        f"Precision={combined_summary['precision']:.4f}  Recall={combined_summary['recall']:.4f}  F1={combined_summary['f1']:.4f}",
        f"Runtime: {combined_summary['runtime_sec']:.2f} sec",
    ]
    y = para(cb_lines, y)

    c.setFont("Helvetica", 9)
    y -= 0.5 * cm
    c.drawString(2 * cm, y, "Note: Combined metrics can change based on conf threshold and IoU threshold.")

    c.save()
    return pdf_path


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-run", required=True, help="YOLO run directory that contains results.csv (e.g. runs_yolo/yolov8m_bird_no_bird_v22)")
    ap.add_argument("--yolo-dataset", required=True, help="YOLO dataset root (e.g. data/yolo_dataset-2)")
    ap.add_argument("--resnet-weights", required=True, help="ResNet weights path (.pth)")
    ap.add_argument("--crop-test", required=True, help="ResNet crop test folder (e.g. classifier/crop_dataset-2/test)")
    ap.add_argument("--yolo-weights", default="", help="Optional: override YOLO best.pt path. If blank, uses yolo-run/weights/best.pt")
    ap.add_argument("--outdir", default="reports/final_report", help="Output folder")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold for combined eval")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for combined eval")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    yolo_run_dir = Path(args.yolo_run)
    yolo_weights = Path(args.yolo_weights) if args.yolo_weights else (yolo_run_dir / "weights" / "best.pt")
    yolo_dataset_dir = Path(args.yolo_dataset)
    resnet_weights = Path(args.resnet_weights)
    crop_test_dir = Path(args.crop_test)

    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    if not resnet_weights.exists():
        raise FileNotFoundError(f"ResNet weights not found: {resnet_weights}")

    print("[1/3] YOLO solo report...")
    yolo_summary = yolo_solo_report(yolo_run_dir, out_dir)
    print("     YOLO summary keys:", list(yolo_summary.keys())[:6], "...")

    print("[2/3] ResNet solo report...")
    resnet_metrics = resnet_solo_report(crop_test_dir, resnet_weights, out_dir)
    print(f"     ResNet accuracy: {resnet_metrics['accuracy_overall']:.4f}")

    print("[3/3] Combined report...")
    combined_summary = combined_report(
        yolo_weights=yolo_weights,
        yolo_dataset_dir=yolo_dataset_dir,
        resnet_weights=resnet_weights,
        out_dir=out_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
    print(f"     Combined precision={combined_summary['precision']:.4f} recall={combined_summary['recall']:.4f}")

    print("[PDF] Building report...")
    pdf_path = build_pdf(out_dir, yolo_summary, resnet_metrics, combined_summary)
    print("\n✅ DONE")
    print("PDF:", pdf_path.resolve())
    print("Folder:", out_dir.resolve())


if __name__ == "__main__":
    main()
