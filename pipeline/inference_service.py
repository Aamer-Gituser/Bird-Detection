# pipeline/inference_service.py
"""
Service-level wrapper to run combined YOLOv8 + ResNet-50 inference.

Exposes:
    - IMAGE_EXTS, VIDEO_EXTS
    - run_combined_inference(input_path: str | Path) -> dict

The returned dict is designed for the Flask frontend:
    {
        "type": "image" | "video",
        "input_filename": "<name>",
        "output_filename": "<name or None>",
        "num_frames": int,
        "num_yolo_bird": int,
        "num_kept_after_clf": int,
        "runtime": float
    }
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from utils.config import CLASS_NAMES, CLF_IMAGE_SIZE
from classifier.inference_resnet_utils import ResNetBirdClassifier

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "test_images"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Update these if you change model versions
YOLO_WEIGHTS = BASE_DIR / "runs_yolo" / "yolov8m_bird_no_bird_v2" / "weights" / "best.pt"

# ✅ IMPORTANT:
# Your screenshot shows weights here:
# classifier/weights/resnet50_best.pth
# If your best v2 is outside weights folder, move/copy it into classifier/weights/
CLF_WEIGHTS = BASE_DIR / "classifier" / "weights" / "resnet50_best_v2.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# -------------------------------------------------------------------
# Load models once (module-level singletons)
# -------------------------------------------------------------------

print(f"[SERVICE] Loading YOLO from {YOLO_WEIGHTS}")
_yolo_model = YOLO(str(YOLO_WEIGHTS))

print(f"[SERVICE] Loading ResNet classifier from {CLF_WEIGHTS}")
_clf = ResNetBirdClassifier(
    weights_path=str(CLF_WEIGHTS),
    class_names=CLASS_NAMES,
    image_size=CLF_IMAGE_SIZE,
)

_yolo_names = _yolo_model.names  # {0:'bird',1:'no-bird'}
print("[SERVICE] YOLO class names:", _yolo_names)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def _safe_box(box, width: int, height: int):
    """Clamp xyxy box to image bounds."""
    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    return x1, y1, x2, y2


def _force_output_name(in_path: Path) -> str:
    """
    For videos -> always output MP4 (browser-friendly).
    For images -> keep same extension.
    """
    ext = in_path.suffix.lower()
    if ext in VIDEO_EXTS:
        return f"{in_path.stem}_combined.mp4"
    return f"{in_path.stem}_combined{ext}"


# -------------------------------------------------------------------
# Core API for Flask
# -------------------------------------------------------------------

def run_combined_inference(input_path) -> dict:
    """
    Run YOLOv8 + ResNet-50 pipeline on a single image or video.
    """
    t0 = time.time()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    out_name = _force_output_name(in_path)
    out_path = MEDIA_DIR / out_name

    total_yolo_bird = 0
    total_kept = 0
    frame_count = 0

    # ---------------- IMAGE ----------------
    if _is_image(in_path):
        media_type = "image"

        img = cv2.imread(str(in_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {in_path}")

        h, w = img.shape[:2]
        results = _yolo_model(img, conf=0.15, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()

            for box, c, s in zip(xyxy, cls, conf):
                cls_idx = int(c)
                cls_name = _yolo_names.get(cls_idx, str(cls_idx))

                # only check YOLO bird boxes
                if cls_name != "bird":
                    continue

                total_yolo_bird += 1

                x1, y1, x2, y2 = _safe_box(box, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                clf_res = _clf.predict(crop, return_proba=True)

                # keep only if classifier says bird
                if clf_res["label"] == "bird":
                    total_kept += 1
                    label_text = f"bird | y:{s*100:.1f}% c:{clf_res['conf']*100:.1f}%"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        label_text,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # ✅ FIX: imwrite needs (path, image)
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")

        frame_count = 1

    # ---------------- VIDEO ----------------
    elif _is_video(in_path):
        media_type = "video"

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {in_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in is None or fps_in <= 1e-6:
            fps_in = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video has invalid width/height (0).")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 codec

        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        if not out_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            results = _yolo_model(frame, conf=0.25, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy()

                for box, c, s in zip(xyxy, cls, conf):
                    cls_idx = int(c)
                    cls_name = _yolo_names.get(cls_idx, str(cls_idx))

                    if cls_name != "bird":
                        continue

                    total_yolo_bird += 1

                    x1, y1, x2, y2 = _safe_box(box, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    clf_res = _clf.predict(crop, return_proba=True)

                    if clf_res["label"] == "bird":
                        total_kept += 1
                        label_text = f"bird | y:{s*100:.1f}% c:{clf_res['conf']*100:.1f}%"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            label_text,
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

            out_writer.write(frame)

        cap.release()
        out_writer.release()

        if frame_count == 0:
            raise RuntimeError("No frames were read from the video (empty/corrupt video).")

    else:
        raise ValueError(f"Unsupported file type: {in_path.suffix}")

    runtime = time.time() - t0

    return {
        "type": media_type,
        "input_filename": in_path.name,
        "output_filename": out_path.name,
        "num_frames": int(frame_count),
        "num_yolo_bird": int(total_yolo_bird),
        "num_kept_after_clf": int(total_kept),
        "runtime": float(runtime),
    }
