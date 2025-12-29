# classifier/prepare_crops.py
"""
Create a cropped patch dataset (bird / no-bird) from the YOLOv8 dataset.

Input  (already present):
    data/yolo_dataset-2/
        train/images/*.jpg
        train/labels/*.txt
        valid/images/*.jpg
        valid/labels/*.txt
        test/images/*.jpg
        test/labels/*.txt

Output (this script will create):
    classifier/crop_dataset-2/
        train/bird/*.jpg
        train/no-bird/*.jpg
        val/bird/*.jpg
        val/no-bird/*.jpg
        test/bird/*.jpg
        test/no-bird/*.jpg
"""

import sys
from pathlib import Path
import argparse

import cv2
from tqdm import tqdm

# --- make project root importable so "from utils.config" works ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------

from utils.config import (
    DATA_DIR,
    CROP_TRAIN_DIR,
    CROP_VAL_DIR,
    CROP_TEST_DIR,
    CLASS_NAMES,
    CLF_IMAGE_SIZE,
    ensure_dirs,
)


def yolo_to_xyxy(box, img_w, img_h, margin_factor=1.0):
    """
    Convert YOLO-normalized bbox [cx, cy, w, h] to pixel [x1, y1, x2, y2].
    Optionally expand by a margin_factor (>1 increases box size).
    """
    cx, cy, w, h = box

    # convert normalized -> absolute
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    # apply margin
    w *= margin_factor
    h *= margin_factor

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    # clip to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def process_split(split_name: str, out_root: Path, margin_factor: float = 1.1):
    """
    Create crops for a given YOLO split (train / valid / test).

    split_name: "train", "valid", or "test"
    out_root: path like CROP_TRAIN_DIR / CROP_VAL_DIR / CROP_TEST_DIR
    """
    images_dir = DATA_DIR / split_name / "images"
    labels_dir = DATA_DIR / split_name / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images or labels for split '{split_name}'")

    # make class subfolders
    for cls_name in CLASS_NAMES:
        (out_root / cls_name).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*.*"))

    print(f"\n[INFO] Processing split '{split_name}' – {len(img_paths)} images")

    for img_path in tqdm(img_paths, desc=f"{split_name} images"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            # no labels → skip
            continue

        # read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # each line: <cls> <cx> <cy> <w> <h>
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue

            cx, cy, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, w, h), img_w, img_h, margin_factor)

            # ignore tiny boxes
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # resize to classifier input size
            crop_resized = cv2.resize(crop, (CLF_IMAGE_SIZE, CLF_IMAGE_SIZE))

            cls_name = CLASS_NAMES[cls_id]
            out_dir = out_root / cls_name

            out_filename = f"{stem}_{idx}_{cls_name}.jpg"
            out_path = out_dir / out_filename

            cv2.imwrite(str(out_path), crop_resized)


def main(margin_factor: float):
    ensure_dirs()

    # Map YOLO's "train/valid/test" -> classifier's crop dirs
    process_split("train", CROP_TRAIN_DIR, margin_factor)
    process_split("valid", CROP_VAL_DIR, margin_factor)
    process_split("test", CROP_TEST_DIR, margin_factor)

    print("\n[INFO] Crop dataset created successfully.")
    print("Train dir:", CROP_TRAIN_DIR)
    print("Val dir:", CROP_VAL_DIR)
    print("Test dir:", CROP_TEST_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create cropped bird/no-bird patches for ResNet-50 classifier."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.1,
        help="Scale bounding box by this factor before cropping (default: 1.1).",
    )
    args = parser.parse_args()
    main(args.margin)
