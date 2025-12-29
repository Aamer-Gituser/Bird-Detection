# inference_combined.py
"""
Run combined inference:
    YOLOv8 bird/no-bird detector  +  ResNet-50 patch classifier.

Flow:
    1. YOLO finds candidate bird boxes in each frame.
    2. For each 'bird' box from YOLO:
        - crop patch from the frame
        - classify with ResNet (bird / no-bird)
    3. Keep and draw only those boxes where ResNet also says 'bird'.
       All others are filtered out (no box shown → transparent).

Usage examples:
    # Webcam
    python inference_combined.py --source 0

    # Video file
    python inference_combined.py --source path/to/video.mp4

    # Single image
    python inference_combined.py --source path/to/image.jpg
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils.config import CLASS_NAMES, CLF_IMAGE_SIZE
from classifier.inference_resnet_utils import ResNetBirdClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined YOLOv8 + ResNet-50 bird/no-bird inference."
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="runs_yolo/yolov8m_bird_no_bird_v2/weights/best.pt",
        help="Path to YOLOv8 bird/no-bird weights.",
    )
    parser.add_argument(
        "--clf-weights",
        type=str,
        default="classifier/resnet50_best_v2.pth",
        help="Path to ResNet-50 classifier weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: 0 (webcam) or path to image/video file.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--show-clf-score",
        action="store_true",
        help="Show classifier confidence in label text.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video/image next to source.",
    )
    return parser.parse_args()


def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def is_video_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]


def main():
    args = parse_args()

    # 1) Load YOLO model
    print(f"[INFO] Loading YOLO model from: {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)

    # 2) Load ResNet classifier
    print(f"[INFO] Loading ResNet-50 classifier from: {args.clf_weights}")
    clf = ResNetBirdClassifier(
        weights_path=args.clf_weights,
        class_names=CLASS_NAMES,
        image_size=CLF_IMAGE_SIZE,
    )
    print("[INFO] Classifier ready with classes:", CLASS_NAMES)

    # Map YOLO class indices to names (should match CLASS_NAMES)
    # Assuming YOLO model trained with same CLASS_NAMES order.
    yolo_names = yolo_model.names
    print("[INFO] YOLO class names:", yolo_names)

    source = args.source

    # Handle webcam (source is "0", "1", etc.)
    if source.isdigit():
        source_int = int(source)
        print(f"[INFO] Using webcam source: {source_int}")
        cap = cv2.VideoCapture(source_int)
        out_writer = None
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Webcam stream ended.")
                break

            start_time = time.time()

            # YOLO inference on this frame
            results = yolo_model(frame, conf=args.conf_thres, verbose=False)[0]

            final_boxes = []
            final_scores = []

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy()

                for box, c, s in zip(xyxy, cls, conf):
                    cls_idx = int(c)
                    cls_name = yolo_names.get(cls_idx, str(cls_idx))

                    # We only apply classifier to YOLO "bird" detections
                    if cls_name != "bird":
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    # Safe clipping
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    # Classifier decision
                    clf_result = clf.predict(crop, return_proba=True)
                    clf_label = clf_result["label"]
                    clf_conf = clf_result["conf"]

                    # Keep only if classifier ALSO says "bird"
                    if clf_label == "bird":
                        final_boxes.append((x1, y1, x2, y2))
                        # Optionally combine YOLO & classifier score
                        combined_score = float(s) * clf_conf
                        final_scores.append(combined_score)

                        # Draw box
                        label_text = f"bird {combined_score*100:.1f}%"
                        if args.show_clf_score:
                            label_text = f"bird | y:{s*100:.1f}% c:{clf_conf*100:.1f}%"

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
                    # else:
                    #   YOLO said bird but classifier said no-bird
                    #   -> DO NOTHING (transparent / filtered)

            fps = 1.0 / (time.time() - start_time + 1e-8)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLOv8 + ResNet50 (Webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # IMAGE or VIDEO file
    if is_image_file(source):
        img = cv2.imread(source)
        if img is None:
            print(f"[ERROR] Could not read image: {source}")
            return

        results = yolo_model(img, conf=args.conf_thres, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()

            for box, c, s in zip(xyxy, cls, conf):
                cls_idx = int(c)
                cls_name = yolo_names.get(cls_idx, str(cls_idx))

                if cls_name != "bird":
                    continue

                x1, y1, x2, y2 = box.astype(int)
                h, w = img.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                clf_result = clf.predict(crop, return_proba=True)
                clf_label = clf_result["label"]
                clf_conf = clf_result["conf"]

                if clf_label == "bird":
                    label_text = f"bird { (s*clf_conf)*100:.1f}%"
                    if args.show_clf_score:
                        label_text = f"bird | y:{s*100:.1f}% c:{clf_conf*100:.1f}%"

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

        cv2.imshow("YOLOv8 + ResNet50 (Image)", img)
        if args.save:
            out_path = str(Path(source).with_name(Path(source).stem + "_combined.jpg"))
            cv2.imwrite(out_path, img)
            print("[INFO] Saved:", out_path)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    elif is_video_file(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {source}")
            return

        out_writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = str(Path(source).with_name(Path(source).stem + "_combined.mp4"))
            out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
            print("[INFO] Saving output video to:", out_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video ended.")
                break

            start_time = time.time()

            results = yolo_model(frame, conf=args.conf_thres, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy()

                for box, c, s in zip(xyxy, cls, conf):
                    cls_idx = int(c)
                    cls_name = yolo_names.get(cls_idx, str(cls_idx))

                    if cls_name != "bird":
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    clf_result = clf.predict(crop, return_proba=True)
                    clf_label = clf_result["label"]
                    clf_conf = clf_result["conf"]

                    if clf_label == "bird":
                        combined_score = float(s) * clf_conf
                        label_text = f"bird {combined_score*100:.1f}%"
                        if args.show_clf_score:
                            label_text = f"bird | y:{s*100:.1f}% c:{clf_conf*100:.1f}%"

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

            fps = 1.0 / (time.time() - start_time + 1e-8)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if out_writer is not None:
                out_writer.write(frame)

            cv2.imshow("YOLOv8 + ResNet50 (Video)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()
        return

    else:
        print(f"[ERROR] Source type not recognized: {source}")
        return


if __name__ == "__main__":
    main()
