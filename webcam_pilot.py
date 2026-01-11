# webcam_pilot.py
import cv2
import time
import math
from collections import deque
from ultralytics import YOLO

# ===================== CONFIG =====================
# Models
YOLO_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\runs_yolo\yolov8m_bird_no_bird_v2\weights\best.pt"
RESNET_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\classifier\weights\resnet50_best_v2.pth"
USE_RESNET = True

# Camera (0=webcam). For phone IP cam use: "http://IP:8080/video"
CAM_SOURCE = 0

# YOLO inference
YOLO_CONF = 0.10          # candidate listing threshold
YOLO_IOU = 0.50           # NMS IoU
MAX_DETS = 50

# Box geometry filters (reduce face/cat false positives)
MIN_BOX_AREA = 0.001      # allow small birds
MAX_BOX_AREA = 0.22       # reject huge boxes (faces/people/large objects)
MIN_BOX_W = 12            # reject tiny noise
MIN_BOX_H = 12
ASPECT_MIN = 0.20         # w/h
ASPECT_MAX = 4.50

# No-bird competition suppression
# If "no-bird" confidence is close/greater than bird, suppress bird strongly.
NO_BIRD_MARGIN = 0.08
NO_BIRD_SUPPRESS_MULT = 0.40

# ResNet (soft influence)
RESNET_EVERY_N_FRAMES = 2       # reduce lag; run ResNet only every N frames
RESNET_SOFT_TH = 0.55
RESNET_BOOST_TH = 0.78
RESNET_PENALTY_MULT = 0.45
RESNET_BOOST_MULT = 1.15

# Temporal stabilizer (best of both scripts)
WINDOW_SIZE = 12                 # frames memory
POS_TH = 0.38                    # score threshold considered "positive"
NEG_TH = 0.22                    # score threshold considered "negative"
MIN_POS_FRAMES_ON = 4            # positives inside window needed to turn ON
MIN_NEG_FRAMES_OFF = 6           # negatives inside window needed to turn OFF
EMA_ALPHA = 0.35                 # smoothing for score

# Debug UI
SHOW_DEBUG = True
SHOW_TOPK = 7
# ==================================================


def box_area_fraction(x1, y1, x2, y2, w, h):
    return max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def norm_name(name: str) -> str:
    return (name or "").lower().strip()


def is_bird_class(name: str) -> bool:
    s = norm_name(name)
    if "bird" not in s:
        return False
    if "no-bird" in s or "no_bird" in s or "nobird" in s or s == "no bird":
        return False
    if s.startswith("no "):
        return False
    return True


def is_no_bird_class(name: str) -> bool:
    s = norm_name(name)
    return ("no-bird" in s) or ("no_bird" in s) or ("nobird" in s) or (s == "no bird")


def passes_geom_filters(x1, y1, x2, y2, w, h) -> bool:
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw < MIN_BOX_W or bh < MIN_BOX_H:
        return False
    area_fr = (bw * bh) / float(w * h)
    if area_fr < MIN_BOX_AREA or area_fr > MAX_BOX_AREA:
        return False
    ar = bw / float(bh + 1e-6)
    if ar < ASPECT_MIN or ar > ASPECT_MAX:
        return False
    return True


def main():
    print("[INFO] Loading YOLO:", YOLO_WEIGHTS)
    yolo = YOLO(YOLO_WEIGHTS)

    clf = None
    if USE_RESNET:
        print("[INFO] Loading ResNet:", RESNET_WEIGHTS)
        from classifier.inference_resnet_utils import ResNetBirdClassifier
        clf = ResNetBirdClassifier(weights_path=RESNET_WEIGHTS)

    cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible. Try CAM_SOURCE=1/2 or an IP camera URL.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Temporal buffers
    score_buf = deque(maxlen=WINDOW_SIZE)
    pos_buf = deque(maxlen=WINDOW_SIZE)
    neg_buf = deque(maxlen=WINDOW_SIZE)

    bird_state = 0
    ema_score = 0.0
    frame_idx = 0
    last_print = 0.0

    print("[INFO] Running... Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_idx += 1
        h, w = frame.shape[:2]

        # -------- YOLO predict --------
        results = yolo.predict(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=MAX_DETS,
            verbose=False
        )

        best_bird = None          # (x1,y1,x2,y2)
        best_bird_conf = 0.0
        best_nobird_conf = 0.0
        debug_lines = []

        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                # Collect debug labels (top conf, regardless of class)
                # and compute best bird / best no-bird
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    cls_name = r.names.get(cls_id, str(cls_id))

                    if SHOW_DEBUG:
                        debug_lines.append((conf, f"{cls_name}:{conf:.2f}"))

                    if is_no_bird_class(cls_name):
                        if conf > best_nobird_conf:
                            best_nobird_conf = conf
                        continue

                    if is_bird_class(cls_name):
                        x1, y1, x2, y2 = map(int, b.xyxy[0])

                        if not passes_geom_filters(x1, y1, x2, y2, w, h):
                            continue

                        if conf > best_bird_conf:
                            best_bird_conf = conf
                            best_bird = (x1, y1, x2, y2)

        debug_lines = [t for _, t in sorted(debug_lines, reverse=True)][:SHOW_TOPK]

        # -------- Composite score (stable + less random) --------
        score = best_bird_conf
        resnet_score = None

        # 1) suppress if "no-bird" is competing (prevents flicker on humans/cats)
        if score > 0 and best_nobird_conf > 0:
            if score < best_nobird_conf + NO_BIRD_MARGIN:
                score *= NO_BIRD_SUPPRESS_MULT

        # 2) ResNet soft influence (run every N frames to reduce lag)
        if best_bird is not None and USE_RESNET and clf is not None and (frame_idx % RESNET_EVERY_N_FRAMES == 0):
            x1, y1, x2, y2 = best_bird
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                raw = clf.predict_raw(crop_rgb)

                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                resnet_score = safe_float(raw, 0.0)

                # If your ResNet returns logits, uncomment sigmoid:
                # resnet_score = 1.0 / (1.0 + math.exp(-resnet_score))

                if resnet_score < RESNET_SOFT_TH:
                    score *= RESNET_PENALTY_MULT
                elif resnet_score > RESNET_BOOST_TH:
                    score *= RESNET_BOOST_MULT

        # -------- Temporal smoothing (window + EMA + hysteresis) --------
        ema_score = (EMA_ALPHA * score) + ((1.0 - EMA_ALPHA) * ema_score)

        is_pos = 1 if score >= POS_TH else 0
        is_neg = 1 if score <= NEG_TH else 0

        score_buf.append(score)
        pos_buf.append(is_pos)
        neg_buf.append(is_neg)

        pos_cnt = sum(pos_buf)
        neg_cnt = sum(neg_buf)

        if bird_state == 0:
            # turn ON only when enough positives, and EMA supports it
            if pos_cnt >= MIN_POS_FRAMES_ON and ema_score >= POS_TH:
                bird_state = 1
                # clear neg history to avoid instant OFF
                neg_buf.clear()
        else:
            # turn OFF when enough negatives AND EMA dropped
            if neg_cnt >= MIN_NEG_FRAMES_OFF and ema_score <= NEG_TH:
                bird_state = 0
                pos_buf.clear()

        # -------- Print throttle --------
        now = time.time()
        if now - last_print > 0.25:
            print(f"BIRD={bird_state}  score={score:.2f}  ema={ema_score:.2f}  yolo_bird={best_bird_conf:.2f}  yolo_nobird={best_nobird_conf:.2f}")
            last_print = now

        # -------- Draw UI --------
        cv2.putText(
            frame, f"BIRD={bird_state}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.25,
            (0, 255, 0) if bird_state else (0, 0, 255),
            3
        )

        if SHOW_DEBUG:
            y = 85
            for line in debug_lines:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                y += 26

            cv2.putText(frame, f"score:{score:.2f} ema:{ema_score:.2f} pos:{pos_cnt}/{len(pos_buf)} neg:{neg_cnt}/{len(neg_buf)}",
                        (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if best_bird is not None:
            x1, y1, x2, y2 = best_bird
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            label = f"bird_yolo={best_bird_conf:.2f} nobird={best_nobird_conf:.2f} score={score:.2f}"
            if resnet_score is not None:
                label += f" r={resnet_score:.2f}"

            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        cv2.imshow("Bird / No-Bird Pilot (Stable)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
