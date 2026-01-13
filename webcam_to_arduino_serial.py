# webcam_to_arduino_serial.py
import cv2
import time
import math
from collections import deque
from ultralytics import YOLO
import serial

# ===================== CONFIG =====================
# Models
YOLO_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\runs_yolo\yolov8m_bird_no_bird_v2\weights\best.pt"
RESNET_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\classifier\weights\resnet50_best_v2.pth"
USE_RESNET = True

# Webcam source
CAM_SOURCE = 0  # 0=default webcam, try 1/2 if needed

# Arduino Serial
SERIAL_ENABLED = True
SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600

# Output mode
SEND_STATE_CONTINUOUS = True
STATE_SEND_INTERVAL_SEC = 0.25

SEND_EVENT_PULSE = True          # send 1 once on NEW detection (0->1)
PULSE_WIDTH_SEC = 0.15           # send 1 then 0 quickly

# Device cooldown (after trigger)
COOLDOWN_SEC = 5.0
FORCE_ZERO_DURING_COOLDOWN = True

# YOLO inference
YOLO_CONF = 0.10
YOLO_IOU = 0.50
MAX_DETS = 50

# Box geometry filters
MIN_BOX_AREA = 0.001
MAX_BOX_AREA = 0.22
MIN_BOX_W = 12
MIN_BOX_H = 12
ASPECT_MIN = 0.20
ASPECT_MAX = 4.50

# No-bird competition suppression
NO_BIRD_MARGIN = 0.08
NO_BIRD_SUPPRESS_MULT = 0.40

# ResNet (soft influence)
RESNET_EVERY_N_FRAMES = 2
RESNET_SOFT_TH = 0.55
RESNET_BOOST_TH = 0.78
RESNET_PENALTY_MULT = 0.45
RESNET_BOOST_MULT = 1.15

# Temporal stabilizer
WINDOW_SIZE = 12
POS_TH = 0.38
NEG_TH = 0.22
MIN_POS_FRAMES_ON = 4
MIN_NEG_FRAMES_OFF = 6
EMA_ALPHA = 0.35

# Debug UI
SHOW_DEBUG = True
SHOW_TOPK = 7
# ==================================================


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


class SerialOut:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)  # Arduino reset time
        self.last_sent = None
        self.last_state_send_ts = 0.0

    def send_state(self, state: int, force=False):
        if force or (self.last_sent != state):
            self.ser.write(f"{int(state)}\n".encode("utf-8"))
            self.last_sent = state

    def send_state_periodic(self, state: int, interval_sec: float):
        now = time.time()
        if now - self.last_state_send_ts >= interval_sec:
            self.send_state(state, force=True)
            self.last_state_send_ts = now

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


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
        raise RuntimeError("❌ Webcam not accessible. Try CAM_SOURCE=1/2.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ser_out = None
    if SERIAL_ENABLED:
        print(f"[INFO] Opening Serial: {SERIAL_PORT} @ {SERIAL_BAUD}")
        ser_out = SerialOut(SERIAL_PORT, SERIAL_BAUD)
        ser_out.send_state(0, force=True)

    pos_buf = deque(maxlen=WINDOW_SIZE)
    neg_buf = deque(maxlen=WINDOW_SIZE)

    bird_state = 0
    ema_score = 0.0
    frame_idx = 0

    cooldown_until = 0.0
    last_print = 0.0

    print("[INFO] Running... Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_idx += 1
        h, w = frame.shape[:2]

        results = yolo.predict(frame, conf=YOLO_CONF, iou=YOLO_IOU, max_det=MAX_DETS, verbose=False)

        best_bird = None
        best_bird_conf = 0.0
        best_nobird_conf = 0.0
        debug_lines = []

        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    cls_name = r.names.get(cls_id, str(cls_id))

                    if SHOW_DEBUG:
                        debug_lines.append((conf, f"{cls_name}:{conf:.2f}"))

                    if is_no_bird_class(cls_name):
                        best_nobird_conf = max(best_nobird_conf, conf)
                        continue

                    if is_bird_class(cls_name):
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        if not passes_geom_filters(x1, y1, x2, y2, w, h):
                            continue
                        if conf > best_bird_conf:
                            best_bird_conf = conf
                            best_bird = (x1, y1, x2, y2)

        debug_lines = [t for _, t in sorted(debug_lines, reverse=True)][:SHOW_TOPK]

        # -------- Composite score --------
        score = best_bird_conf
        resnet_score = None

        if score > 0 and best_nobird_conf > 0:
            if score < best_nobird_conf + NO_BIRD_MARGIN:
                score *= NO_BIRD_SUPPRESS_MULT

        if best_bird is not None and USE_RESNET and clf is not None and (frame_idx % RESNET_EVERY_N_FRAMES == 0):
            x1, y1, x2, y2 = best_bird
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                raw = clf.predict_raw(crop_rgb)
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                resnet_score = safe_float(raw, 0.0)

                # If logits:
                # resnet_score = 1.0 / (1.0 + math.exp(-resnet_score))

                if resnet_score < RESNET_SOFT_TH:
                    score *= RESNET_PENALTY_MULT
                elif resnet_score > RESNET_BOOST_TH:
                    score *= RESNET_BOOST_MULT

        ema_score = (EMA_ALPHA * score) + ((1.0 - EMA_ALPHA) * ema_score)

        is_pos = 1 if score >= POS_TH else 0
        is_neg = 1 if score <= NEG_TH else 0
        pos_buf.append(is_pos)
        neg_buf.append(is_neg)

        pos_cnt = sum(pos_buf)
        neg_cnt = sum(neg_buf)

        prev_state = bird_state

        now = time.time()
        in_cooldown = now < cooldown_until

        if not in_cooldown:
            if bird_state == 0:
                if pos_cnt >= MIN_POS_FRAMES_ON and ema_score >= POS_TH:
                    bird_state = 1
                    neg_buf.clear()
            else:
                if neg_cnt >= MIN_NEG_FRAMES_OFF and ema_score <= NEG_TH:
                    bird_state = 0
                    pos_buf.clear()
        else:
            if FORCE_ZERO_DURING_COOLDOWN:
                bird_state = 0

        if (prev_state == 0) and (bird_state == 1):
            cooldown_until = time.time() + COOLDOWN_SEC

        # -------- SERIAL OUTPUT --------
        if ser_out is not None:
            if SEND_EVENT_PULSE and (prev_state == 0) and (bird_state == 1):
                ser_out.send_state(1, force=True)
                time.sleep(PULSE_WIDTH_SEC)
                ser_out.send_state(0, force=True)

            if SEND_STATE_CONTINUOUS:
                ser_out.send_state_periodic(bird_state, STATE_SEND_INTERVAL_SEC)
            else:
                if prev_state != bird_state:
                    ser_out.send_state(bird_state, force=True)

        # -------- Print throttle --------
        if now - last_print > 0.25:
            print(
                f"BIRD={bird_state} score={score:.2f} ema={ema_score:.2f} "
                f"yolo_bird={best_bird_conf:.2f} yolo_nobird={best_nobird_conf:.2f} "
                f"cooldown={'YES' if in_cooldown else 'NO'}"
            )
            last_print = now

        # -------- UI --------
        cv2.putText(frame, f"BIRD={bird_state}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (0, 255, 0) if bird_state else (0, 0, 255), 3)

        if SHOW_DEBUG:
            y = 85
            for line in debug_lines:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                y += 26
            cv2.putText(frame, f"score:{score:.2f} ema:{ema_score:.2f} pos:{pos_cnt} neg:{neg_cnt}",
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

        cv2.imshow("Webcam Bird/No-Bird -> Arduino", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if ser_out is not None:
        ser_out.send_state(0, force=True)
        ser_out.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
