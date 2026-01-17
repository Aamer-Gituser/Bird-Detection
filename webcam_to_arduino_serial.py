# webcam_to_arduino_serial_fixed_v2.py
# ✅ Changes you asked:
# 1) Removed center-block gate + removed drawing of center block rectangle
# 2) CONFIRM time = 1.3 sec
# 3) COOLDOWN time = 2.0 sec
# 4) Keeps: box drawing + terminal logs + Arduino pulse "1\n" then "0\n"
# 5) Auto-detects Arduino COM port (still supports --port COMx)

import cv2
import time
import argparse
from collections import deque
from ultralytics import YOLO

import serial
from serial.tools import list_ports
from serial.serialutil import SerialException

# ===================== CONFIG =====================
YOLO_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\runs_yolo\yolov8m_bird_no_bird_v2\weights\best.pt"
RESNET_WEIGHTS = r"D:\Projects\Bird Detection\bird_detection_project\classifier\weights\resnet50_best_v2.pth"
USE_RESNET = True

CAM_SOURCE = 0  # laptop webcam default

SERIAL_ENABLED = True
SERIAL_BAUD = 9600

# ✅ confirm + cooldown (updated)
CONFIRM_SEC = 1.3
COOLDOWN_SEC = 2.0

# Arduino pulse
PULSE_WIDTH_SEC = 0.10

# YOLO
YOLO_CONF = 0.18
YOLO_IOU = 0.50
MAX_DETS = 30

# Geometry filters (strict to kill false positives)
MIN_BOX_AREA = 0.001
MAX_BOX_AREA = 0.10
MIN_BOX_W = 18
MIN_BOX_H = 18
ASPECT_MIN = 0.25
ASPECT_MAX = 3.20

# Motion gate
MOTION_ENABLED = True
MOTION_DIFF_TH = 18
MOTION_MIN_FRAC = 0.003

# No-bird competition suppression
NO_BIRD_MARGIN = 0.10
NO_BIRD_SUPPRESS_MULT = 0.35

# ResNet influence (optional)
RESNET_EVERY_N_FRAMES = 2
RESNET_SOFT_TH = 0.60
RESNET_BOOST_TH = 0.80
RESNET_PENALTY_MULT = 0.35
RESNET_BOOST_MULT = 1.10

# Temporal stabilizer
WINDOW_SIZE = 12
POS_TH = 0.55
NEG_TH = 0.25
MIN_POS_FRAMES_ON = 6
MIN_NEG_FRAMES_OFF = 6
EMA_ALPHA = 0.35

SHOW_DEBUG = True
SHOW_TOPK = 6
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


def list_serial_ports():
    ports = list_ports.comports()
    out = []
    for p in ports:
        out.append({
            "device": p.device,
            "description": (p.description or ""),
            "hwid": (p.hwid or ""),
            "manufacturer": (p.manufacturer or ""),
        })
    return out


def score_port(p):
    text = f"{p['device']} {p['description']} {p['manufacturer']} {p['hwid']}".lower()
    keywords = [
        ("arduino", 100),
        ("ch340", 90),
        ("cp210", 90),
        ("silicon labs", 80),
        ("ftdi", 80),
        ("usb serial", 70),
        ("serial", 30),
    ]
    score = 0
    for k, w in keywords:
        if k in text:
            score += w
    try:
        if p["device"].upper().startswith("COM"):
            n = int(p["device"][3:])
            score += max(0, 20 - min(n, 20))
    except Exception:
        pass
    return score


def auto_detect_serial_port(preferred_port=None):
    if preferred_port:
        return preferred_port
    ports = list_serial_ports()
    if not ports:
        return None
    ports_sorted = sorted(ports, key=score_port, reverse=True)
    return ports_sorted[0]["device"]


class SerialOut:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.open()

    def open(self):
        self.close()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2.0)  # Arduino reset

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def send_line(self, s: str):
        if self.ser is None:
            return
        try:
            self.ser.write(s.encode("utf-8"))
        except SerialException as e:
            print(f"[WARN] Serial write failed: {e}. Reconnecting...")
            try:
                time.sleep(0.5)
                self.open()
                self.ser.write(s.encode("utf-8"))
            except Exception as e2:
                print(f"[WARN] Serial reconnect failed: {e2}. Disabling serial.")
                self.close()

    def pulse_one(self, width_sec: float):
        self.send_line("1\n")
        time.sleep(width_sec)
        self.send_line("0\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=str, default=str(CAM_SOURCE),
                    help="Camera source: 0/1/2 or RTSP/URL path")
    ap.add_argument("--port", type=str, default="",
                    help="Force serial port, e.g. COM5. If empty, auto-detect.")
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD,
                    help="Serial baud rate (default 9600).")
    ap.add_argument("--no-serial", action="store_true",
                    help="Disable serial output entirely.")
    return ap.parse_args()


def main():
    args = parse_args()

    cam_src = args.cam
    if isinstance(cam_src, str) and cam_src.isdigit():
        cam_src = int(cam_src)

    print("[INFO] Loading YOLO:", YOLO_WEIGHTS)
    yolo = YOLO(YOLO_WEIGHTS)

    clf = None
    if USE_RESNET:
        print("[INFO] Loading ResNet:", RESNET_WEIGHTS)
        from classifier.inference_resnet_utils import ResNetBirdClassifier
        clf = ResNetBirdClassifier(weights_path=RESNET_WEIGHTS)

    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        raise RuntimeError("❌ Camera not accessible. Try: --cam 0 (laptop), --cam 1 (USB).")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Serial init
    ser_out = None
    if SERIAL_ENABLED and (not args.no_serial):
        preferred = args.port.strip() if args.port else None
        port = auto_detect_serial_port(preferred_port=preferred)
        if not port:
            print("[WARN] No serial ports detected. Running WITHOUT Arduino output.")
        else:
            print(f"[INFO] Using Serial: {port} @ {args.baud}")
            try:
                ser_out = SerialOut(port, args.baud)
                ser_out.send_line("0\n")
            except Exception as e:
                print(f"[WARN] Could not open serial port {port}: {e}")
                ser_out = None

    # Stabilizers
    pos_buf = deque(maxlen=WINDOW_SIZE)
    neg_buf = deque(maxlen=WINDOW_SIZE)

    stable_bird = 0
    ema_score = 0.0
    frame_idx = 0

    confirm_started_at = None
    cooldown_until = 0.0
    last_print = 0.0

    # Motion
    prev_gray = None

    print("[INFO] Running... Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_idx += 1
        h, w = frame.shape[:2]
        now = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # ----- score -----
        score = best_bird_conf
        resnet_score = None

        # suppress if no-bird competes
        if score > 0 and best_nobird_conf > 0 and score < best_nobird_conf + NO_BIRD_MARGIN:
            score *= NO_BIRD_SUPPRESS_MULT

        # motion gate (reduce false positives)
        motion_ok = True
        if MOTION_ENABLED and best_bird is not None and prev_gray is not None:
            x1, y1, x2, y2 = best_bird
            roi_now = gray[y1:y2, x1:x2]
            roi_prev = prev_gray[y1:y2, x1:x2]
            if roi_now.size > 0 and roi_prev.size > 0:
                diff = cv2.absdiff(roi_now, roi_prev)
                changed = (diff > MOTION_DIFF_TH).mean()  # fraction changed
                motion_ok = changed >= MOTION_MIN_FRAC
            else:
                motion_ok = False

        if best_bird is not None and not motion_ok:
            score *= 0.25

        # ResNet soft influence
        if best_bird is not None and USE_RESNET and clf is not None and (frame_idx % RESNET_EVERY_N_FRAMES == 0):
            x1, y1, x2, y2 = best_bird
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                raw = clf.predict_raw(crop_rgb)
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                resnet_score = safe_float(raw, 0.0)

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

        # stable decision
        if stable_bird == 0:
            if pos_cnt >= MIN_POS_FRAMES_ON and ema_score >= POS_TH:
                stable_bird = 1
                neg_buf.clear()
        else:
            if neg_cnt >= MIN_NEG_FRAMES_OFF and ema_score <= NEG_TH:
                stable_bird = 0
                pos_buf.clear()

        # confirm + cooldown
        in_cooldown = now < cooldown_until
        triggered = False

        if in_cooldown:
            confirm_started_at = None
        else:
            if stable_bird == 1:
                if confirm_started_at is None:
                    confirm_started_at = now
                elif (now - confirm_started_at) >= CONFIRM_SEC:
                    triggered = True
                    cooldown_until = now + COOLDOWN_SEC
                    confirm_started_at = None
            else:
                confirm_started_at = None

        if ser_out is not None and triggered:
            ser_out.pulse_one(PULSE_WIDTH_SEC)

        # logs
        if now - last_print > 0.25:
            conf_t = 0.0 if confirm_started_at is None else (now - confirm_started_at)
            cool_left = max(0.0, cooldown_until - now)
            print(
                f"stable={stable_bird} score={score:.2f} ema={ema_score:.2f} "
                f"yolo_bird={best_bird_conf:.2f} nobird={best_nobird_conf:.2f} "
                f"motion={'OK' if motion_ok else 'NO'} "
                f"confirm={conf_t:.1f}/{CONFIRM_SEC:.1f}s cooldown_left={cool_left:.1f}s "
                f"{'TRIGGERED!' if triggered else ''}"
            )
            last_print = now

        # UI
        cool_left = max(0.0, cooldown_until - now)
        conf_t = 0.0 if confirm_started_at is None else (now - confirm_started_at)

        cv2.putText(frame, f"STABLE_BIRD={stable_bird}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if stable_bird else (0, 0, 255), 3)

        if cool_left > 0:
            cv2.putText(frame, f"COOLDOWN: {cool_left:.1f}s", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif stable_bird == 1:
            cv2.putText(frame, f"CONFIRMING: {conf_t:.1f}/{CONFIRM_SEC:.1f}s", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"MOTION={'OK' if motion_ok else 'NO'}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if best_bird is not None:
            x1, y1, x2, y2 = best_bird
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"bird={best_bird_conf:.2f} nobird={best_nobird_conf:.2f} score={score:.2f}"
            if resnet_score is not None:
                label += f" r={resnet_score:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if SHOW_DEBUG:
            y = 155
            for line in debug_lines:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                y += 22

        cv2.imshow("Bird/No-Bird -> Arduino (1.3s Confirm + 2.0s Cooldown)", frame)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if ser_out is not None:
        try:
            ser_out.send_line("0\n")
            ser_out.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
