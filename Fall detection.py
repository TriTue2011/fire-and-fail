"""
V12.6_SmartSense — Multi-camera fall + fire/smoke detection
Tối ưu RTSP ổn định ở TARGET_FPS 10 hoặc 15 (mặc định 15).

Best practices (GitHub / OpenCV community):
- Intel Geti IPCameraStream: FFmpeg tcp + nobuffer + low_delay, grab/retrieve
- CAP_PROP_BUFFERSIZE=1, chỉ giữ frame mới nhất (tránh lag buffer)
- Reader thread tách inference → không back-pressure RTSP
- Inference throttle = 1/TARGET_FPS (CPU/GPU không quá tải)
- YOLO predict có lock khi multi-camera
"""
import cv2
import os
import time
import math
import threading
import numpy as np
from collections import deque
from datetime import datetime
import requests
from ultralytics import YOLO
import warnings
from flask import Flask, Response, render_template_string, request, send_from_directory
import json
import urllib.request

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== CONFIG =====================
CONFIG_FILE = "cameras.json"

# FPS xử lý mục tiêu: 10 hoặc 15 (cộng đồng khuyến nghị cho CPU + multi-cam RTSP)
# Đặt SMARTSENSE_TARGET_FPS=10 trong môi trường nếu máy yếu / nhiều camera.
TARGET_FPS = int(os.environ.get("SMARTSENSE_TARGET_FPS", "15"))
if TARGET_FPS not in (10, 15):
    print(f"[Config] TARGET_FPS={TARGET_FPS} không hợp lệ → dùng 15")
    TARGET_FPS = 15

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
    # Hỗ trợ format mới: {"target_fps": 15, "cameras": {...}} hoặc dict URL cũ
    if isinstance(_cfg, dict) and "cameras" in _cfg:
        CAMERAS = _cfg["cameras"]
        if "target_fps" in _cfg:
            TARGET_FPS = int(_cfg["target_fps"])
            if TARGET_FPS not in (10, 15):
                TARGET_FPS = 15
    else:
        CAMERAS = _cfg
else:
    CAMERAS = {
        "Cam1": "rtsp://admin:password@192.168.1.3:554/cam/realmonitor?channel=1&subtype=1"
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"target_fps": TARGET_FPS, "cameras": CAMERAS}, f, indent=4, ensure_ascii=False)

EVENT_DIR = "events"
os.makedirs(EVENT_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("SMARTSENSE_POSE_MODEL", "yolov8n-pose.pt")
CONF_THRES = float(os.environ.get("SMARTSENSE_CONF", "0.35"))
IMG_SIZE = int(os.environ.get("SMARTSENSE_IMG_SIZE", "320"))  # 320 = nhanh trên CPU
INFER_PERIOD = 1.0 / TARGET_FPS  # 15fps → ~0.067s | 10fps → 0.1s

FIRE_INFER_PERIOD = float(os.environ.get("SMARTSENSE_FIRE_PERIOD", "0.5"))  # 2 Hz
FIRE_COOLDOWN_SECONDS = 10
FIRE_CONF_THRES = 0.35

pose_model_lock = threading.Lock()
fire_model_lock = threading.Lock()

# Tải YOLO model báo cháy (open-source)
FIRE_MODEL_PATH = os.environ.get("SMARTSENSE_FIRE_MODEL", "fire_model.pt")
if not os.path.exists(FIRE_MODEL_PATH):
    print("🔥 Downloading Open-Source Fire YOLO model...")
    urllib.request.urlretrieve(
        "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt",
        FIRE_MODEL_PATH,
    )
fire_model = YOLO(FIRE_MODEL_PATH)

# Fall detection
TORSO_ANGLE_THRESHOLD = 45  # góc so với trục dọc (0=đứng, 90=nằm)
STATIONARY_SECONDS = 2.2
STATIONARY_MOVEMENT_PX = 35
FALL_CONFIRM_FRAMES = 2
COOLDOWN_SECONDS = 10
RECOVER_SECONDS = 2.0
ID_DISTANCE_PX = 80

# Motion prefilter
MOTION_THRESHOLD = 40
MOTION_MIN_AREA = 5000

# Telegram — ưu tiên biến môi trường (không hardcode token trên git)
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8490100526:AAFhLTElDAO20zsw-Fi3gIs58jR5sY9Cc-M",
)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7423399175")

# Live tile
TILE_W, TILE_H = 640, 360

# Event clip
PRE_EVENT_SECONDS = 5
POST_EVENT_SECONDS = 5
EVENT_DURATION = PRE_EVENT_SECONDS + POST_EVENT_SECONDS

# RTSP FFmpeg options (Intel Geti / OpenCV community)
# tcp: ổn định LAN; nobuffer+low_delay: giảm lag; stimeout: µs timeout
_RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;500000|"
    "reorder_queue_size;0|"
    "stimeout;5000000"
)

# ===================== LOAD YOLO =====================
print(f"🚀 Loading YOLO pose model... TARGET_FPS={TARGET_FPS}")
model = YOLO(MODEL_PATH)
device = "cuda" if model.device.type == "cuda" else "cpu"
print(f"✅ Pose device: {device} | Infer every {INFER_PERIOD*1000:.0f}ms")

# ===================== TELEGRAM =====================
def send_telegram_photo(image_path, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] ⚠️ Chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return
    try:
        with open(image_path, "rb") as photo:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=10,
            )
    except Exception as e:
        print(f"[Telegram] ❌ Failed to send photo: {e}")


# ===================== RTSP READER =====================
class ReaderThread(threading.Thread):
    """
    Drain RTSP liên tục, chỉ giữ frame mới nhất.
    Pattern: grab() + retrieve() (Geti / OpenCV) → consumer chậm không làm trễ stream.
    """

    MAX_RECONNECT_ATTEMPTS = 0  # 0 = vô hạn
    BACKOFF_BASE = 1.0
    BACKOFF_MAX = 30.0

    def __init__(self, name, url):
        super().__init__(daemon=True)
        self.name, self.url = name, url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.seq = 0
        self.last_ok = 0.0
        self.measured_fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.time()
        self.is_rtsp = str(url).lower().startswith("rtsp")

    def open_stream(self):
        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_FFMPEG_OPTIONS
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Một số backend hỗ trợ giới hạn FPS đọc (không phải lúc nào cũng có hiệu lực)
        try:
            cap.set(cv2.CAP_PROP_FPS, float(TARGET_FPS))
        except Exception:
            pass
        return cap

    def run(self):
        fail_streak = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                backoff = min(self.BACKOFF_BASE * (2 ** min(fail_streak, 4)), self.BACKOFF_MAX)
                print(f"[{self.name}] 🔌 Kết nối camera... (backoff {backoff:.0f}s)")
                self.cap = self.open_stream()
                time.sleep(0.5)
                if not self.cap.isOpened():
                    fail_streak += 1
                    print(f"[{self.name}] ❌ Không kết nối được, thử lại sau {backoff:.0f}s.")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(backoff)
                    continue
                fail_streak = 0
                print(f"[{self.name}] ✅ Đã kết nối RTSP.")

            # grab nhanh để xả buffer; retrieve chỉ khi grab OK
            grabbed = self.cap.grab()
            if not grabbed:
                fail_streak += 1
                if fail_streak >= 5:
                    print(f"[{self.name}] ⚠️ Mất tín hiệu → reconnect")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(min(self.BACKOFF_BASE * fail_streak, self.BACKOFF_MAX))
                else:
                    time.sleep(0.02 * fail_streak)
                continue

            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                fail_streak += 1
                continue

            fail_streak = 0
            with self.lock:
                self.frame = frame  # giữ ref mới nhất; detector sẽ copy khi cần
                self.seq += 1
                self.last_ok = time.time()

            self._fps_count += 1
            now = time.time()
            if now - self._fps_t0 >= 2.0:
                self.measured_fps = self._fps_count / (now - self._fps_t0)
                self._fps_count = 0
                self._fps_t0 = now

            # Không sleep 0 — luôn drain socket; CPU idle khi không có frame mới
            # (grab blocking trên RTSP)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_fps(self):
        """FPS dùng cho clip event: ưu tiên measured, rồi TARGET_FPS."""
        if self.measured_fps and self.measured_fps > 1:
            return max(5, min(30, int(round(self.measured_fps))))
        try:
            if self.cap:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 1:
                    return max(5, min(30, int(round(fps))))
        except Exception:
            pass
        return TARGET_FPS

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ===================== DETECTOR =====================
class DetectorThread(threading.Thread):
    def __init__(self, name, reader):
        super().__init__(daemon=True)
        self.name = name
        self.reader = reader
        self.motion_bg = None
        self.last_infer = 0.0
        # Buffer theo TARGET_FPS (không hardcode 30) → đúng PRE/POST giây, tiết kiệm RAM
        buf_len = max(int(EVENT_DURATION * TARGET_FPS) + TARGET_FPS, 30)
        self.buffer = deque(maxlen=buf_len)
        self.people = {}
        self.next_id = 1
        self.frame_vis = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        self.lock = threading.Lock()
        self.last_fire_infer = 0.0
        self.last_fire_alert = 0.0
        self.infer_fps = 0.0
        self._infer_count = 0
        self._infer_t0 = time.time()
        self._last_seq = -1

    def assign_id(self, centroid):
        for pid, pdata in self.people.items():
            if "centroid" in pdata:
                d = math.hypot(
                    centroid[0] - pdata["centroid"][0],
                    centroid[1] - pdata["centroid"][1],
                )
                if d < ID_DISTANCE_PX:
                    return pid
        pid = self.next_id
        self.next_id += 1
        return pid

    def draw_skeleton(self, frame, keypoints_xy, color=(0, 255, 0)):
        pairs = [
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
        ]
        for (x, y) in keypoints_xy:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        for a, b in pairs:
            if a < len(keypoints_xy) and b < len(keypoints_xy):
                p1, p2 = keypoints_xy[a], keypoints_xy[b]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

    @staticmethod
    def torso_angle_from_vertical(shoulder_mid, hip_mid):
        """
        Góc thân so với trục dọc (độ):
        0° = đứng thẳng, 90° = nằm ngang.
        Dùng atan2(|dx|, |dy|) — chuẩn fall-detection pose community.
        """
        dx = float(hip_mid[0] - shoulder_mid[0])
        dy = float(hip_mid[1] - shoulder_mid[1])
        # dy ~ 0 khi nằm ngang → góc ~ 90
        return abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    def run(self):
        while True:
            frame = self.reader.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Chỉ xử lý khi có frame mới (tránh spin cùng 1 frame)
            seq = self.reader.seq
            if seq == self._last_seq:
                time.sleep(0.005)
                continue
            self._last_seq = seq

            # motion prefilter
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if self.motion_bg is None:
                self.motion_bg = gray
                self._set_vis(frame, annotate_only=True)
                continue
            diff = cv2.absdiff(self.motion_bg, gray)
            _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            motion_area = np.sum(thresh) / 255.0
            self.motion_bg = cv2.addWeighted(self.motion_bg, 0.9, gray, 0.1, 0)
            self.buffer.append(frame.copy())

            now = time.time()
            if motion_area < MOTION_MIN_AREA:
                self._set_vis(frame, annotate_only=True)
                # Pace detector ~ TARGET_FPS khi idle
                time.sleep(max(0.0, INFER_PERIOD * 0.5))
                continue

            if now - self.last_infer < INFER_PERIOD:
                self._set_vis(frame, annotate_only=True)
                time.sleep(0.005)
                continue
            self.last_infer = now

            # YOLO pose (thread-safe)
            try:
                with pose_model_lock:
                    results = model.predict(
                        frame,
                        conf=CONF_THRES,
                        imgsz=IMG_SIZE,
                        verbose=False,
                        device=device,
                    )
            except Exception as e:
                print(f"[{self.name}] Model predict error: {e}")
                self._set_vis(frame, annotate_only=True)
                continue

            self._infer_count += 1
            if now - self._infer_t0 >= 2.0:
                self.infer_fps = self._infer_count / (now - self._infer_t0)
                self._infer_count = 0
                self._infer_t0 = now

            # Fire detection (tần suất thấp hơn pose)
            if now - self.last_fire_infer > FIRE_INFER_PERIOD:
                self.last_fire_infer = now
                try:
                    with fire_model_lock:
                        fire_results = fire_model.predict(
                            frame,
                            conf=FIRE_CONF_THRES,
                            imgsz=IMG_SIZE,
                            verbose=False,
                            device=device,
                        )
                    fire_detected = False
                    for box in fire_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label_name = fire_results[0].names[cls]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(
                            frame,
                            f"{label_name} {conf:.2f}",
                            (x1, max(10, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )
                        fire_detected = True

                    if fire_detected and now - self.last_fire_alert > FIRE_COOLDOWN_SECONDS:
                        self.last_fire_alert = now
                        print(f"[{self.name}] 🔥 FIRE/SMOKE DETECTED!")
                        event_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(EVENT_DIR, f"{self.name}_FIRE_{event_time}.jpg")
                        cv2.imwrite(img_path, frame)
                        send_telegram_photo(
                            img_path,
                            caption=f"🚨 FIRE/SMOKE in {self.name} at {event_time}",
                        )
                        threading.Thread(target=self._save_clip, args=("FIRE",), daemon=True).start()
                except Exception as e:
                    print(f"[{self.name}] Fire detection error: {e}")

            persons = []
            try:
                if results[0].keypoints is not None:
                    persons = results[0].keypoints.xy
            except Exception:
                persons = []

            now = time.time()
            seen_pids = set()
            for kp in persons:
                try:
                    kparr = kp.cpu().numpy()[:, :2] if hasattr(kp, "cpu") else np.asarray(kp)[:, :2]
                except Exception:
                    continue
                if kparr.shape[0] < 13:
                    continue
                shoulder_mid = np.mean(kparr[[5, 6]], axis=0)
                hip_mid = np.mean(kparr[[11, 12]], axis=0)
                angle = self.torso_angle_from_vertical(shoulder_mid, hip_mid)
                pid = self.assign_id(hip_mid)
                seen_pids.add(pid)

                if pid not in self.people:
                    self.people[pid] = {
                        "centroid": hip_mid,
                        "positions": deque(maxlen=max(60, TARGET_FPS * 5)),
                        "fall_frame_count": 0,
                        "candidate_start": None,
                        "was_fallen": False,
                        "last_alert": 0,
                        "last_seen": now,
                    }
                pdata = self.people[pid]
                pdata["centroid"] = hip_mid
                pdata["positions"].append((now, hip_mid))
                pdata["last_seen"] = now

                recent = [(t, p) for t, p in pdata["positions"] if now - t < STATIONARY_SECONDS]
                moved = sum(
                    math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    for (_, p1), (_, p2) in zip(recent, list(recent)[1:])
                )
                # angle > threshold = thân nghiêng/nằm + đứng yên
                fall_flag = angle > TORSO_ANGLE_THRESHOLD and moved < STATIONARY_MOVEMENT_PX

                if fall_flag:
                    pdata["fall_frame_count"] += 1
                else:
                    pdata["fall_frame_count"] = max(0, pdata["fall_frame_count"] - 1)

                if pdata["fall_frame_count"] >= FALL_CONFIRM_FRAMES and not pdata["was_fallen"]:
                    if pdata["candidate_start"] is None:
                        pdata["candidate_start"] = now
                    else:
                        if now - pdata["candidate_start"] >= STATIONARY_SECONDS:
                            confirm_recent = [
                                (t, p) for t, p in pdata["positions"] if now - t < STATIONARY_SECONDS
                            ]
                            moved_confirm = sum(
                                math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                                for (_, p1), (_, p2) in zip(confirm_recent, list(confirm_recent)[1:])
                            )
                            if moved_confirm < STATIONARY_MOVEMENT_PX:
                                pdata["was_fallen"] = True
                                pdata["fall_frame_count"] = 0
                                pdata["candidate_start"] = None
                                print(
                                    f"[{self.name}] CONFIRMED FALL -> ID {pid} "
                                    f"angle={angle:.1f}° moved={moved_confirm:.0f}px"
                                )
                                self._alert_and_save(frame.copy(), pid)
                            else:
                                pdata["candidate_start"] = None
                                pdata["fall_frame_count"] = 0

                if pdata["was_fallen"]:
                    rec_window = [
                        (t, p) for t, p in pdata["positions"] if now - t < RECOVER_SECONDS
                    ]
                    moved_rec = sum(
                        math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        for (_, p1), (_, p2) in zip(rec_window, list(rec_window)[1:])
                    )
                    if moved_rec > STATIONARY_MOVEMENT_PX:
                        pdata["was_fallen"] = False
                        pdata["candidate_start"] = None
                        pdata["fall_frame_count"] = 0
                        print(f"[{self.name}] RECOVERED ID {pid}")

                color = (0, 0, 255) if pdata["was_fallen"] else (0, 255, 0)
                try:
                    self.draw_skeleton(frame, kparr, color)
                except Exception:
                    pass
                cv2.putText(
                    frame,
                    f"ID {pid} [{'FALL' if pdata['was_fallen'] else 'OK'}] {angle:.0f}deg",
                    (int(hip_mid[0]) + 8, int(hip_mid[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                self.people[pid] = pdata

            stale = [pid for pid, pd in self.people.items() if now - pd.get("last_seen", now) > 6.0]
            for pid in stale:
                del self.people[pid]

            self._set_vis(frame)
            # Pace loop gần TARGET_FPS
            elapsed = time.time() - self.last_infer
            time.sleep(max(0.0, min(0.02, INFER_PERIOD - elapsed)))

    def _alert_and_save(self, frame_snapshot, pid):
        pdata = self.people.get(pid)
        if pdata is None:
            return
        now = time.time()
        if now - pdata.get("last_alert", 0) < COOLDOWN_SECONDS:
            return
        pdata["last_alert"] = now
        self.people[pid] = pdata

        event_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(EVENT_DIR, f"{self.name}_ID{pid}_{event_time}.jpg")
        cv2.imwrite(img_path, frame_snapshot)
        send_telegram_photo(img_path, caption=f"🚨 FALL {self.name} ID {pid} at {event_time}")
        threading.Thread(target=self._save_clip, args=(pid,), daemon=True).start()

    def _save_clip(self, pid):
        fps = TARGET_FPS  # clip playback = target process FPS (ổn định)
        pre_count = int(PRE_EVENT_SECONDS * fps)
        buf = list(self.buffer)
        pre_frames = buf[-pre_count:] if len(buf) >= pre_count else buf[:]
        if not pre_frames:
            return
        h, w = pre_frames[0].shape[:2]
        event_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_path = os.path.join(EVENT_DIR, f"{self.name}_ID{pid}_{event_time}.mp4")
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in pre_frames:
            out.write(f)
        t_end = time.time() + POST_EVENT_SECONDS
        interval = 1.0 / max(1, fps)
        while time.time() < t_end:
            f = self.reader.get_frame()
            if f is not None:
                out.write(f)
            time.sleep(interval)
        out.release()
        print(f"[{self.name}] 💾 Saved clip: {vid_path}")

    def _set_vis(self, frame, annotate_only=False):
        vis = cv2.resize(frame, (TILE_W, TILE_H))
        # HUD: FPS đọc / FPS suy luận
        r_fps = self.reader.measured_fps or 0
        i_fps = self.infer_fps or 0
        hud = f"{self.name} | RTSP {r_fps:.1f}fps | Infer {i_fps:.1f}/{TARGET_FPS}"
        cv2.putText(vis, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        with self.lock:
            self.frame_vis = vis


# ===================== LIVE VIEW (optional desktop) =====================
class LiveViewThread(threading.Thread):
    def __init__(self, detectors):
        super().__init__(daemon=True)
        self.detectors = detectors

    def run(self):
        print("[LiveView] Press ESC to quit")
        time.sleep(1.5)
        while True:
            frames = []
            for name, det in self.detectors.items():
                try:
                    frame = getattr(det, "frame_vis", None)
                except Exception:
                    frame = None
                if frame is None:
                    frame = np.zeros((TILE_H, TILE_W, 3), np.uint8)
                    cv2.putText(
                        frame,
                        f"{name} Loading...",
                        (50, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )
                frames.append(frame)
            if len(frames) >= 4:
                top = np.hstack(frames[0:2])
                bottom = np.hstack(frames[2:4])
                grid = np.vstack([top, bottom])
            else:
                grid = np.hstack(frames) if frames else np.zeros((TILE_H, TILE_W, 3), np.uint8)
            cv2.imshow("Multi-Live View SmartSense", grid)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                print("[LiveView] ESC -> exit")
                os._exit(0)
            time.sleep(0.03)


# ===================== WEB UI =====================
web_app = Flask(__name__)
global_detectors = {}
global_readers = {}


def generate_frames(cam_name):
    detector = global_detectors.get(cam_name)
    # Stream MJPEG ~ TARGET_FPS (không spam 20+ jpg/s)
    stream_period = 1.0 / max(5, min(TARGET_FPS, 15))
    while detector:
        frame = getattr(detector, "frame_vis", None)
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            time.sleep(0.1)
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(stream_period)


@web_app.route("/stream/<cam_name>")
def stream(cam_name):
    return Response(
        generate_frames(cam_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@web_app.route("/")
def index():
    html = f"""
    <html><head><title>SmartSense Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
      body{{background:#111;color:#fff;font-family:sans-serif;padding:16px;}}
      img{{border:2px solid #555;border-radius:5px;max-width:100%;}}
      .nav{{margin-bottom:20px;}}
      .nav a{{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;}}
      .badge{{display:inline-block;background:#222;border:1px solid #00ffcc;padding:4px 10px;border-radius:6px;margin-bottom:12px;}}
    </style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
        <a href="/health">💚 Health</a>
    </div>
    <h1>Live Camera Feeds</h1>
    <div class="badge">TARGET_FPS = {TARGET_FPS} · device = {device}</div>
    """
    for name in global_detectors.keys():
        html += f"<h3>{name}</h3><img src='/stream/{name}' width='640'/><br>"
    html += "</body></html>"
    return render_template_string(html)


@web_app.route("/health")
def health():
    cams = {}
    for name, reader in global_readers.items():
        det = global_detectors.get(name)
        cams[name] = {
            "rtsp_fps": round(reader.measured_fps, 2),
            "infer_fps": round(getattr(det, "infer_fps", 0), 2) if det else 0,
            "seq": reader.seq,
            "last_frame_age_s": round(time.time() - reader.last_ok, 2) if reader.last_ok else None,
            "connected": reader.cap is not None and reader.cap.isOpened() if reader.cap else False,
        }
    return {
        "ok": True,
        "target_fps": TARGET_FPS,
        "device": device,
        "cameras": cams,
    }


@web_app.route("/events")
def list_events():
    files = sorted(
        os.listdir(EVENT_DIR),
        key=lambda x: os.path.getmtime(os.path.join(EVENT_DIR, x)),
        reverse=True,
    )
    html = """
    <html><head><title>Events</title>
    <style>body{background:#111;color:#fff;font-family:sans-serif;} .nav a{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;} .file-list{margin-top:20px;} .file-item{margin-bottom:10px;}</style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
        <a href="/health">💚 Health</a>
    </div>
    <h1>Saved Events</h1>
    <div class="file-list">
    """
    for f in files:
        if f.endswith(".jpg"):
            html += f"<div class='file-item'>📸 <a style='color:#fff;' target='_blank' href='/events/{f}'>{f}</a></div>"
        elif f.endswith(".mp4"):
            html += f"<div class='file-item'>🎥 <a style='color:#fff;' target='_blank' href='/events/{f}'>{f}</a></div>"
    html += "</div></body></html>"
    return render_template_string(html)


@web_app.route("/events/<path:filename>")
def serve_event(filename):
    return send_from_directory(EVENT_DIR, filename)


@web_app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        new_conf = request.form.get("cameras_json")
        try:
            parsed = json.loads(new_conf)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=4, ensure_ascii=False)
            threading.Thread(target=lambda: (time.sleep(1), os._exit(1))).start()
            return (
                "<h2 style='color:green;'>Lưu thành công! Đang khởi động lại... "
                "Tải lại trang sau 5 giây.</h2>"
            )
        except Exception as e:
            return f"<h2 style='color:red;'>Lỗi cú pháp JSON: {e}</h2>"

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            conf_str = f.read()
    else:
        conf_str = json.dumps({"target_fps": TARGET_FPS, "cameras": CAMERAS}, indent=4)

    html = (
        """
    <html><head><title>Settings</title>
    <style>body{background:#111;color:#fff;font-family:sans-serif;} .nav a{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;} textarea{width:80%; height:300px; background:#222; color:#fff; font-family:monospace; padding:10px; font-size:16px;} .hint{color:#aaa;max-width:800px;}</style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
        <a href="/health">💚 Health</a>
    </div>
    <h1>Camera Configuration (JSON)</h1>
    <p class="hint">
      Format khuyến nghị:<br>
      <code>{"target_fps": 15, "cameras": {"Cam1": "rtsp://..."}}</code><br>
      Dùng <b>subtype=1</b> (substream) trên Dahua/Kbvision để nhẹ CPU — thường 640x360 @ 10–15fps.<br>
      target_fps chỉ nhận <b>10</b> hoặc <b>15</b>.
    </p>
    <form method="POST">
        <textarea name="cameras_json">"""
        + conf_str
        + """</textarea><br><br>
        <button type="submit" style="padding:10px 20px; background:#00ffcc; color:#000; font-weight:bold; font-size:16px; border:none; border-radius:5px; cursor:pointer;">Save & Restart System</button>
    </form>
    </body></html>
    """
    )
    return render_template_string(html)


def run_webui():
    web_app.run(host="0.0.0.0", port=9000, debug=False, use_reloader=False)


# ===================== MAIN =====================
def main():
    global global_detectors, global_readers
    print("=" * 60)
    print(f" SmartSense V12.6 | TARGET_FPS={TARGET_FPS} | cameras={len(CAMERAS)}")
    print(" RTSP: tcp + nobuffer + low_delay | grab/retrieve latest-frame")
    print(" Web UI: http://0.0.0.0:9000  | Health: /health")
    print("=" * 60)

    readers = {name: ReaderThread(name, url) for name, url in CAMERAS.items()}
    for r in readers.values():
        r.start()
    detectors = {name: DetectorThread(name, readers[name]) for name in readers}
    global_detectors = detectors
    global_readers = readers
    for d in detectors.values():
        d.start()

    threading.Thread(target=run_webui, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for r in readers.values():
            r.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
