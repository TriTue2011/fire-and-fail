"""
V12.5_SmartSense.py — Multi-camera, multi-person fall detection with skeleton
Features:
- Multi-ID tracking (centroid matching)
- Skeleton visualization
- Fall detection with fall_streak + smart cooldown
- Telegram alerts (only photo)
- Save pre/post event clips locally
- Motion prefilter to reduce unnecessary inference
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
from ctypes import *
from firesdk import *
from flask import Flask, Response, render_template_string, request, send_from_directory
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== CONFIG =====================
CONFIG_FILE = "cameras.json"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CAMERAS = json.load(f)
else:
    CAMERAS = {
        "Cam1": "rtsp://admin:Tung1234@192.168.1.3:554/cam/realmonitor?channel=1&subtype=0"
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(CAMERAS, f, indent=4)

EVENT_DIR = "events"
os.makedirs(EVENT_DIR, exist_ok=True)

MODEL_PATH = "yolov8n-pose.pt"
CONF_THRES = 0.35
IMG_SIZE = 640
INFER_PERIOD = 0.25  # giây giữa các lần inference

FIRE_INFER_PERIOD = 1.0
FIRE_COOLDOWN_SECONDS = 10
fire_sdk_lock = threading.Lock()

def init_fire_sdk():
    machineCode = getMachineCode()
    print("machineCode: ", machineCode.decode('utf-8'))
    try:
        with open("license.txt", 'r') as file:
            license_key = file.read().strip()
    except IOError as exc:
        print("failed to open license.txt: ", exc.errno)
        license_key = ""
    ret = setActivation(license_key.encode('utf-8'))
    print("activation: ", ret)
    ret = initSDK()
    print("init fire SDK: ", ret)

def mat_to_bytes(mat):
    is_success, buffer = cv2.imencode(".png", mat)
    return buffer.tobytes()

# Fall detection parameters (giữ nguyên độ nhạy hiện tại)
TORSO_ANGLE_THRESHOLD = 45
STATIONARY_SECONDS = 2.2
STATIONARY_MOVEMENT_PX = 35
FALL_CONFIRM_FRAMES = 2
COOLDOWN_SECONDS = 10
RECOVER_SECONDS = 2.0
ID_DISTANCE_PX = 80  # khoảng cách centroid tối đa để match ID

# Motion prefilter
MOTION_THRESHOLD = 40
MOTION_MIN_AREA = 5000

# Telegram
TELEGRAM_BOT_TOKEN = "8490100526:AAFhLTElDAO20zsw-Fi3gIs58jR5sY9Cc-M"
TELEGRAM_CHAT_ID = "7423399175"

# Live window size
TILE_W, TILE_H = 640, 360

# Event clip duration
PRE_EVENT_SECONDS = 5
POST_EVENT_SECONDS = 5
EVENT_DURATION = PRE_EVENT_SECONDS + POST_EVENT_SECONDS

# ===================== LOAD YOLO =====================
print("🚀 Loading YOLO pose model...")
model = YOLO(MODEL_PATH)
device = "cuda" if model.device.type == "cuda" else "cpu"
print(f"✅ Using device: {device}")

# ===================== TELEGRAM HELPERS =====================
def send_telegram_photo(image_path, caption=""):
    try:
        with open(image_path, "rb") as photo:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=10
            )
    except Exception as e:
        print(f"[Telegram] ❌ Failed to send photo: {e}")

# ===================== THREADS =====================
class ReaderThread(threading.Thread):
    def __init__(self, name, url):
        super().__init__(daemon=True)
        self.name, self.url = name, url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def open_stream(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[{self.name}] 🔌 Kết nối camera...")
                self.cap = self.open_stream()
                time.sleep(1)
                if not self.cap.isOpened():
                    print(f"[{self.name}] ❌ Không kết nối được, thử lại sau 5s.")
                    self.cap = None
                    time.sleep(5)
                    continue
                print(f"[{self.name}] ✅ Đã kết nối.")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"[{self.name}] ⚠️ Mất tín hiệu → reconnect")
                self.cap.release()
                self.cap = None
                time.sleep(2)
                continue
            with self.lock:
                self.frame = frame.copy()
            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_fps(self):
        try:
            if self.cap:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 1: return int(round(fps))
        except:
            pass
        return 25

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ===================== DETECTOR THREAD =====================
class DetectorThread(threading.Thread):
    def __init__(self, name, reader):
        super().__init__(daemon=True)
        self.name = name
        self.reader = reader
        self.motion_bg = None
        self.last_infer = 0
        self.buffer = deque(maxlen=int(EVENT_DURATION*30))
        self.people = {}  # pid -> state
        self.next_id = 1
        self.frame_vis = np.zeros((TILE_H, TILE_W,3), np.uint8)
        self.lock = threading.Lock()
        self.last_fire_infer = 0
        self.last_fire_alert = 0

    def assign_id(self, centroid):
        for pid, pdata in self.people.items():
            if "centroid" in pdata:
                d = math.hypot(centroid[0]-pdata["centroid"][0], centroid[1]-pdata["centroid"][1])
                if d < ID_DISTANCE_PX:
                    return pid
        pid = self.next_id
        self.next_id +=1
        return pid

    def draw_skeleton(self, frame, keypoints_xy, color=(0,255,0)):
        pairs = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16),(11,12)]
        for (x,y) in keypoints_xy:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        for a,b in pairs:
            if a < len(keypoints_xy) and b < len(keypoints_xy):
                p1, p2 = keypoints_xy[a], keypoints_xy[b]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color,2)

    def run(self):
        while True:
            frame = self.reader.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # motion prefilter
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(5,5),0)
            if self.motion_bg is None:
                self.motion_bg = gray
                continue
            diff = cv2.absdiff(self.motion_bg, gray)
            _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            motion_area = np.sum(thresh)/255
            self.motion_bg = cv2.addWeighted(self.motion_bg,0.9,gray,0.1,0)
            self.buffer.append(frame.copy())

            if motion_area < MOTION_MIN_AREA:
                self._set_vis(frame)
                time.sleep(0.02)
                continue
            if time.time()-self.last_infer < INFER_PERIOD:
                self._set_vis(frame)
                time.sleep(0.01)
                continue
            self.last_infer = time.time()

            # YOLO pose
            try:
                results = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False, device=device)
            except Exception as e:
                print(f"[{self.name}] Model predict error: {e}")
                self._set_vis(frame)
                continue
            
            # Fire detection
            if time.time() - self.last_fire_infer > FIRE_INFER_PERIOD:
                self.last_fire_infer = time.time()
                try:
                    img_byte = mat_to_bytes(frame)
                    box_array = (c_int * 1024)()
                    score_array = (c_float * 1024)()
                    label_array = (c_int * 1024)()
                    
                    with fire_sdk_lock:
                        cnt = getFireDetection(img_byte, len(img_byte), label_array, box_array, score_array)
                    
                    if cnt > 0:
                        fire_detected = False
                        for i in range(cnt):
                            x, y, w, h = box_array[i*4], box_array[i*4+1], box_array[i*4+2], box_array[i*4+3]
                            label = "fire" if label_array[i] == 0 else "smoke"
                            score = score_array[i]
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                            cv2.putText(frame, f"{label} {score:.2f}", (x, max(10, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            fire_detected = True
                        
                        if fire_detected and time.time() - self.last_fire_alert > FIRE_COOLDOWN_SECONDS:
                            self.last_fire_alert = time.time()
                            print(f"[{self.name}] 🔥 FIRE/SMOKE DETECTED!")
                            event_time=datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path=os.path.join(EVENT_DIR,f"{self.name}_FIRE_{event_time}.jpg")
                            cv2.imwrite(img_path, frame)
                            send_telegram_photo(img_path,caption=f"🚨 FIRE/SMOKE in {self.name} at {event_time}")
                            threading.Thread(target=self._save_clip,args=("FIRE",),daemon=True).start()
                except Exception as e:
                    print(f"[{self.name}] Fire detection error: {e}")

            persons = []
            try:
                if results[0].keypoints is not None:
                    persons = results[0].keypoints.xy
            except:
                persons = []

            now = time.time()
            seen_pids = set()
            for kp in persons:
                try:
                    kparr = kp.cpu().numpy()[:,:2] if hasattr(kp,"cpu") else np.asarray(kp)[:,:2]
                except:
                    continue
                if kparr.shape[0]<13: continue
                shoulder_mid = np.mean(kparr[[5,6]],axis=0)
                hip_mid = np.mean(kparr[[11,12]],axis=0)
                dx,dy = hip_mid[0]-shoulder_mid[0], hip_mid[1]-shoulder_mid[1]
                angle = abs(math.degrees(math.atan2(dy, dx)))
                pid = self.assign_id(hip_mid)
                seen_pids.add(pid)

                if pid not in self.people:
                    self.people[pid] = {"centroid":hip_mid, "positions":deque(maxlen=60),
                                        "fall_frame_count":0,"candidate_start":None,
                                        "was_fallen":False,"last_alert":0,"last_seen":now}
                pdata = self.people[pid]
                pdata["centroid"] = hip_mid
                pdata["positions"].append((now, hip_mid))
                pdata["last_seen"] = now

                # check movement
                recent = [(t,p) for t,p in pdata["positions"] if now-t<STATIONARY_SECONDS]
                moved = sum(math.hypot(p2[0]-p1[0],p2[1]-p1[1])
                            for (_,p1),(_,p2) in zip(recent,list(recent)[1:]))
                fall_flag = angle>TORSO_ANGLE_THRESHOLD and moved<STATIONARY_MOVEMENT_PX

                # fall streak
                if fall_flag: pdata["fall_frame_count"]+=1
                else: pdata["fall_frame_count"]=max(0,pdata["fall_frame_count"]-1)

                # confirm fall
                if pdata["fall_frame_count"]>=FALL_CONFIRM_FRAMES and not pdata["was_fallen"]:
                    if pdata["candidate_start"] is None:
                        pdata["candidate_start"]=now
                    else:
                        if now-pdata["candidate_start"]>=STATIONARY_SECONDS:
                            confirm_recent=[(t,p) for t,p in pdata["positions"] if now-t<STATIONARY_SECONDS]
                            moved_confirm=sum(math.hypot(p2[0]-p1[0],p2[1]-p1[1])
                                              for (_,p1),(_,p2) in zip(confirm_recent,list(confirm_recent)[1:]))
                            if moved_confirm<STATIONARY_MOVEMENT_PX:
                                pdata["was_fallen"]=True
                                pdata["fall_frame_count"]=0
                                pdata["candidate_start"]=None
                                print(f"[{self.name}] CONFIRMED FALL -> ID {pid} angle={angle:.1f}")
                                self._alert_and_save(frame.copy(), pid)
                            else:
                                pdata["candidate_start"]=None
                                pdata["fall_frame_count"]=0

                # recovery
                if pdata["was_fallen"]:
                    rec_window=[(t,p) for t,p in pdata["positions"] if now-t<RECOVER_SECONDS]
                    moved_rec=sum(math.hypot(p2[0]-p1[0],p2[1]-p1[1])
                                  for (_,p1),(_,p2) in zip(rec_window,list(rec_window)[1:]))
                    if moved_rec>STATIONARY_MOVEMENT_PX:
                        pdata["was_fallen"]=False
                        pdata["candidate_start"]=None
                        pdata["fall_frame_count"]=0
                        print(f"[{self.name}] RECOVERED ID {pid}")

                # draw skeleton
                color=(0,0,255) if pdata["was_fallen"] else (0,255,0)
                try:
                    self.draw_skeleton(frame,kparr,color)
                except:
                    pass
                cv2.putText(frame,f"ID {pid} [{'FALL' if pdata['was_fallen'] else 'OK'}]",
                            (int(hip_mid[0])+8,int(hip_mid[1])-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                self.people[pid]=pdata

            # cleanup stale people
            stale=[pid for pid,pd in self.people.items() if now-pd.get("last_seen",now)>6.0]
            for pid in stale: del self.people[pid]

            self._set_vis(frame)
            time.sleep(0.005)

    def _alert_and_save(self, frame_snapshot, pid):
        pdata=self.people.get(pid)
        if pdata is None: return
        now=time.time()
        if now-pdata.get("last_alert",0)<COOLDOWN_SECONDS:
            # Không log suppressed → tránh spam
            return
        pdata["last_alert"]=now
        self.people[pid]=pdata

        event_time=datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path=os.path.join(EVENT_DIR,f"{self.name}_ID{pid}_{event_time}.jpg")
        cv2.imwrite(img_path,frame_snapshot)
        send_telegram_photo(img_path,caption=f"🚨 FALL {self.name} ID {pid} at {event_time}")
        threading.Thread(target=self._save_clip,args=(pid,),daemon=True).start()

    def _save_clip(self,pid):
        fps=self.reader.get_fps() or 25
        pre_count=int(PRE_EVENT_SECONDS*fps)
        buf=list(self.buffer)
        pre_frames=buf[-pre_count:] if len(buf)>=pre_count else buf[:]
        if not pre_frames: return
        h,w=pre_frames[0].shape[:2]
        event_time=datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_path=os.path.join(EVENT_DIR,f"{self.name}_ID{pid}_{event_time}.mp4")
        out=cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
        for f in pre_frames: out.write(f)
        t_end=time.time()+POST_EVENT_SECONDS
        while time.time()<t_end:
            f=self.reader.get_frame()
            if f is not None: out.write(f)
            time.sleep(1.0/max(1,fps))
        out.release()
        print(f"[{self.name}] 💾 Saved clip: {vid_path}")

    def _set_vis(self, frame):
        vis=cv2.resize(frame,(TILE_W,TILE_H))
        with self.lock:
            self.frame_vis=vis

# ===================== LIVE VIEW =====================
class LiveViewThread(threading.Thread):
    def __init__(self, detectors):
        super().__init__(daemon=True)
        self.detectors=detectors

    def run(self):
        print("[LiveView] Press ESC to quit")
        time.sleep(1.5)
        while True:
            frames=[]
            for name,det in self.detectors.items():
                try: frame=getattr(det,"frame_vis",None)
                except: frame=None
                if frame is None:
                    frame=np.zeros((TILE_H,TILE_W,3),np.uint8)
                    cv2.putText(frame,f"{name} Loading...",(50,TILE_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
                frames.append(frame)
            if len(frames)>=4:
                top=np.hstack(frames[0:2])
                bottom=np.hstack(frames[2:4])
                grid=np.vstack([top,bottom])
            else: grid=np.hstack(frames)
            cv2.imshow("Multi-Live View V12.5 SmartSense",grid)
            key=cv2.waitKey(1)
            if key & 0xFF==27:
                print("[LiveView] ESC -> exit")
                os._exit(0)
            time.sleep(0.01)

# ===================== WEB UI =====================
web_app = Flask(__name__)
global_detectors = {}

def generate_frames(cam_name):
    detector = global_detectors.get(cam_name)
    while detector:
        frame = getattr(detector, "frame_vis", None)
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.1)
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@web_app.route('/stream/<cam_name>')
def stream(cam_name):
    return Response(generate_frames(cam_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@web_app.route('/')
def index():
    html = """
    <html><head><title>SmartSense Dashboard</title>
    <style>body{background:#111;color:#fff;font-family:sans-serif;} img{border:2px solid #555;border-radius:5px;} .nav{margin-bottom:20px;} .nav a{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;}</style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
    </div>
    <h1>Live Camera Feeds</h1>
    """
    for name in global_detectors.keys():
        html += f"<h3>{name}</h3><img src='/stream/{name}' width='640'/><br>"
    html += "</body></html>"
    return render_template_string(html)

@web_app.route('/events')
def list_events():
    files = sorted(os.listdir(EVENT_DIR), key=lambda x: os.path.getmtime(os.path.join(EVENT_DIR, x)), reverse=True)
    html = """
    <html><head><title>Events</title>
    <style>body{background:#111;color:#fff;font-family:sans-serif;} .nav a{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;} .file-list{margin-top:20px;} .file-item{margin-bottom:10px;} video{max-width:640px; margin-bottom:20px;}</style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
    </div>
    <h1>Saved Events</h1>
    <div class="file-list">
    """
    for f in files:
        if f.endswith('.jpg'):
            html += f"<div class='file-item'>📸 <a style='color:#fff;' target='_blank' href='/events/{f}'>{f}</a></div>"
        elif f.endswith('.mp4'):
            html += f"<div class='file-item'>🎥 <a style='color:#fff;' target='_blank' href='/events/{f}'>{f}</a></div>"
    html += "</div></body></html>"
    return render_template_string(html)

@web_app.route('/events/<path:filename>')
def serve_event(filename):
    return send_from_directory(EVENT_DIR, filename)

@web_app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        new_conf = request.form.get('cameras_json')
        try:
            parsed = json.loads(new_conf)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=4)
            # Khởi động lại container sau 1s
            threading.Thread(target=lambda: (time.sleep(1), os._exit(1))).start()
            return "<h2 style='color:green;'>Lưu thành công! Đang khởi động lại hệ thống... Vui lòng tải lại trang sau 5 giây.</h2>"
        except Exception as e:
            return f"<h2 style='color:red;'>Lỗi cú pháp JSON: {e}</h2>"
            
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        conf_str = f.read()
        
    html = """
    <html><head><title>Settings</title>
    <style>body{background:#111;color:#fff;font-family:sans-serif;} .nav a{color:#00ffcc; margin-right:15px; text-decoration:none; font-size:18px;} textarea{width:80%; height:300px; background:#222; color:#fff; font-family:monospace; padding:10px; font-size:16px;}</style>
    </head><body>
    <div class="nav">
        <a href="/">📺 Live Feeds</a>
        <a href="/events">🎞️ View Events</a>
        <a href="/config">⚙️ Settings</a>
    </div>
    <h1>Camera Configuration (JSON)</h1>
    <form method="POST">
        <textarea name="cameras_json">""" + conf_str + """</textarea><br><br>
        <button type="submit" style="padding:10px 20px; background:#00ffcc; color:#000; font-weight:bold; font-size:16px; border:none; border-radius:5px; cursor:pointer;">Save & Restart System</button>
    </form>
    </body></html>
    """
    return render_template_string(html)

def run_webui():
    web_app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False)

# ===================== MAIN =====================
def main():
    global global_detectors
    init_fire_sdk()
    readers={name:ReaderThread(name,url) for name,url in CAMERAS.items()}
    for r in readers.values(): r.start()
    detectors={name:DetectorThread(name,readers[name]) for name in readers}
    global_detectors = detectors
    for d in detectors.values(): d.start()
    
    # Start Web UI
    threading.Thread(target=run_webui, daemon=True).start()
    
    # live=LiveViewThread(detectors)
    # live.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for r in readers.values(): r.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
