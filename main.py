
# main_gpu.py
"""
GPU-optimized multithreaded Video Attendance Processor with smooth Tkinter UI.

- Grabber thread reads frames at source framerate (webcam/phone/video).
- Tracker/display thread updates OpenCV trackers (CSRT/KCF/MOSSE) and publishes annotated frames.
- Recognition threadpool runs face_recognition on detection ROIs in parallel.
- Detection runs every DETECT_INTERVAL_SECONDS on a downscaled frame (DETECTION_DOWNSCALE).
- Marks "Present" only after PRESENCE_SECONDS continuous detection/tracking.
- Uses pygame for ding sound (non-blocking).
- Attempts to use cv2.cuda for faster resizing if available.
- Saves attendance CSV (first-seen timestamps).
"""

import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel

import cv2
import numpy as np
import face_recognition
import pandas as pd
from PIL import Image, ImageTk,ImageFile

# pygame for sound (non-blocking)
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# ---------------- CONFIG ----------------
KNOWN_DIR = "known_faces"
CSV_FOLDER = "attendance_logs"
DING_FILE = "ding.wav"               # wav file in project root recommended

DETECT_INTERVAL_SECONDS = 5.0        # run full detection every N seconds
DETECTION_DOWNSCALE = 0.5           # downscale for detection (smaller -> faster)
DETECTION_MODEL = "hog"              # 'cnn' (slower but better) or 'hog'
TRACKER_PREF = [ "MOSSE","CSRT", "KCF"]
RECOGNITION_COOLDOWN = 2.5           # seconds before re-recognizing the same area
PRESENCE_SECONDS = 0              # sustained seconds to mark present
MAX_WORKERS = 8                      # parallel recognition workers
MIN_FACE_H = 40                      # skip tiny faces
UI_TARGET_WIDTH = 960
UI_TARGET_HEIGHT = 540
UI_REFRESH_MS = 50                  # ~30 FPS UI
# ----------------------------------------

# create folders
KNOWN_DIR = r"C:\Users\Admin\PycharmProjects\digimark2\known_faces"


Path(CSV_FOLDER).mkdir(exist_ok=True)

# Check for cv2.cuda availability (OpenCV compiled with CUDA)
USE_CV2_CUDA = False
try:
    if hasattr(cv2, "cuda"):
        cnt = cv2.cuda.getCudaEnabledDeviceCount()
        if cnt > 0:
            USE_CV2_CUDA = True
except Exception:
    USE_CV2_CUDA = False

# utility: create tracker
def make_tracker():
    for t in TRACKER_PREF:
        try:
            if t.upper() == "CSRT":
                return cv2.TrackerCSRT_create()
            if t.upper() == "KCF":
                return cv2.TrackerKCF_create()
            if t.upper() == "MOSSE":
                return cv2.TrackerMOSSE_create()
        except Exception:
            continue
    return None

def rect_from_loc(face_location, scale):
    top, right, bottom, left = face_location
    x = int(left / scale)
    y = int(top / scale)
    w = int((right - left) / scale)
    h = int((bottom - top) / scale)
    return (x, y, w, h)

def bbox_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2]); yB = min(a[1] + a[3], b[1] + b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    union = a[2]*a[3] + b[2]*b[3] - interArea
    if union <= 0: return 0.0
    return interArea / union

def save_attendance_csv(recognized_map, all_names):
    today = date.today().isoformat()
    out = Path(CSV_FOLDER) / f"attendance_{today}.csv"
    existing = {}
    if out.exists():
        try:
            df_old = pd.read_csv(out)
            for _, r in df_old.iterrows():
                existing[r["Name"]] = (r.get("Status",""), r.get("FirstSeen",""))
        except Exception:
            existing = {}
    rows = []
    for name in all_names:
        if name in existing and existing[name][0] == "Present":
            rows.append({"Name": name, "Status": "Present", "FirstSeen": existing[name][1]})
        elif name in recognized_map:
            rows.append({"Name": name, "Status": "Present", "FirstSeen": recognized_map[name]})
        else:
            rows.append({"Name": name, "Status": "Absent", "FirstSeen": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out

ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize_path(p: str) -> str:
    """Return a safe absolute path (handles raw/backslashes)."""
    if not p:
        return p
    # Expand user/home and make absolute, replace backslashes
    p = os.path.expanduser(p)
    p = os.path.abspath(p)
    return p

def load_image_any(path):
    """
    Load image from path and return HxWx3 uint8 RGB numpy array.
    Tries cv2.imread, then PIL. Handles 16-bit images by converting to 8-bit.
    Raises ValueError if image cannot be loaded or converted.
    """
    path = normalize_path(path)
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    # Try cv2 first (BGR)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        # If grayscale convert to RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If has alpha channel drop it
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # If not uint8 (e.g., uint16), convert by scaling
        if img.dtype != np.uint8:
            # Common case: uint16 -> scale down by 256
            try:
                if np.issubdtype(img.dtype, np.integer):
                    img = (img / (img.dtype.type(256))).astype(np.uint8)
                else:
                    # fallback normalize to 0-255
                    mn, mx = img.min(), img.max()
                    if mx > mn:
                        img = ((img - mn) * (255.0 / (mx - mn))).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
            except Exception:
                img = cv2.convertScaleAbs(img)
        # Convert BGR -> RGB for face_recognition
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"OpenCV conversion error for {path}: {e}")
        return rgb

    # Fallback to PIL (handles many formats)
    try:
        pil = Image.open(path)
        pil = pil.convert("RGB")  # Forces 8-bit RGB
        arr = np.array(pil)
        if arr.dtype != np.uint8:
            # convert if PIL returned a different dtype
            arr = (255 * (arr.astype(np.float32) / arr.max())).astype(np.uint8)
        return arr
    except Exception as e:
        raise ValueError(f"Cannot load image {path}: {e}")

def load_known_faces(folder=KNOWN_DIR):
    encodings = []
    names = []

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Created folder: {folder}")
        return encodings, names

    print(f"[INFO] Loading known faces from: {folder}")

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            print(f"⚠️ Skipping (not found): {path}")
            continue

        try:
            # Read safely with OpenCV (handles 16-bit/8-bit automatically)
            img = cv2.imread(path)
            if img is None:
                print(f"❌ Could not load {fname} (invalid file or path)")
                continue

            # Convert to RGB for face_recognition
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Encode face
            enc = face_recognition.face_encodings(rgb)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(fname)[0])
                print(f"✅ Loaded: {fname}")
            else:
                print(f"⚠️ No face found in {fname}, skipping.")

        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")

    print(f"[INFO] Total known faces loaded: {len(names)}")
    return encodings, names

def recognize_face_roi(face_bgr, known_encodings, known_names, tolerance=0.5):
    """Run face_recognition on a BGR ROI; return name or 'Unknown'."""
    try:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return "Unknown"
        enc = encs[0]
        if not known_encodings:
            return "Unknown"
        dists = face_recognition.face_distance(known_encodings, enc)
        idx = int(np.argmin(dists))
        if dists[idx] <= tolerance:
            return known_names[idx]
        return "Unknown"
    except Exception:
        return "Unknown"

# ---------------- Core App ----------------
class MainGPUAttendance:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("GPU-Optimized Attendance (Detect+Track)")

        # UI
        self.video_panel = tk.Label(root, bg="black")
        self.video_panel.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, pady=6)
        tk.Button(ctrl, text="Select Video/File", command=self.select_video).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Use Webcam", command=self.use_webcam).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Use Phone Stream", command=self.set_phone_stream).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Start", command=self.start).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Reload Faces", command=self.reload_faces).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Save CSV", command=self.save_now).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="Present: 0 / 0")
        tk.Label(ctrl, textvariable=self.status_var, font=("Arial", 12)).pack(side=tk.RIGHT, padx=6)

        # logs
        bottom = tk.Frame(root)
        bottom.pack(fill=tk.BOTH, expand=False, padx=6, pady=6)
        self.logbox = tk.Text(bottom, height=8)
        self.logbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(bottom, command=self.logbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.logbox.configure(yscrollcommand=sb.set)

        # progress
        self.progress = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=4)

        # state
        self.known_encodings, self.known_names = load_known_faces()
        self.recognized_names = {}       # name->first_seen_time
        self.name_timers = {}            # name->first_detection_epoch (for PRESENCE_SECONDS)
        self.recognition_cooldowns = {}  # bbox_key->last_recog_time

        self.trackers = []               # list of dicts {id, tracker, bbox, name, last_seen, last_recog}
        self.trackers_lock = threading.Lock()

        self.frame_queue = queue.Queue(maxsize=2)  # latest frames
        self.frame_for_ui = None
        self.frame_for_ui_lock = threading.Lock()

        self.cap = None
        self.video_path = None
        self.source_type = "webcam"      # "webcam"/"phone"/"video"
        self.phone_url = ""

        self.fps = 25.0
        self.total_frames = 0
        self.cur_frame_idx = 0

        # threads & executor
        self.running = False
        self.grabber_thread = None
        self.detect_thread = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.pending_futures = set()
        self.last_detection_time = 0.0

        # UI loop
        self.root.after(UI_REFRESH_MS, self.update_ui)

        self.log(f"CUDA available: {USE_CV2_CUDA} | pygame sound: {PYGAME_AVAILABLE}")

    # ----- logging / UI helpers -----
    def log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            # schedule append on main thread
            self.root.after(0, lambda: (self.logbox.insert("end", f"[{ts}] {text}\n"), self.logbox.see("end")))
        except Exception:
            print(f"[{ts}] {text}")

    def update_status(self):
        total = len(self.known_names)
        present = len(self.recognized_names)
        try:
            self.root.after(0, lambda: self.status_var.set(f"Present: {present} / {total}"))
        except Exception:
            pass

    # ----- face file management -----
    def reload_faces(self):
        with self.trackers_lock:
            self.known_encodings, self.known_names = load_known_faces()
            self.recognized_names.clear()
            self.name_timers.clear()
            self.trackers.clear()
        self.log(f"Faces reloaded ({len(self.known_names)} known).")
        self.update_status()

    def save_now(self):
        out = save_attendance_csv(self.recognized_names, self.known_names)
        messagebox.showinfo("Saved", f"Attendance saved to:\n{out}")
        self.log(f"Saved CSV -> {out}")

    # ----- camera / video selection -----
    def select_video(self):
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.video_path = path
        self.source_type = "video"
        self.cap = None
        # read fps & frames
        tmp = cv2.VideoCapture(self.video_path)
        if tmp.isOpened():
            self.fps = tmp.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            tmp.release()
        self.log(f"Selected video: {os.path.basename(self.video_path)} | FPS: {self.fps:.2f} | Frames: {self.total_frames}")
        messagebox.showinfo("Video Selected", f"{os.path.basename(self.video_path)}\nFPS: {self.fps:.2f} | Frames: {self.total_frames}")

    def use_webcam(self):
        self.source_type = "webcam"
        self.video_path = None
        self.phone_url = ""
        self.cap = None
        self.log("Source set: Webcam")

    def set_phone_stream(self):
        url = tk.simpledialog.askstring("Phone stream URL", "Enter phone camera URL (e.g. http://192.168.0.xx:8080/video):")
        if not url:
            return
        self.phone_url = url.strip()
        self.source_type = "phone"
        self.video_path = None
        self.cap = None
        self.log(f"Phone stream set: {self.phone_url}")

    def restart_capture(self):
        # release
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        # open
        src = 0
        if self.source_type == "video" and self.video_path:
            src = self.video_path
        elif self.source_type == "phone" and self.phone_url:
            src = self.phone_url
        else:
            src = 0
        self.cap = cv2.VideoCapture(src)
        # hints (may be ignored)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, UI_TARGET_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, UI_TARGET_HEIGHT)
        except Exception:
            pass
        time.sleep(0.05)

    # ----- sound -----
    def play_ding(self):
        if PYGAME_AVAILABLE and os.path.exists(DING_FILE):
            try:
                pygame.mixer.Sound(DING_FILE).play()
            except Exception:
                pass
        else:
            # fallback beep
            print("\a", end="", flush=True)

    # ----- start / stop lifecycle -----
    def start(self):
        if self.running:
            return
        self.running = True
        self.recognized_names.clear()
        self.name_timers.clear()
        self.recognition_cooldowns.clear()
        with self.trackers_lock:
            self.trackers.clear()
        # prepare capture
        self.restart_capture()
        if not self.cap or not self.cap.isOpened():
            self.log("⚠️ Unable to open capture source.")
            messagebox.showerror("Camera Error", "Unable to open camera / file / stream.")
            self.running = False
            return
        # start threads
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.grabber_thread.start()
        self.detect_thread.start()
        self.log("Started processing (grabber + detector + recognition pool).")

    def stop(self):
        if not self.running:
            return
        self.running = False
        time.sleep(0.1)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        # cancel pending futures
        for f in list(self.pending_futures):
            try:
                f.cancel()
            except Exception:
                pass
        self.pending_futures.clear()
        self.log("Stopped processing.")
        # show results if video file or stream ended
        # results displayed by monitor thread if necessary

    # ----- grabber thread: read frames at source fps and put latest into queue -----
    def _grabber_loop(self):
        cap = self.cap
        # determine fps for playback timing (video) else use default
        fps = 25.0
        try:
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps = fps_val if fps_val and not np.isnan(fps_val) and fps_val > 1 else fps
        except Exception:
            fps = 25.0
        frame_delay = 1.0 / max(1.0, fps)
        self.cur_frame_idx = 0
        eof = False
        while self.running and cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                eof = True
                break
            self.cur_frame_idx += 1
            # push latest frame (drop older)
            try:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            elapsed = time.time() - t0
            to_sleep = frame_delay - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
        # done
        self.running = False
        self.root.after(100, self.show_results_and_save)

        if eof:
            self.log("📽 End of video / stream closed.")
            # show results now
            self.root.after(50, self.show_results_and_save)

    # ----- detection loop: runs periodic detection on frames and schedules recognition jobs -----
    def _detection_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.6)
            except queue.Empty:
                continue

            now = time.time()
            self._update_trackers(frame)

            # Skip detection if not enough time has passed
            if (now - self.last_detection_time) < DETECT_INTERVAL_SECONDS:
                continue

            # --- Preprocess for CCTV ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame_eq = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Downscale with GPU acceleration if available
            if USE_CV2_CUDA:
                try:
                    gmat = cv2.cuda_GpuMat()
                    gmat.upload(frame_eq)
                    gsz = (int(frame_eq.shape[1] * DETECTION_DOWNSCALE),
                           int(frame_eq.shape[0] * DETECTION_DOWNSCALE))
                    gres = cv2.cuda.resize(gmat, gsz)
                    small = gres.download()
                except Exception:
                    small = cv2.resize(frame_eq, (0, 0), fx=DETECTION_DOWNSCALE, fy=DETECTION_DOWNSCALE)
            else:
                small = cv2.resize(frame_eq, (0, 0), fx=DETECTION_DOWNSCALE, fy=DETECTION_DOWNSCALE)

            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            try:
                locations = face_recognition.face_locations(rgb_small, model=DETECTION_MODEL)
            except Exception:
                locations = face_recognition.face_locations(rgb_small, model="hog")

            if not locations:
                # Try a higher resolution pass if no faces found
                small2 = cv2.resize(frame_eq, (0, 0), fx=0.6, fy=0.6)
                rgb_small2 = cv2.cvtColor(small2, cv2.COLOR_BGR2RGB)
                try:
                    locations = face_recognition.face_locations(rgb_small2, model=DETECTION_MODEL)
                except Exception:
                    locations = []

            encs = face_recognition.face_encodings(rgb_small, locations) if locations else []

            for loc, enc in zip(locations, encs):
                bbox = rect_from_loc(loc, DETECTION_DOWNSCALE)
                x, y, w, h = bbox
                if h < MIN_FACE_H:
                    continue

                # Avoid overlapping trackers
                overlapped = False
                with self.trackers_lock:
                    for tr in self.trackers:
                        if bbox_iou(bbox, tr["bbox"]) > 0.35:
                            overlapped = True
                            break
                if overlapped:
                    continue

                tracker_inst = make_tracker()
                if tracker_inst is not None:
                    try:
                        tracker_inst.init(frame, tuple(bbox))
                        tid = str(uuid4())[:8]
                        tr = {"id": tid, "tracker": tracker_inst, "bbox": bbox,
                              "name": "Unknown", "last_seen": time.time(), "last_recog": 0.0}
                        with self.trackers_lock:
                            self.trackers.append(tr)
                    except Exception:
                        pass

                bbox_key = (int(x / 10), int(y / 10), int(w / 10), int(h / 10))
                last = self.recognition_cooldowns.get(bbox_key, 0)
                if (time.time() - last) > RECOGNITION_COOLDOWN:
                    try:
                        face_roi = frame[y:y + h, x:x + w].copy()
                        future = self.executor.submit(
                            recognize_face_roi, face_roi,
                            self.known_encodings, self.known_names, 0.45
                        )
                        future._meta = {"bbox": bbox, "bbox_key": bbox_key}
                        self.pending_futures.add(future)
                        self.recognition_cooldowns[bbox_key] = time.time()
                    except Exception:
                        continue

            # Process completed recognitions
            done = {f for f in self.pending_futures if f.done()}
            for f in done:
                try:
                    name = f.result()
                except Exception:
                    name = "Unknown"
                meta = getattr(f, "_meta", {})
                bbox = meta.get("bbox")
                if bbox:
                    with self.trackers_lock:
                        for tr in self.trackers:
                            if bbox_iou(bbox, tr["bbox"]) > 0.35:
                                tr["name"] = name
                                tr["last_recog"] = now
                                tr["last_seen"] = now
                                if name != "Unknown":
                                    if name not in self.name_timers:
                                        self.name_timers[name] = now
                self.pending_futures.discard(f)

            self.last_detection_time = time.time()

    # ----- trackers update & mark present -----
    def _update_trackers(self, frame):
        remove_ids = []
        now = time.time()
        with self.trackers_lock:
            for tr in list(self.trackers):
                try:
                    ok, bbox = tr["tracker"].update(frame)
                except Exception:
                    ok = False
                    bbox = tr["bbox"]
                if ok:
                    tr["bbox"] = tuple(map(int, bbox))
                    tr["last_seen"] = now
                else:
                    # if tracker lost for a while, drop it
                    if now - tr.get("last_seen", now) > 2.5:
                        remove_ids.append(tr["id"])
            if remove_ids:
                self.trackers = [t for t in self.trackers if t["id"] not in remove_ids]

        # check named trackers to mark present after PRESENCE_SECONDS
        with self.trackers_lock:
            for tr in self.trackers:
                name = tr.get("name", "Unknown")
                if name != "Unknown":
                    # initialize timer if absent
                    with threading.Lock():
                        if name not in self.name_timers:
                            self.name_timers[name] = now
                        else:
                            if (now - self.name_timers[name]) >= PRESENCE_SECONDS:
                                if name not in self.recognized_names:
                                    ts = datetime.now().strftime("%H:%M:%S")
                                    self.recognized_names[name] = ts
                                    self.log(f"Marked present: {name} at {ts}")
                                    # ding (non-blocking)
                                    threading.Thread(target=self.play_ding, daemon=True).start()
                                    self.update_status()

    # ----- UI update: called on main thread via after() -----
    def update_ui(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            # Update trackers and draw only current ones
            self._update_trackers(frame)
            draw = frame.copy()
            with self.trackers_lock:
                for tr in list(self.trackers):
                    x, y, w, h = tr["bbox"]
                    name = tr.get("name", "Unknown")
                    color = (0, 255, 0) if name in self.recognized_names else (0, 128, 255)
                    cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(draw, name, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            h0, w0 = draw.shape[:2]
            scale = min(UI_TARGET_WIDTH / w0, UI_TARGET_HEIGHT / h0)
            if scale < 1.0:
                draw = cv2.resize(draw, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)

            # Progress bar
            if self.total_frames > 0:
                progress = (self.cur_frame_idx / self.total_frames) * 100
                self.progress["value"] = min(100, progress)

        self.root.after(UI_REFRESH_MS, self.update_ui)

    # ----- final results and popup table -----
    def show_results_and_save(self):
        known_set = set(self.known_names)
        present_set = set(self.recognized_names.keys())
        absent_set = known_set - present_set

        # save csv
        rows = []
        for name in sorted(known_set):
            status = "Present" if name in present_set else "Absent"
            ts = self.recognized_names.get(name, "")
            rows.append({"Name": name, "Status": status, "FirstSeen": ts})
        df = pd.DataFrame(rows)
        out = Path(CSV_FOLDER) / f"attendance_{date.today().isoformat()}.csv"
        df.to_csv(out, index=False)
        self.log(f"Saved final CSV -> {out}")

        # show popup table on main thread
        def popup():
            win = Toplevel(self.root)
            win.title("Attendance Results")
            tree = ttk.Treeview(win, columns=("Name","Status","FirstSeen"), show="headings")
            tree.heading("Name", text="Name"); tree.heading("Status", text="Status"); tree.heading("FirstSeen", text="First Seen")
            tree.pack(fill="both", expand=True)
            for _, r in df.iterrows():
                tree.insert("", "end", values=(r["Name"], r["Status"], r["FirstSeen"]))
            tk.Label(win, text=f"Present: {len(present_set)}  |  Absent: {len(absent_set)}").pack(pady=6)
        self.root.after(10, popup)

    # ----- monitor thread helper to detect completion and pop results -----
    def monitor_completion(self):
        # call externally (in background) to watch running flag
        while self.running:
            time.sleep(0.5)
        # if stopped due to EOF or manual stop, show results
        self.show_results_and_save()

# ------------------ run ------------------
def main():
    root = tk.Tk()
    app = MainGPUAttendance(root)
    # start completion monitor
    threading.Thread(target=app.monitor_completion, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    main()


#final