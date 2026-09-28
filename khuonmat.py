#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACHINE VISION - KIỂM TRA TIP (2 CAMERA) CHO RASPBERRY PI 4
===========================================================
Luồng xử lý mỗi camera:
    YOLO (NCNN) định vị Tip  ->  cắt ROI vuông  ->  MobileNetV2 lấy đặc trưng
    từng patch (ONNX Runtime)  ->  so với "ngân hàng mẫu OK"  ->  OK / NG

Kết quả cuối: OK khi CẢ 2 camera OK. Bất kỳ lỗi nào (mất camera, không thấy
sản phẩm, chưa học mẫu, exception...) đều trả NG (fail-safe).

Giao thức Arduino (giữ nguyên như bản cũ):
    Arduino -> Pi : "TRIG\\n"
    Pi -> Arduino : "OK\\n" hoặc "NG\\n"

------------------------------------------------------------
CÀI ĐẶT TRÊN PI 4 (Raspberry Pi OS 64-bit, chạy trong VS Code)
------------------------------------------------------------
    sudo apt update
    sudo apt install -y python3-tk python3-pil.imagetk v4l-utils
    sudo usermod -aG dialout $USER        # để đọc cổng serial, rồi đăng xuất/đăng nhập lại

    cd <thư mục chứa file này>
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip install ultralytics ncnn onnxruntime pyserial numpy opencv-python-headless

Trong VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> chọn ./venv/bin/python

------------------------------------------------------------
CHUẨN BỊ MODEL (làm 1 lần)
------------------------------------------------------------
1) YOLO -> NCNN (chạy trên PC hoặc Pi, imgsz phải khớp YOLO_IMGSZ bên dưới):
       yolo export model=best.pt format=ncnn imgsz=320
   Chép thư mục best_ncnn_model vào  ./models/best_ncnn_model
   (hoặc chép best.pt vào ./models/best.pt để chạy tạm bằng PyTorch)

2) Backbone đặc trưng -> ONNX (chạy trên PC có torch + torchvision):
       python inspection_app.py --export-onnx
   Chép ./models/patchnet.onnx sang Pi. Nếu thiếu file này, chương trình sẽ
   tự dùng PyTorch (chậm hơn và cần tải trọng số ImageNet lần đầu).

------------------------------------------------------------
CÁCH DÙNG
------------------------------------------------------------
1) Đặt sản phẩm OK vào vị trí kiểm -> bấm "+ Mẫu OK" ở từng camera.
   Nên thêm 20-40 mẫu OK, thay đổi nhẹ vị trí/ánh sáng như thực tế sản xuất.
2) Ngưỡng tự tính bằng leave-one-out (LOO). Sau đó BẮT BUỘC thử vài sản phẩm
   NG thật, xem "Điểm" và chỉnh ô "Ngưỡng" cho tách được OK / NG.
3) Ảnh NG được lưu ở ./history để sau này huấn luyện lại.
"""

import os
import sys
import json
import time
import glob
import queue
import shutil
import threading
import subprocess
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

import numpy as np
import cv2
from PIL import Image, ImageTk
import serial

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_NCNN_PATH = os.path.join(MODEL_DIR, "best_ncnn_model")   # thư mục NCNN
YOLO_PT_PATH = os.path.join(MODEL_DIR, "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")             # dự phòng
PATCHNET_ONNX = os.path.join(MODEL_DIR, "patchnet.onnx")
BANK_DIR = os.path.join(BASE_DIR, "banks")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

# --- Camera -------------------------------------------------
# source: số thứ tự /dev/videoN hoặc đường dẫn cố định (khuyến nghị):
#   "/dev/v4l/by-id/usb-XXXX-video-index0"
# Xem danh sách: v4l2-ctl --list-devices   (trên Pi, cam thứ 2 thường là 2, 4...)
# Chỉ dùng 1 camera: xóa bớt 1 phần tử trong danh sách.
# v4l2_ctrls: khóa exposure / white balance (tên control tùy camera, xem bằng
#   v4l2-ctl -d /dev/video0 --list-ctrls). Ví dụ:
#   ["auto_exposure=1", "exposure_time_absolute=150", "white_balance_automatic=0"]
CAMERAS = [
    {"name": "CAM 1", "key": "cam1", "source": 0, "v4l2_ctrls": []},
    {"name": "CAM 2", "key": "cam2", "source": 2, "v4l2_ctrls": []},
]
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 15

# --- YOLO (chỉ để ĐỊNH VỊ) ----------------------------------
YOLO_IMGSZ = 320               # phải bằng imgsz lúc export NCNN
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
ALLOWED_CLASSES = None         # None = nhận mọi class; hoặc {"tip"}
ZONE_X_MIN, ZONE_X_MAX = 0.01, 0.95
ZONE_Y_MIN, ZONE_Y_MAX = 0.01, 0.95

# --- ROI ----------------------------------------------------
# "box"   : cửa sổ vuông quanh tâm box YOLO, cạnh = cạnh lớn nhất của box * ROI_BOX_SCALE
# "window": cửa sổ vuông CỐ ĐỊNH ROI_WINDOW px quanh tâm box (ổn định hơn nếu YOLO
#           được train để detect riêng Tip)
ROI_MODE = "box"
ROI_BOX_SCALE = 1.15
ROI_WINDOW = 160

# --- Đặc trưng patch ---------------------------------------
EMBED_SIZE = 224               # ảnh đưa vào backbone
GRID = 14                      # 224/16
FEAT_DIM = 160                 # 64 (layer 10) + 96 (layer 13) của MobileNetV2
ORT_THREADS = 3                # chừa 1 nhân cho giao diện/serial

# --- Ngưỡng -------------------------------------------------
THRESHOLD_MARGIN = 1.15        # ngưỡng = max(điểm LOO) * hệ số này
AUTO_CALIB_ON_ADD = True       # tự tính lại ngưỡng mỗi lần thêm mẫu (ghi đè ngưỡng tay)
MIN_SAMPLES_CALIB = 3
MAX_SAMPLES = 80

# --- Trigger / Serial --------------------------------------
TRIGGER_DELAY = 0.5            # chờ sau khi nhận TRIG (cơ khí ổn định)
MIN_TRIGGER_INTERVAL = 1.0
FRESH_TIMEOUT = 1.5            # chờ ảnh mới tối đa (s), quá thời gian -> NG
SERIAL_PORT = "auto"           # "auto" hoặc "/dev/ttyUSB0" / "/dev/ttyACM0"
SERIAL_BAUD = 115200
SEND_SERIAL_ON_MANUAL_TEST = False   # nút "Test" có gửi OK/NG cho Arduino không

# --- Lưu ảnh ------------------------------------------------
SAVE_NG = True
SAVE_OK = False
MIN_FREE_MB = 500              # dưới mức trống này thì ngừng lưu ảnh

# --- Giao diện ----------------------------------------------
PREVIEW_INTERVAL_MS = 100
FULLSCREEN = False             # F11 để bật/tắt
BG_DARK = "#3c3f41"
BG_PANEL = "#e8e8e8"
BG_IMAGE = "#2b2b2b"
COLOR_OK = (0, 200, 0)
COLOR_NG = (0, 0, 255)


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def in_zone(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


def crop_square(frame, cx, cy, side):
    """Cắt cửa sổ vuông tâm (cx,cy), cạnh `side`; phần ngoài khung được đệm đen.
    Trả về (ảnh_vuông, (x0, y0, side)) hoặc None."""
    side = max(32, int(round(side)))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    H, W = frame.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + side), min(H, y0 + side)
    if xb <= xa or yb <= ya:
        return None
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[ya - y0:yb - y0, xa - x0:xb - x0] = frame[ya:yb, xa:xb]
    return canvas, (x0, y0, side)


def make_roi(frame, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if ROI_MODE == "window":
        side = ROI_WINDOW
    else:
        side = max(x2 - x1, y2 - y1) * ROI_BOX_SCALE
    return crop_square(frame, cx, cy, side)


def patch_dists(q, bank, bank_sq):
    """Khoảng cách Euclid nhỏ nhất từ mỗi patch của q [P,C] tới ngân hàng [M,C].
    Trả về vector [P]."""
    q_sq = (q * q).sum(1, keepdims=True)
    d2 = q_sq + bank_sq[None, :] - 2.0 * (q @ bank.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2.min(axis=1))


def overlay_heat(display, heat, thr, x0, y0, side):
    """Tô màu các vùng có khoảng cách >= 0.7*ngưỡng lên ảnh kết quả."""
    if thr <= 0:
        return
    H, W = display.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + side), min(H, y0 + side)
    if xb <= xa or yb <= ya:
        return
    up = cv2.resize(heat.astype(np.float32), (side, side), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(up / (thr * 1.5), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = up >= 0.7 * thr
    sub = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    region = display[ya:yb, xa:xb]
    blended = cv2.addWeighted(region, 0.5, color[sub], 0.5, 0)
    m = mask[sub]
    region[m] = blended[m]


def put_text(img, text, org, color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def placeholder(text, w=640, h=360):
    img = np.full((h, w, 3), 35, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2, cv2.LINE_AA)
    return img


# ============================================================
# BACKBONE ĐẶC TRƯNG PATCH (MobileNetV2 cắt cụt, stride 16)
# ============================================================
def build_patchnet():
    """Dựng mạng: MobileNetV2.features[:14], lấy đầu ra tại layer 10 (64ch) và
    13 (96ch), ghép lại (160ch, 14x14) rồi gộp trung bình 3x3 (ngữ cảnh lân cận)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except ImportError:
        from torchvision.models import mobilenet_v2
        m = mobilenet_v2(pretrained=True)

    class PatchNet(nn.Module):
        def __init__(self, feats):
            super().__init__()
            self.feats = feats

        def forward(self, x):
            outs = []
            for i, layer in enumerate(self.feats):
                x = layer(x)
                if i in (10, 13):
                    outs.append(x)
            return F.avg_pool2d(torch.cat(outs, 1), 3, 1, 1)

    return PatchNet(m.features[:14]).eval()


def export_onnx():
    import torch
    os.makedirs(MODEL_DIR, exist_ok=True)
    net = build_patchnet()
    dummy = torch.zeros(1, 3, EMBED_SIZE, EMBED_SIZE)
    kw = dict(input_names=["input"], output_names=["feat"], opset_version=13)
    try:
        torch.onnx.export(net, dummy, PATCHNET_ONNX, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, dummy, PATCHNET_ONNX, **kw)
    print(f"Đã xuất {PATCHNET_ONNX}")


class PatchExtractor:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self):
        self.backend = None
        if os.path.exists(PATCHNET_ONNX):
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = ORT_THREADS
                self.sess = ort.InferenceSession(
                    PATCHNET_ONNX, so, providers=["CPUExecutionProvider"])
                self.in_name = self.sess.get_inputs()[0].name
                self.backend = "onnx"
            except Exception as e:
                print(f"⚠️ Không nạp được ONNX ({e}), chuyển sang PyTorch")
        if self.backend is None:
            import torch
            torch.set_num_threads(ORT_THREADS)
            self.torch = torch
            self.net = build_patchnet()
            self.backend = "torch"
        print(f"✅ Backbone đặc trưng: {self.backend}")
        self.extract(np.zeros((EMBED_SIZE, EMBED_SIZE, 3), np.uint8))   # warm-up

    def _prep(self, bgr):
        interp = cv2.INTER_AREA if bgr.shape[0] > EMBED_SIZE else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(bgr, (EMBED_SIZE, EMBED_SIZE), interpolation=interp),
                           cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def extract(self, bgr):
        """Trả về [GRID*GRID, FEAT_DIM] float32."""
        x = self._prep(bgr)
        if self.backend == "onnx":
            out = self.sess.run(None, {self.in_name: x})[0]
        else:
            with self.torch.no_grad():
                out = self.net(self.torch.from_numpy(x)).numpy()
        out = out[0]                                   # [C,G,G]
        c, g, _ = out.shape
        return np.ascontiguousarray(out.transpose(1, 2, 0).reshape(g * g, c), dtype=np.float32)


# ============================================================
# CAMERA (luồng đọc nền, luôn giữ khung hình mới nhất)
# ============================================================
class Camera:
    def __init__(self, name, source, ctrls):
        self.name = name
        self.source = source
        self.ctrls = ctrls or []
        self.cap = None
        self.frame = None
        self.frame_ts = 0.0
        self.connected = False
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _open(self):
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, api)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.read()                       # bắt đầu stream rồi mới áp control
        self._apply_ctrls()
        return cap

    def _apply_ctrls(self):
        if not self.ctrls:
            return
        dev = self.source if isinstance(self.source, str) else f"/dev/video{self.source}"
        for c in self.ctrls:
            try:
                subprocess.run(["v4l2-ctl", "-d", dev, "-c", c],
                               check=False, timeout=3, capture_output=True)
            except Exception as e:
                print(f"⚠️ {self.name}: không áp được '{c}': {e}")

    def _loop(self):
        fails = 0
        while self.running:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    self.connected = False
                    time.sleep(1.0)
                    continue
                fails = 0
            ok, frame = self.cap.read()
            if ok and frame is not None:
                fails = 0
                self.connected = True
                with self.lock:
                    self.frame, self.frame_ts = frame, time.time()
            else:
                fails += 1
                if fails > 30:
                    self.connected = False
                    self.cap.release()
                    self.cap = None
                else:
                    time.sleep(0.02)
        if self.cap is not None:
            self.cap.release()

    def latest(self):
        """Khung hình mới nhất; None nếu không có hoặc đã cũ hơn 2 giây."""
        with self.lock:
            if self.frame is None or time.time() - self.frame_ts > 2.0:
                return None
            return self.frame

    def latest_ts(self):
        with self.lock:
            return self.frame_ts

    def wait_fresh(self, t_ref, timeout):
        """Đợi khung hình được chụp SAU thời điểm t_ref. Hết hạn -> None."""
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.frame is not None and self.frame_ts > t_ref:
                    return self.frame
            time.sleep(0.01)
        return None


# ============================================================
# MỖI CAMERA: NGÂN HÀNG MẪU OK + NGƯỠNG
# ============================================================
class CamUnit:
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.key = cfg["key"]
        self.cam = Camera(self.name, cfg["source"], cfg.get("v4l2_ctrls"))
        self.bank_path = os.path.join(BANK_DIR, f"bank_{self.key}.npy")
        self.samples = np.zeros((0, GRID * GRID, FEAT_DIM), np.float32)
        self.threshold = 0.0
        self.flat = np.zeros((0, FEAT_DIM), np.float32)
        self.flat_sq = np.zeros((0,), np.float32)
        self.last_shown_ts = 0.0
        # widget (do App gán)
        self.live_view = self.res_view = None
        self.count_var = self.score_var = self.thr_var = self.cam_var = None

    def load(self, threshold):
        self.threshold = float(threshold)
        if os.path.exists(self.bank_path):
            try:
                arr = np.load(self.bank_path)
                if arr.ndim == 3 and arr.shape[1:] == (GRID * GRID, FEAT_DIM):
                    self.samples = arr.astype(np.float32)
                else:
                    print(f"⚠️ {self.name}: ngân hàng mẫu không khớp cấu hình, bỏ qua")
            except Exception as e:
                print(f"⚠️ {self.name}: lỗi đọc ngân hàng mẫu: {e}")
        self.rebuild()

    def rebuild(self):
        self.flat = np.ascontiguousarray(self.samples.reshape(-1, FEAT_DIM), dtype=np.float32)
        self.flat_sq = (self.flat * self.flat).sum(1)

    def save_bank(self):
        os.makedirs(BANK_DIR, exist_ok=True)
        np.save(self.bank_path, self.samples)

    def add(self, feat):
        self.samples = np.concatenate([self.samples, feat[None]], axis=0)
        self.rebuild()
        self.save_bank()

    def clear(self):
        self.samples = np.zeros((0, GRID * GRID, FEAT_DIM), np.float32)
        self.rebuild()
        self.save_bank()

    def score(self, feat):
        d = patch_dists(feat, self.flat, self.flat_sq)
        return float(d.max()), d

    def calibrate(self):
        """Leave-one-out: mỗi mẫu OK được chấm điểm so với các mẫu còn lại.
        Trả về (ngưỡng, danh_sách_điểm) hoặc None nếu chưa đủ mẫu."""
        n = len(self.samples)
        if n < MIN_SAMPLES_CALIB:
            return None
        scores = []
        for i in range(n):
            others = np.concatenate([self.samples[:i], self.samples[i + 1:]]).reshape(-1, FEAT_DIM)
            sq = (others * others).sum(1)
            scores.append(float(patch_dists(self.samples[i], others, sq).max()))
        return max(scores) * THRESHOLD_MARGIN, scores


# ============================================================
# HIỂN THỊ ẢNH (Canvas, không làm thay đổi kích thước layout)
# ============================================================
class ImageView:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, bg=BG_IMAGE, highlightthickness=0, width=10, height=10)
        self.w = self.h = 10
        self.item = None
        self.tk_img = None
        self.canvas.bind("<Configure>", self._on_cfg)

    def _on_cfg(self, e):
        self.w, self.h = e.width, e.height

    def show(self, bgr):
        if bgr is None or self.w <= 2 or self.h <= 2:
            return
        ih, iw = bgr.shape[:2]
        s = min(self.w / iw, self.h / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        cx, cy = self.w // 2, self.h // 2
        if self.item is None:
            self.item = self.canvas.create_image(cx, cy, image=img)
        else:
            self.canvas.coords(self.item, cx, cy)
            self.canvas.itemconfig(self.item, image=img)
        self.tk_img = img


# ============================================================
# ỨNG DỤNG CHÍNH
# ============================================================
class MachineVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Vision - Kiểm tra Tip (2 camera)")
        self.root.geometry("1280x720")
        self.root.configure(bg=BG_DARK)
        if FULLSCREEN:
            self.root.attributes("-fullscreen", True)
        self.root.bind("<F11>", lambda e: self.root.attributes(
            "-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.ui_queue = queue.Queue()
        self.infer_lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.serial_port = None
        self.last_serial_try = 0.0
        self.busy = False
        self.models_ready = False
        self.yolo = None
        self.extractor = None
        self.last_trigger = 0.0
        self.ok_count = 0
        self.ng_count = 0

        self.units = [CamUnit(c) for c in CAMERAS]
        settings = self.load_settings()
        for u in self.units:
            u.load(settings.get(u.key, 0.0))

        self.build_ui()
        for u in self.units:
            u.cam.start()
            self.refresh_unit_ui(u)

        self.status("Đang tải model (YOLO + backbone)...")
        threading.Thread(target=self.load_models, daemon=True).start()
        self.root.after(300, self.connect_serial)

        self._poll_ui()
        self.update_previews()
        self.check_serial()

    # ------------------------------------------------------
    # Settings
    # ------------------------------------------------------
    def load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({u.key: u.threshold for u in self.units}, f, indent=2)
        except Exception as e:
            print(f"⚠️ Lỗi lưu settings: {e}")

    # ------------------------------------------------------
    # Giao diện
    # ------------------------------------------------------
    def build_ui(self):
        r = self.root
        r.grid_columnconfigure(0, weight=0, minsize=190)
        for i in range(len(self.units)):
            r.grid_columnconfigure(i + 1, weight=1, uniform="cam")
        r.grid_rowconfigure(0, weight=1)

        f_btn = tkfont.Font(family="Arial", size=9)
        f_sec = tkfont.Font(family="Arial", size=8, weight="bold")
        f_small = tkfont.Font(family="Arial", size=8)

        # ---- Panel trái ----
        panel = tk.Frame(r, bg=BG_PANEL, width=190)
        panel.grid(row=0, column=0, sticky="ns")
        panel.grid_propagate(False)

        def section(text):
            tk.Label(panel, text=text, font=f_sec, bg=BG_PANEL, fg="#333").pack(
                anchor="w", padx=6, pady=(8, 2))

        section("HỆ THỐNG")
        self.sys_label = tk.Label(panel, text="Đang tải model...", bg=BG_PANEL, font=f_small,
                                  fg="#a60", wraplength=175, justify="left")
        self.sys_label.pack(anchor="w", padx=6)

        section("SERIAL")
        self.serial_label = tk.Label(panel, text="Chưa kết nối", bg=BG_PANEL, font=f_small,
                                     fg="#a00", wraplength=175, justify="left")
        self.serial_label.pack(anchor="w", padx=6)
        tk.Button(panel, text="Kết nối lại", font=f_btn,
                  command=self.connect_serial).pack(fill="x", padx=6, pady=2)

        section("KẾT QUẢ")
        self.big_label = tk.Label(panel, text="--", font=tkfont.Font(family="Arial", size=28, weight="bold"),
                                  bg=BG_PANEL, fg="#666")
        self.big_label.pack(pady=(0, 0))
        self.time_label = tk.Label(panel, text="", bg=BG_PANEL, font=f_small, fg="#333")
        self.time_label.pack()

        section("THỐNG KÊ")
        self.ok_label = tk.Label(panel, text="OK: 0", bg=BG_PANEL, fg="green",
                                 font=tkfont.Font(size=11, weight="bold"))
        self.ok_label.pack(anchor="w", padx=12)
        self.ng_label = tk.Label(panel, text="NG: 0", bg=BG_PANEL, fg="red",
                                 font=tkfont.Font(size=11, weight="bold"))
        self.ng_label.pack(anchor="w", padx=12)
        tk.Button(panel, text="Reset Counter", font=f_btn,
                  command=self.reset_count).pack(fill="x", padx=6, pady=4)

        tk.Button(panel, text="Test chụp 1 lần", font=f_btn, bg="#cfe8ff",
                  command=lambda: self.trigger("button")).pack(fill="x", padx=6, pady=(10, 2))

        # ---- Mỗi camera 1 cột ----
        for i, u in enumerate(self.units):
            col = tk.Frame(r, bg="black")
            col.grid(row=0, column=i + 1, sticky="nsew", padx=(4, 2), pady=4)
            col.grid_columnconfigure(0, weight=1)
            col.grid_rowconfigure(2, weight=1)
            col.grid_rowconfigure(4, weight=1)

            head = tk.Frame(col, bg="black")
            head.grid(row=0, column=0, sticky="ew")
            tk.Label(head, text=u.name, bg="black", fg="white",
                     font=tkfont.Font(size=11, weight="bold")).pack(side="left", padx=6, pady=2)
            u.cam_var = tk.StringVar(value="...")
            tk.Label(head, textvariable=u.cam_var, bg="black", fg="#aaa", font=f_small).pack(side="right", padx=6)

            tk.Label(col, text="LIVE", bg="black", fg="#888", font=f_small).grid(row=1, column=0, sticky="w", padx=6)
            u.live_view = ImageView(col)
            u.live_view.canvas.grid(row=2, column=0, sticky="nsew", padx=3, pady=1)

            tk.Label(col, text="KẾT QUẢ", bg="black", fg="#0f0", font=f_small).grid(row=3, column=0, sticky="w", padx=6)
            u.res_view = ImageView(col)
            u.res_view.canvas.grid(row=4, column=0, sticky="nsew", padx=3, pady=1)

            ctl = tk.Frame(col, bg=BG_PANEL)
            ctl.grid(row=5, column=0, sticky="ew", padx=3, pady=3)
            u.count_var = tk.StringVar(value="Đã học: 0 mẫu")
            u.score_var = tk.StringVar(value="Điểm: --")
            tk.Label(ctl, textvariable=u.count_var, bg=BG_PANEL, font=f_small).grid(row=0, column=0, sticky="w", padx=4)
            tk.Label(ctl, textvariable=u.score_var, bg=BG_PANEL, font=f_small, fg="#036").grid(
                row=0, column=1, columnspan=3, sticky="w", padx=4)

            tk.Button(ctl, text="+ Mẫu OK", font=f_btn, bg="#c8f7c5",
                      command=lambda un=u: self.add_sample(un)).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
            tk.Button(ctl, text="Xóa mẫu", font=f_btn, bg="#f7c5c5",
                      command=lambda un=u: self.clear_samples(un)).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
            tk.Button(ctl, text="Tính ngưỡng", font=f_btn,
                      command=lambda un=u: self.calibrate_unit(un)).grid(row=1, column=2, sticky="ew", padx=2, pady=2)

            tk.Label(ctl, text="Ngưỡng:", bg=BG_PANEL, font=f_small).grid(row=2, column=0, sticky="e", padx=2)
            u.thr_var = tk.StringVar(value=f"{u.threshold:.3f}")
            sp = ttk.Spinbox(ctl, from_=0.0, to=999.0, increment=0.05, width=8,
                             textvariable=u.thr_var)
            sp.grid(row=2, column=1, sticky="w", padx=2)
            u.thr_var.trace_add("write", lambda *a, un=u: self._on_thr_change(un))
            for c in range(3):
                ctl.grid_columnconfigure(c, weight=1)

        # ---- Thanh trạng thái ----
        self.status_var = tk.StringVar(value="")
        tk.Label(r, textvariable=self.status_var, bg="#2b2b2b", fg="#ddd", anchor="w",
                 font=f_small).grid(row=1, column=0, columnspan=len(self.units) + 1, sticky="ew")

    def _on_thr_change(self, unit):
        try:
            v = float(unit.thr_var.get())
        except (ValueError, tk.TclError):
            return
        unit.threshold = max(0.0, v)
        self.save_settings()

    def refresh_unit_ui(self, unit):
        unit.count_var.set(f"Đã học: {len(unit.samples)} mẫu")
        cur = unit.thr_var.get()
        new = f"{unit.threshold:.3f}"
        try:
            if abs(float(cur) - unit.threshold) > 1e-9:
                unit.thr_var.set(new)
        except ValueError:
            unit.thr_var.set(new)

    def status(self, msg):
        print(msg)
        self.ui_call(self.status_var.set, msg)

    # ------------------------------------------------------
    # Hàng đợi gọi giao diện từ luồng khác
    # ------------------------------------------------------
    def ui_call(self, fn, *args):
        self.ui_queue.put((fn, args))

    def _poll_ui(self):
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(30, self._poll_ui)

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------
    def update_previews(self):
        for u in self.units:
            frame = u.cam.latest()
            if frame is None:
                u.cam_var.set("MẤT TÍN HIỆU")
                if u.last_shown_ts != -1:
                    u.live_view.show(placeholder("NO SIGNAL"))
                    u.last_shown_ts = -1
            else:
                u.cam_var.set("OK")
                ts = u.cam.latest_ts()
                if ts != u.last_shown_ts:
                    u.live_view.show(frame)
                    u.last_shown_ts = ts
        self.root.after(PREVIEW_INTERVAL_MS, self.update_previews)

    # ------------------------------------------------------
    # Nạp model
    # ------------------------------------------------------
    def load_models(self):
        try:
            from ultralytics import YOLO
            if os.path.isdir(YOLO_NCNN_PATH):
                path = YOLO_NCNN_PATH
            elif os.path.exists(YOLO_PT_PATH):
                path = YOLO_PT_PATH
                print("⚠️ Không có model NCNN, dùng best.pt (chậm hơn)")
            else:
                raise FileNotFoundError(f"Không thấy YOLO tại {YOLO_NCNN_PATH} hoặc {YOLO_PT_PATH}")
            self.yolo = YOLO(path, task="detect")
            self.yolo.predict(np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8),
                              imgsz=YOLO_IMGSZ, verbose=False)      # warm-up
            print(f"✅ Đã nạp YOLO: {path}")

            self.extractor = PatchExtractor()
            self.models_ready = True
            self.ui_call(self.sys_label.config,
                         {"text": f"Model sẵn sàng\n(backbone: {self.extractor.backend})", "fg": "#080"})
            self.status("Sẵn sàng.")
        except Exception as e:
            traceback.print_exc()
            self.ui_call(self.sys_label.config, {"text": f"LỖI MODEL: {e}", "fg": "#a00"})
            self.status(f"Lỗi nạp model: {e}")

    # ------------------------------------------------------
    # Serial
    # ------------------------------------------------------
    def find_serial_port(self):
        if SERIAL_PORT != "auto":
            return SERIAL_PORT
        cands = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        return cands[0] if cands else None

    def connect_serial(self):
        self.last_serial_try = time.time()
        try:
            with self.serial_lock:
                if self.serial_port is not None and self.serial_port.is_open:
                    self.serial_port.close()
                self.serial_port = None
                port = self.find_serial_port()
                if port is None:
                    raise RuntimeError("không thấy /dev/ttyUSB* hoặc /dev/ttyACM*")
                sp = serial.Serial(port, SERIAL_BAUD, timeout=0.05)
                time.sleep(1.8)                 # Arduino Nano tự reset khi mở cổng
                sp.reset_input_buffer()
                self.serial_port = sp
            self.serial_label.config(text=f"Đã kết nối {port}", fg="#080")
            print(f"Đã kết nối {port}")
        except Exception as e:
            self.serial_port = None
            self.serial_label.config(text=f"Lỗi: {e}", fg="#a00")

    def check_serial(self):
        sp = self.serial_port
        if sp is not None:
            try:
                if sp.is_open and sp.in_waiting > 0:
                    data = sp.readline().decode(errors="ignore").strip()
                    if data == "TRIG":
                        self.trigger("serial")
            except Exception as e:
                print(f"Lỗi Serial: {e}")
                try:
                    sp.close()
                except Exception:
                    pass
                self.serial_port = None
                self.serial_label.config(text="Mất kết nối", fg="#a00")
        elif time.time() - self.last_serial_try > 3.0:
            self.connect_serial()             # tự nối lại mỗi 3 giây
        self.root.after(20, self.check_serial)

    def send_serial(self, cmd):
        with self.serial_lock:
            sp = self.serial_port
            if sp is not None and sp.is_open:
                try:
                    sp.write((cmd + "\n").encode())
                    print(f"→ Gửi '{cmd}'")
                except Exception as e:
                    print(f"Lỗi gửi Serial: {e}")

    # ------------------------------------------------------
    # Định vị + kiểm tra
    # ------------------------------------------------------
    def locate(self, frame):
        """YOLO định vị. Trả về (box_tốt_nhất_trong_vùng | None, [box_ngoài_vùng])."""
        h, w = frame.shape[:2]
        res = self.yolo.predict(frame, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD,
                                iou=IOU_THRESHOLD, verbose=False)[0]
        best, best_conf, outs = None, -1.0, []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
                if ALLOWED_CLASSES and res.names[ci] not in ALLOWED_CLASSES:
                    continue
                box = (int(x1), int(y1), int(x2), int(y2))
                if not in_zone(*box, w, h):
                    outs.append(box)
                    continue
                if cf > best_conf:
                    best, best_conf = box, float(cf)
        return best, outs

    def inspect(self, unit, frame):
        t0 = time.time()
        r = {"cam": unit.name, "result": "NG", "reason": "", "score": None,
             "thr": unit.threshold, "raw": frame, "display": None, "ms": 0}
        if frame is None:
            r["reason"] = "không có ảnh mới từ camera"
            r["display"] = placeholder("KHONG CO ANH")
            return r

        display = frame.copy()
        h, w = display.shape[:2]
        fs = max(0.5, w / 1000.0)
        box, outs = self.locate(frame)
        for (x1, y1, x2, y2) in outs:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            put_text(display, "OUT ZONE", (x1, max(15, y1 - 8)), (0, 165, 255), fs * 0.7, 1)

        if box is None:
            r["reason"] = "không thấy sản phẩm" if not outs else "sản phẩm ngoài vùng kiểm"
        else:
            crop = make_roi(frame, box)
            if crop is None:
                r["reason"] = "ROI rỗng"
            elif len(unit.samples) == 0:
                r["reason"] = "chưa học mẫu OK"
            elif unit.threshold <= 0:
                r["reason"] = "chưa có ngưỡng (bấm 'Tính ngưỡng')"
            else:
                img, (x0, y0, side) = crop
                feat = self.extractor.extract(img)
                score, d = unit.score(feat)
                r["score"] = score
                overlay_heat(display, d.reshape(GRID, GRID), unit.threshold, x0, y0, side)
                if score <= unit.threshold:
                    r["result"] = "OK"
                    r["reason"] = f"score {score:.3f} <= {unit.threshold:.3f}"
                else:
                    r["reason"] = f"score {score:.3f} > {unit.threshold:.3f}"
            color = COLOR_OK if r["result"] == "OK" else COLOR_NG
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        color = COLOR_OK if r["result"] == "OK" else COLOR_NG
        put_text(display, f"{unit.name}: {r['result']}", (15, int(40 * fs)), color, fs * 1.4, 3)
        put_text(display, r["reason"], (15, int(70 * fs)), color, fs * 0.6, 1)
        r["display"] = display
        r["ms"] = int((time.time() - t0) * 1000)
        return r

    # ------------------------------------------------------
    # Trigger + luồng kiểm tra
    # ------------------------------------------------------
    def trigger(self, source):
        now = time.time()
        if self.busy or now - self.last_trigger < MIN_TRIGGER_INTERVAL:
            return
        self.last_trigger = now
        self.busy = True
        print(f"Nhận trigger ({source})")
        threading.Thread(target=self._inspection_worker, args=(source,), daemon=True).start()

    def _inspection_worker(self, source):
        sent = False
        try:
            t0 = time.time()
            if not self.models_ready:
                raise RuntimeError("model chưa sẵn sàng")
            time.sleep(TRIGGER_DELAY if source == "serial" else 0.1)
            t_ref = time.time()
            frames = [u.cam.wait_fresh(t_ref, FRESH_TIMEOUT) for u in self.units]
            with self.infer_lock:
                results = [self.inspect(u, f) for u, f in zip(self.units, frames)]
            final = "OK" if all(r["result"] == "OK" for r in results) else "NG"
            if source == "serial" or SEND_SERIAL_ON_MANUAL_TEST:
                self.send_serial(final)
                sent = True
            total_ms = int((time.time() - t0) * 1000)
            print(f"Kết quả: {final} | " + " | ".join(f"{r['cam']}: {r['result']} ({r['reason']})" for r in results))
            self.ui_call(self.show_results, results, final, total_ms)
            self.save_history(results, final)
        except Exception as e:
            traceback.print_exc()
            if source == "serial" and not sent:
                self.send_serial("NG")            # fail-safe
            self.status(f"Lỗi kiểm tra: {e}")
        finally:
            self.busy = False

    def show_results(self, results, final, total_ms):
        for u, r in zip(self.units, results):
            u.res_view.show(r["display"])
            if r["score"] is not None:
                u.score_var.set(f"Điểm {r['score']:.3f} / Ngưỡng {r['thr']:.3f} → {r['result']}")
            else:
                u.score_var.set(f"{r['result']}: {r['reason']}")
        if final == "OK":
            self.ok_count += 1
            self.big_label.config(text="OK", fg="#1e9e1e")
        else:
            self.ng_count += 1
            self.big_label.config(text="NG", fg="#d01010")
        self.ok_label.config(text=f"OK: {self.ok_count}")
        self.ng_label.config(text=f"NG: {self.ng_count}")
        self.time_label.config(text=f"{total_ms} ms")

    def reset_count(self):
        self.ok_count = self.ng_count = 0
        self.ok_label.config(text="OK: 0")
        self.ng_label.config(text="NG: 0")

    def save_history(self, results, final):
        if (final == "OK" and not SAVE_OK) or (final == "NG" and not SAVE_NG):
            return
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            if shutil.disk_usage(HISTORY_DIR).free < MIN_FREE_MB * 1024 * 1024:
                print("⚠️ Ổ đĩa gần đầy, ngừng lưu ảnh")
                return
            now = datetime.now()
            folder = os.path.join(HISTORY_DIR, now.strftime("%Y-%m-%d"))
            os.makedirs(folder, exist_ok=True)
            stamp = now.strftime("%H%M%S_%f")[:9]
            for r in results:
                sc = f"{r['score']:.2f}" if r["score"] is not None else "na"
                base = os.path.join(folder, f"{stamp}_{final}_{r['cam'].replace(' ', '')}_{sc}")
                if r["raw"] is not None:
                    cv2.imwrite(base + "_raw.jpg", r["raw"], [cv2.IMWRITE_JPEG_QUALITY, 92])
                if r["display"] is not None:
                    cv2.imwrite(base + "_res.jpg", r["display"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            print(f"⚠️ Lỗi lưu ảnh: {e}")

    # ------------------------------------------------------
    # Học mẫu OK / tính ngưỡng
    # ------------------------------------------------------
    def add_sample(self, unit):
        if self.busy or not self.models_ready:
            self.status("Đang bận hoặc model chưa sẵn sàng")
            return
        if len(unit.samples) >= MAX_SAMPLES:
            self.status(f"{unit.name}: đã đủ {MAX_SAMPLES} mẫu, hãy xóa bớt nếu muốn học lại")
            return
        self.busy = True
        threading.Thread(target=self._learn_worker, args=(unit,), daemon=True).start()

    def _learn_worker(self, unit):
        try:
            with self.infer_lock:
                frame = unit.cam.latest()
                if frame is None:
                    self.status(f"{unit.name}: không có ảnh từ camera")
                    return
                box, _ = self.locate(frame)
                if box is None:
                    self.status(f"{unit.name}: không thấy sản phẩm trong vùng kiểm, không lấy mẫu")
                    return
                crop = make_roi(frame, box)
                if crop is None:
                    self.status(f"{unit.name}: ROI rỗng")
                    return
                unit.add(self.extractor.extract(crop[0]))
                msg = f"{unit.name}: đã thêm mẫu OK (tổng {len(unit.samples)})"
                if AUTO_CALIB_ON_ADD and len(unit.samples) >= MIN_SAMPLES_CALIB:
                    thr, scores = unit.calibrate()
                    unit.threshold = round(thr, 3)
                    self.save_settings()
                    msg += f" | ngưỡng mới {unit.threshold:.3f} (LOO max {max(scores):.3f})"
            self.ui_call(self.refresh_unit_ui, unit)
            self.status(msg)
        except Exception as e:
            traceback.print_exc()
            self.status(f"Lỗi thêm mẫu: {e}")
        finally:
            self.busy = False

    def calibrate_unit(self, unit):
        if self.busy:
            return
        if len(unit.samples) < MIN_SAMPLES_CALIB:
            self.status(f"{unit.name}: cần ít nhất {MIN_SAMPLES_CALIB} mẫu để tính ngưỡng")
            return
        self.busy = True

        def work():
            try:
                with self.infer_lock:
                    thr, scores = unit.calibrate()
                    unit.threshold = round(thr, 3)
                    self.save_settings()
                self.ui_call(self.refresh_unit_ui, unit)
                self.status(f"{unit.name}: ngưỡng {unit.threshold:.3f} "
                            f"(LOO min {min(scores):.3f} / max {max(scores):.3f}). "
                            f"Hãy thử ảnh NG thật để kiểm chứng.")
            except Exception as e:
                traceback.print_exc()
                self.status(f"Lỗi tính ngưỡng: {e}")
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def clear_samples(self, unit):
        if self.busy:
            return
        unit.clear()
        self.refresh_unit_ui(unit)
        self.status(f"{unit.name}: đã xóa toàn bộ mẫu OK")

    # ------------------------------------------------------
    def on_close(self):
        for u in self.units:
            u.cam.stop()
        try:
            if self.serial_port is not None:
                self.serial_port.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    if "--export-onnx" in sys.argv:
        export_onnx()
        sys.exit(0)
    root = tk.Tk()
    app = MachineVisionApp(root)
    root.mainloop()