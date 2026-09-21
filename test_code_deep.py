import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from tkinter import ttk
import time
import serial
import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")

SERIAL_PORT = 'COM4'   # Arduino Nano
SERIAL_BAUD = 115200

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

ZONE_X_MIN, ZONE_X_MAX = 0.01, 0.95
ZONE_Y_MIN, ZONE_Y_MAX = 0.01, 0.95

CLASS_OK = "product_ok"
CLASS_NG = "product_ng"

TRIGGER_DELAY = 0.5            # Delay 0.5s sau khi nhận TRIG
MIN_TRIGGER_INTERVAL = 1.0     # Tối thiểu 1s giữa 2 trigger

# --- CAU HINH DEEP EMBEDDING (AI hoc mau OK ngay tren UI, kieu Keyence IV4) ---
EMBED_BANK_PATH = os.path.join(BASE_DIR, "ok_embeddings.npy")   # noi luu cac mau OK da hoc
EMBED_SIM_THRESHOLD = 0.80     # nguong cosine similarity: >= nguong -> OK, < nguong -> NG
EMBED_IMG_SIZE = 224           # kich thuoc dau vao chuan cua MobileNetV2
EMBED_DIM = 1280               # so chieu embedding dau ra cua MobileNetV2

BG_DARK = "#3c3f41"
BG_PANEL = "#e8e8e8"
BG_IMAGE = "#4a4a4a"
GREEN = "#00ff00"


def is_in_zone(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


class MachineVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Vision - YOLO Insert Check")
        self.root.geometry("1200x700")
        self.root.configure(bg=BG_DARK)

        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Không tìm thấy model tại {MODEL_PATH}")
            self.model = None
        else:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ Đã load model: {MODEL_PATH}")

        # Model deep embedding (MobileNetV2) de hoc mau OK ngay tren giao dien.
        # YOLO o tren chi dung de DINH VI san pham; viec phan loai OK/NG se dua
        # vao do tuong dong (cosine similarity) giua embedding anh moi va
        # "ngan hang mau OK" ma nguoi dung tu them qua nut "+ Them mau OK".
        self.init_embedding_model()
        self.load_embedding_bank()

        self.cap = None
        self.current_frame = None
        self.serial_port = None
        self.is_processing = False
        self.last_trigger_time = 0
        self.ok_count = 0
        self.ng_count = 0
        self.available_devices = []

        self.setup_ui()
        self.sample_label.config(text=f"Đã học: {len(self.ok_embeddings)} mẫu")
        self.search_device()

        if self.available_devices:
            self.camera_combo.current(0)
            self.open_camera()

        self.connect_serial()
        self.update_camera()
        self.check_serial()

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=0, minsize=180)  # Panel cố định
        self.root.grid_columnconfigure(1, weight=1, uniform="img")  # Camera
        self.root.grid_columnconfigure(2, weight=1, uniform="img")  # Kết quả
        self.root.grid_rowconfigure(0, weight=1)

        # =========================================================
        # PANEL ĐIỀU KHIỂN BÊN TRÁI
        # =========================================================
        panel = tk.Frame(self.root, bg=BG_PANEL, width=180)
        panel.grid(row=0, column=0, sticky="ns")
        panel.grid_propagate(False)

        btn_font = tkfont.Font(family="Arial", size=9)
        section_font = tkfont.Font(family="Arial", size=8, weight="bold")

        # --- CAMERA ---
        tk.Label(panel, text="CAMERA", font=section_font,
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=6, pady=(6, 2))

        self.camera_combo = ttk.Combobox(panel, font=btn_font, state="readonly")
        self.camera_combo.pack(fill="x", padx=6, pady=1)

        tk.Button(panel, text="Search Device", font=btn_font,
                  command=self.search_device).pack(fill="x", padx=6, pady=1)
        tk.Button(panel, text="Open Camera", font=btn_font,
                  command=self.open_camera).pack(fill="x", padx=6, pady=1)
        tk.Button(panel, text="Close Camera", font=btn_font,
                  command=self.close_camera).pack(fill="x", padx=6, pady=1)

        self.cam_status = tk.Label(panel, text="Chưa mở",
                                   bg=BG_PANEL, font=tkfont.Font(size=7), fg="#a00")
        self.cam_status.pack(anchor="w", padx=6, pady=(0, 4))

        # --- SERIAL ---
        tk.Label(panel, text="SERIAL", font=section_font,
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=6, pady=(6, 2))

        self.serial_status = tk.Label(panel, text="Chưa kết nối",
                                      bg=BG_PANEL, font=tkfont.Font(size=7), fg="#a00",
                                      wraplength=160, justify="left")
        self.serial_status.pack(anchor="w", padx=6, pady=1)

        tk.Button(panel, text="Kết nối lại", font=btn_font,
                  command=self.connect_serial).pack(fill="x", padx=6, pady=1)

        # --- HOC MAU (AI DEEP EMBEDDING) ---
        tk.Label(panel, text="HỌC MẪU (AI)", font=section_font,
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=6, pady=(6, 2))

        self.sample_label = tk.Label(panel, text="Đã học: 0 mẫu",
                                     bg=BG_PANEL, font=tkfont.Font(size=8), fg="#333")
        self.sample_label.pack(anchor="w", padx=6, pady=(0, 2))

        tk.Button(panel, text="+ Thêm mẫu OK", font=btn_font, bg="#c8f7c5",
                  command=self.add_ok_sample).pack(fill="x", padx=6, pady=1)
        tk.Button(panel, text="Xóa hết mẫu", font=btn_font, bg="#f7c5c5",
                  command=self.clear_ok_samples).pack(fill="x", padx=6, pady=1)

        # --- THỐNG KÊ ---
        tk.Label(panel, text="THỐNG KÊ", font=section_font,
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=6, pady=(6, 2))

        self.ok_label = tk.Label(panel, text="OK: 0", bg=BG_PANEL,
                                 font=tkfont.Font(size=10, weight="bold"), fg="green")
        self.ok_label.pack(anchor="w", padx=12)

        self.ng_label = tk.Label(panel, text="NG: 0", bg=BG_PANEL,
                                 font=tkfont.Font(size=10, weight="bold"), fg="red")
        self.ng_label.pack(anchor="w", padx=12)

        tk.Button(panel, text="Reset Counter", font=btn_font,
                  command=self.reset_count).pack(fill="x", padx=6, pady=6)

        tk.Button(panel, text="Test chụp 1 lần", font=btn_font, bg="#cfe8ff",
                  command=self.trigger_once).pack(fill="x", padx=6, pady=1)

        # =========================================================
        # KHUNG CAMERA (bên phải, 50% chiều rộng còn lại)
        # =========================================================
        cam_frame = tk.Frame(self.root, bg="black")
        cam_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        cam_frame.grid_rowconfigure(1, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        tk.Label(cam_frame, text="CAMERA (RAW)", font=tkfont.Font(size=12, weight="bold"),
                 bg="black", fg="white").grid(row=0, column=0, sticky="ew", pady=4)

        # Label ảnh: dùng grid + sticky để tự giãn full khung
        self.cam_label = tk.Label(cam_frame, bg=BG_IMAGE)
        self.cam_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # =========================================================
        # KHUNG KẾT QUẢ (bên phải, 50% chiều rộng còn lại)
        # =========================================================
        res_frame = tk.Frame(self.root, bg=BG_IMAGE)
        res_frame.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)
        res_frame.grid_rowconfigure(1, weight=1)
        res_frame.grid_columnconfigure(0, weight=1)

        tk.Label(res_frame, text="KẾT QUẢ YOLO", font=tkfont.Font(size=12, weight="bold"),
                 bg=BG_IMAGE, fg=GREEN).grid(row=0, column=0, sticky="ew", pady=4)

        self.res_label = tk.Label(res_frame, bg=BG_IMAGE)
        self.res_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    def search_device(self):
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(i)
                cap.release()
        self.available_devices = found
        if found:
            self.camera_combo["values"] = [f"Camera {i}" for i in found]
            self.camera_combo.current(0)
            print(f"Tìm thấy {len(found)} camera: {found}")

    def open_camera(self):
        if not self.available_devices:
            return
        idx = self.available_devices[self.camera_combo.current()]
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if self.cap.isOpened():
            self.cam_status.config(text=f"Đã mở Camera {idx}", fg="#080")

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.cam_status.config(text="Đã đóng", fg="#a00")

    def update_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.show_image(self.cam_label, frame)
        self.root.after(30, self.update_camera)

    def show_image(self, widget, img):
        widget.update_idletasks()
        w = widget.winfo_width()
        h = widget.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 500, 400
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        yo, xo = (h - nh) // 2, (w - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        widget.imgtk = imgtk
        widget.configure(image=imgtk)

    def connect_serial(self):
        try:
            if self.serial_port is not None and self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
            self.serial_status.config(text=f"Đã kết nối {SERIAL_PORT}", fg="#080")
            print(f"Đã kết nối {SERIAL_PORT}")
        except Exception as e:
            self.serial_port = None
            self.serial_status.config(text=f"Lỗi: {e}", fg="#a00")
            print(f"Không kết nối được {SERIAL_PORT}: {e}")

    def check_serial(self):
        if self.serial_port is not None and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.readline().decode(errors='ignore').strip()
                    if data == "TRIG":
                        now = time.time()
                        if not self.is_processing and (now - self.last_trigger_time) > MIN_TRIGGER_INTERVAL:
                            self.last_trigger_time = now
                            self.is_processing = True
                            print(f"Nhận TRIG, chờ {TRIGGER_DELAY}s rồi chụp...")
                            self.root.after(int(TRIGGER_DELAY * 1000), self.trigger_once)
            except Exception as e:
                print(f"Lỗi Serial: {e}")
                try:
                    self.serial_port.close()
                except:
                    pass
                self.serial_port = None
                self.serial_status.config(text="Mất kết nối", fg="#a00")
        self.root.after(20, self.check_serial)

    def send_serial(self, cmd):
        if self.serial_port is not None and self.serial_port.is_open:
            try:
                self.serial_port.write((cmd + '\n').encode())
                print(f"→ Gửi '{cmd}'")
            except Exception as e:
                print(f"Lỗi gửi Serial: {e}")

    def trigger_once(self):
        try:
            if self.cap is None or self.current_frame is None:
                self.is_processing = False
                return
            frame = self.current_frame.copy()
            if self.model is None:
                self.is_processing = False
                return

            display = frame.copy()
            h, w = frame.shape[:2]

            # YOLO o day CHI dung de DINH VI san pham (khong con quyet dinh OK/NG
            # theo class product_ok/product_ng nua). Chi lay 1 box duy nhat -
            # box co confidence cao nhat nam trong vung kiem tra - de tranh
            # bat nham khi co 2 san pham lot vao khung cung luc.
            results = self.model(frame, imgsz=320, conf=CONF_THRESHOLD,
                                 iou=IOU_THRESHOLD, verbose=False)
            boxes = results[0].boxes

            best_box, best_conf = None, -1.0
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_in_zone(x1, y1, x2, y2, w, h):
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(display, "OUT ZONE", (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

            if len(boxes) == 0:
                final_result, reason = "NG", "khong phat hien san pham"
                cv2.putText(display, "KHONG PHAT HIEN", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif best_box is None:
                final_result, reason = "NG", "san pham ngoai vung kiem tra"
            else:
                x1, y1, x2, y2 = best_box
                pad = max(10, int(0.06 * max(x2 - x1, y2 - y1)))
                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
                roi = frame[y1p:y2p, x1p:x2p]

                if roi.size == 0:
                    final_result, reason = "NG", "roi rong"
                else:
                    # Phan loai OK/NG bang deep embedding, khong dung class cua YOLO
                    final_result, reason = self.classify_by_embedding(roi)

                color = (0, 255, 0) if final_result == "OK" else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{final_result} ({reason})", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            print(f"Ly do phan loai: {reason}")

            if final_result == "OK":
                self.ok_count += 1
                color_text = (0, 255, 0)
            else:
                self.ng_count += 1
                color_text = (0, 0, 255)

            cv2.putText(display, final_result, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_text, 3)

            self.ok_label.config(text=f"OK: {self.ok_count}")
            self.ng_label.config(text=f"NG: {self.ng_count}")
            self.show_image(self.res_label, display)

            self.send_serial(final_result)
            print(f"Kết quả: {final_result} | OK={self.ok_count} NG={self.ng_count}")

        finally:
            self.is_processing = False

    def reset_count(self):
        self.ok_count = 0
        self.ng_count = 0
        self.ok_label.config(text="OK: 0")
        self.ng_label.config(text="NG: 0")

    # =========================================================
    # DEEP EMBEDDING - hoc mau OK ngay tren UI (kieu Keyence IV4)
    # =========================================================
    def init_embedding_model(self):
        """Nap MobileNetV2 pretrained tren ImageNet, bo lop classifier,
        chi giu phan 'features' de trich embedding (vector dac trung sau)."""
        try:
            try:
                from torchvision.models import MobileNet_V2_Weights
                backbone_full = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            except ImportError:
                # torchvision cu khong co API 'weights' -> dung API 'pretrained'
                backbone_full = mobilenet_v2(pretrained=True)

            self.embed_backbone = backbone_full.features
            self.embed_backbone.eval()
            self.embed_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((EMBED_IMG_SIZE, EMBED_IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("✅ Đã load model embedding MobileNetV2 (deep feature)")
        except Exception as e:
            self.embed_backbone = None
            self.embed_transform = None
            print(f"⚠️ Không load được model embedding MobileNetV2: {e}")

    def load_embedding_bank(self):
        """Nap ngan hang mau OK da hoc tu lan chay truoc (neu co), giup
        khong bi mat mau moi khi tat/bat lai chuong trinh."""
        if os.path.exists(EMBED_BANK_PATH):
            try:
                self.ok_embeddings = np.load(EMBED_BANK_PATH)
                print(f"✅ Đã nạp {len(self.ok_embeddings)} mẫu OK đã học trước đó")
                return
            except Exception as e:
                print(f"⚠️ Lỗi đọc ngân hàng mẫu cũ, tạo mới: {e}")
        self.ok_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def save_embedding_bank(self):
        try:
            np.save(EMBED_BANK_PATH, self.ok_embeddings)
        except Exception as e:
            print(f"⚠️ Lỗi lưu ngân hàng mẫu: {e}")

    def get_embedding(self, roi_bgr):
        """Trich vector embedding (da chuan hoa L2) tu 1 anh ROI (BGR)."""
        if self.embed_backbone is None:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.embed_transform(rgb).unsqueeze(0)
        with torch.no_grad():
            feat = self.embed_backbone(tensor)                       # [1, 1280, H, W]
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)  # [1, 1280, 1, 1]
            feat = feat.flatten(1)                                   # [1, 1280]
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)   # L2 normalize
        return feat.squeeze(0).numpy()

    def _get_best_roi_in_zone(self, frame):
        """Chay YOLO chi de DINH VI: tra ve ROI (da padding) cua box co confidence
        cao nhat nam trong vung kiem tra. Chi lay DUY NHAT 1 san pham, tranh
        bat nham khi co 2 vat lot vao khung cung luc."""
        if self.model is None:
            return None
        results = self.model(frame, imgsz=320, conf=CONF_THRESHOLD,
                             iou=IOU_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        h, w = frame.shape[:2]
        best_box, best_conf = None, -1.0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_zone(x1, y1, x2, y2, w, h):
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        pad = max(10, int(0.06 * max(x2 - x1, y2 - y1)))
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        roi = frame[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            return None
        return roi, best_box

    def add_ok_sample(self):
        """Nut '+ Them mau OK': lay san pham dang trong khung, trich embedding,
        them vao ngan hang mau OK va luu xuong file de dung lau dai."""
        if self.current_frame is None:
            print("Chưa có ảnh từ camera")
            return
        if self.embed_backbone is None:
            print("⚠️ Model embedding chưa sẵn sàng, không thể thêm mẫu")
            return

        frame = self.current_frame.copy()
        found = self._get_best_roi_in_zone(frame)
        if found is None:
            print("Không phát hiện sản phẩm trong vùng kiểm tra để lấy mẫu")
            return

        roi, _ = found
        emb = self.get_embedding(roi)
        if emb is None:
            return

        self.ok_embeddings = np.vstack([self.ok_embeddings, emb[None, :]])
        self.save_embedding_bank()
        self.sample_label.config(text=f"Đã học: {len(self.ok_embeddings)} mẫu")
        print(f"✅ Đã thêm mẫu OK (deep embedding). Tổng: {len(self.ok_embeddings)} mẫu")

    def clear_ok_samples(self):
        self.ok_embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self.save_embedding_bank()
        self.sample_label.config(text="Đã học: 0 mẫu")
        print("Đã xóa toàn bộ ngân hàng mẫu OK")

    def classify_by_embedding(self, roi):
        """So sanh embedding cua ROI voi toan bo ngan hang mau OK bang cosine
        similarity. Vi embedding da duoc L2-normalize, tich vo huong (dot product)
        chinh la cosine similarity. Similarity cao nhat >= nguong -> OK."""
        if self.embed_backbone is None:
            return "NG", "model embedding chua san sang"
        if len(self.ok_embeddings) == 0:
            return "NG", "chua hoc mau OK nao"

        emb = self.get_embedding(roi)
        if emb is None:
            return "NG", "khong trich duoc embedding"

        similarities = self.ok_embeddings @ emb
        best_sim = float(np.max(similarities))

        if best_sim >= EMBED_SIM_THRESHOLD:
            return "OK", f"sim={best_sim:.3f}"
        else:
            return "NG", f"sim={best_sim:.3f} < nguong {EMBED_SIM_THRESHOLD}"


if __name__ == "__main__":
    root = tk.Tk()
    app = MachineVisionApp(root)
    root.mainloop()