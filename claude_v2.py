import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

REFERENCE_PATH = "reference_stats.json"
MIN_OK_SAMPLES = 20
Z_SCORE_THRESHOLD = 2.5
DEFECT_DEPTH_THRESHOLD = 2.0

# 1. QUAN TRỌNG: Thu hẹp vùng kiểm tra (20% - 80%) để chặn các lỗi ở rìa màn hình như logo iVCam
ZONE_X_MIN, ZONE_X_MAX = 0.01, 0.95
ZONE_Y_MIN, ZONE_Y_MAX = 0.01, 0.95

# Ngưỡng màu cho sản phẩm xanh (Điều chỉnh nếu ánh sáng thay đổi)
LOWER_COLOR = np.array([70, 40, 40])
UPPER_COLOR = np.array([100, 255, 255])

# ============================================================
# BƯỚC 1: Xử lý ảnh (HYBRID: HSV Mask + Canny)
# ============================================================
def is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


def extract_aligned_contour(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return None

    # --- MẶT NẠ HSV: Chặn nền gỗ ---
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    kernel_hsv = np.ones((5, 5), np.uint8)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_hsv)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel_hsv)

    # --- XỬ LÝ CANNY ---
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    gray_blur = cv2.medianBlur(gray_clahe, 5)
    edges = cv2.Canny(gray_blur, 50, 150)

    # --- KẾT HỢP (Giao hai ảnh) ---
    edges_clean = cv2.bitwise_and(edges, edges, mask=mask_hsv)

    # --- NỐI MẠCH CONTOUR ---
    kernel_repair = np.ones((3, 3), np.uint8)
    edges_clean = cv2.dilate(edges_clean, kernel_repair, iterations=3)

    # --- TÌM CONTOUR ---
    contours, _ = cv2.findContours(edges_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    # 2. QUAN TRỌNG: Tăng diện tích lọc tối thiểu lên 350 pixel để loại bỏ chữ iVCam
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 350]
    if not valid_contours: return None

    main_contour = max(valid_contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 100: return None

    # --- XOAY ẢNH (ALIGN) ---
    rect = cv2.minAreaRect(main_contour)
    angle = rect[2]
    if angle < -45: angle = 90 + angle
    center = tuple(rect[0])
    (h, w) = roi_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_roi = cv2.warpAffine(roi_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

    # --- TÌM LẠI CONTOUR SAU XOAY ---
    gray_rot = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
    hsv_rot = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2HSV)
    mask_hsv_rot = cv2.inRange(hsv_rot, LOWER_COLOR, UPPER_COLOR)
    mask_hsv_rot = cv2.morphologyEx(mask_hsv_rot, cv2.MORPH_CLOSE, kernel_hsv)
    mask_hsv_rot = cv2.morphologyEx(mask_hsv_rot, cv2.MORPH_OPEN, kernel_hsv)

    edges_rot = cv2.Canny(gray_rot, 50, 150)
    edges_rot_clean = cv2.bitwise_and(edges_rot, edges_rot, mask=mask_hsv_rot)
    edges_rot_clean = cv2.dilate(edges_rot_clean, kernel_repair, iterations=3)

    c_rot, _ = cv2.findContours(edges_rot_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not c_rot: return None
    valid_rot = [cnt for cnt in c_rot if cv2.contourArea(cnt) > 350]
    if not valid_rot: return None

    return max(valid_rot, key=cv2.contourArea)


def measure_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
# BƯỚC 2: Tính đặc trưng HÌNH HỌC
# ============================================================
def get_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = area / hull_area if hull_area > 0 else 0
    perimeter_ratio = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    norm_area = area / bbox_area if bbox_area > 0 else 0

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx_contour = M["m10"] / M["m00"]
        cy_contour = M["m01"] / M["m00"]
        cx_bbox = x + w / 2
        cy_bbox = y + h / 2
        centroid_shift = np.sqrt((cx_contour - cx_bbox) ** 2 + (cy_contour - cy_bbox) ** 2) / max(w, h)
    else:
        centroid_shift = 0.0

    defects_count = 0
    max_defect_depth = 0.0
    depth_values = []
    hull_idx = cv2.convexHull(contour, returnPoints=False)

    if hull_idx is not None and len(hull_idx) > 3:
        hull_idx = np.sort(hull_idx, axis=0)
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                scale = np.sqrt(area) if area > 0 else 1
                for i in range(defects.shape[0]):
                    _, _, _, d = defects[i, 0]
                    depth = (d / 256.0) / scale * 100
                    depth_values.append(depth)
                    defects_count += 1
                    max_defect_depth = max(max_defect_depth, depth)
        except cv2.error:
            pass

    return {
        "area": float(area),
        "norm_area": float(norm_area),
        "centroid_shift": float(centroid_shift),
        "solidity": float(solidity),
        "perimeter_ratio": float(perimeter_ratio),
        "defects_count": int(defects_count),
        "max_defect_depth": float(max_defect_depth),
        "mean_defect_depth": float(np.mean(depth_values)) if depth_values else 0.0,
    }


# ============================================================
# BƯỚC 3: Quản lý bộ thống kê (Z-Score)
# ============================================================
def load_reference():
    if os.path.exists(REFERENCE_PATH):
        with open(REFERENCE_PATH, "r") as f:
            return json.load(f)
    return {"samples": []}

def save_reference(ref):
    with open(REFERENCE_PATH, "w") as f:
        json.dump(ref, f, indent=2)

def compute_stats(ref):
    samples = ref.get("samples", [])
    if len(samples) < 2:
        return None
    stats = {}
    keys = ["norm_area", "centroid_shift", "solidity", "perimeter_ratio", "defects_count", "max_defect_depth", "mean_defect_depth"]
    for k in keys:
        # Chỉ lấy các mẫu CÓ trường này - tránh crash nếu lẫn mẫu cũ (từ phiên bản trước) thiếu trường mới
        vals = np.array([s[k] for s in samples if k in s])
        if len(vals) < 2:
            continue
        stats[k] = {"mean": float(vals.mean()), "std": float(max(vals.std(), 1e-4))}
    return stats if stats else None

def classify_by_shape(features, stats):
    if stats is None:
        return "CHƯA ĐỦ MẪU", []

    reasons = []
    MIN_STD_SOLIDITY = 0.02
    MIN_STD_PERIMETER = 0.03
    MIN_STD_DEPTH = 0.8

    if "norm_area" in stats:
        mean_norm_area = stats["norm_area"]["mean"]
        if features["norm_area"] < mean_norm_area * 0.88:
            reasons.append(f"tỷ lệ diện tích giảm {(1 - features['norm_area']/mean_norm_area)*100:.1f}%")
            return "NG", reasons

    if "centroid_shift" in stats:
        mean_shift = stats["centroid_shift"]["mean"]
        std_shift = stats["centroid_shift"]["std"]
        effective_std_shift = max(std_shift, 0.005)
        SAFE_TOLERANCE = 0.05
        if features["centroid_shift"] > (mean_shift + 3 * effective_std_shift) and \
           features["centroid_shift"] > (mean_shift + SAFE_TOLERANCE):
            reasons.append(f"khối tâm lệch {features['centroid_shift']:.3f}")

    for key in ["solidity", "perimeter_ratio"]:
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        effective_std = max(std, MIN_STD_SOLIDITY if key == "solidity" else MIN_STD_PERIMETER)
        z = abs(features[key] - mean) / effective_std
        if z > Z_SCORE_THRESHOLD:
            reasons.append(f"{key} lệch {z:.1f} lần")

    mean_depth = stats["max_defect_depth"]["mean"]
    std_depth = stats["max_defect_depth"]["std"]
    effective_std_depth = max(std_depth, MIN_STD_DEPTH)
    depth_limit = max(mean_depth + Z_SCORE_THRESHOLD * effective_std_depth, DEFECT_DEPTH_THRESHOLD)
    if features["max_defect_depth"] > depth_limit:
        reasons.append(f"vết lõm sâu {features['max_defect_depth']:.2f}")

    if "mean_defect_depth" in stats:
        mean_md = stats["mean_defect_depth"]["mean"]
        std_md = stats["mean_defect_depth"]["std"]
        if features["mean_defect_depth"] > mean_md + 2 * max(std_md, 0.3):
            reasons.append(f"độ sâu vết lõm trung bình tăng {features['mean_defect_depth']:.2f}")

    return "NG" if reasons else "OK", reasons


# ============================================================
# BƯỚC 4: Giao diện UI và Logic chính
# ============================================================
BG_DARK = "#3c3f41"
BG_PANEL = "#e8e8e8"
BG_IMAGE = "#4a4a4a"
GREEN = "#00ff00"

class MachineVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Vision Inspection System")
        self.root.geometry("1400x700")
        self.root.configure(bg=BG_DARK)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy model tại {model_path}! Vui lòng đặt file best.pt cùng thư mục.")
            model_path = "best.pt"
        self.model = YOLO(model_path)

        self.reference = load_reference()
        self.ref_stats = compute_stats(self.reference)

        self.cap = None
        self.available_devices = []
        self.continuous_running = False
        self.continuous_job = None

        self.current_frame = None
        self.ok_count = 0
        self.ng_count = 0

        self.setup_ui()
        self.search_device()      # tự quét camera khi khởi động
        if self.available_devices:
            self.device_combo.current(0)
            self.open_device()    # tự mở camera đầu tiên tìm được
        self.update_camera()
        self.refresh_sample_count()

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1, uniform="img")
        self.root.grid_columnconfigure(2, weight=1, uniform="img")
        self.root.grid_rowconfigure(0, weight=1)

        control_panel = tk.Frame(self.root, bg=BG_PANEL, width=230)
        control_panel.grid(row=0, column=0, sticky="ns")
        control_panel.grid_propagate(False)

        btn_font = tkfont.Font(family="Arial", size=10)

        self._section_label(control_panel, "Initialization")
        tk.Button(control_panel, text="Search Device", font=btn_font,
                  command=self.search_device).pack(fill="x", padx=10, pady=2)
        self.device_combo = ttk.Combobox(control_panel, font=btn_font, state="readonly")
        self.device_combo.pack(fill="x", padx=10, pady=2)
        row = tk.Frame(control_panel, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=2)
        tk.Button(row, text="Open Device", font=btn_font,
                  command=self.open_device).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Close Device", font=btn_font,
                  command=self.close_device).pack(side="left", expand=True, fill="x")
        self.device_status_label = tk.Label(control_panel, text="Trang thai: chua mo",
                                              bg=BG_PANEL, font=tkfont.Font(size=8), fg="#a00")
        self.device_status_label.pack(anchor="w", padx=10)

        self._section_label(control_panel, "Image Acquisition")
        self.mode_var = tk.StringVar(value="trigger")
        tk.Radiobutton(control_panel, text="Continuous", variable=self.mode_var, value="continuous",
                       bg=BG_PANEL, font=btn_font, command=self.set_continuous_mode).pack(anchor="w", padx=10)
        tk.Radiobutton(control_panel, text="Trigger Mode", variable=self.mode_var, value="trigger",
                       bg=BG_PANEL, font=btn_font, command=self.set_trigger_mode).pack(anchor="w", padx=10)

        self.trigger_btn = tk.Button(control_panel, text="Trigger Once (SPACE)", font=btn_font,
                                     bg="#cfe8ff", command=self.run_inspection)
        self.trigger_btn.pack(fill="x", padx=10, pady=(8, 2))

        self._section_label(control_panel, "Bo mau tham chieu (OK)")
        self.sample_count_label = tk.Label(control_panel, text="Da thu thap: 0 mau",
                                           bg=BG_PANEL, font=btn_font, fg="#333")
        self.sample_count_label.pack(anchor="w", padx=10)

        tk.Button(control_panel, text="+ Them mau OK (tu khung trai)", font=btn_font, bg="#d4f7d4",
                  command=self.add_ok_sample).pack(fill="x", padx=10, pady=(4, 2))
        tk.Button(control_panel, text="Xoa het mau", font=btn_font, bg="#f7d4d4",
                  command=self.clear_samples).pack(fill="x", padx=10, pady=2)
        tk.Button(control_panel, text="Test mau (khong luu)", font=btn_font, bg="#fff3cd",
                  command=self.test_sample_debug).pack(fill="x", padx=10, pady=(4, 2))

        self._section_label(control_panel, "Parameters")
        self._param_row(control_panel, "Z-score Threshold", str(Z_SCORE_THRESHOLD))
        self._param_row(control_panel, "Conf Threshold", "0.5")

        self._section_label(control_panel, "Statistics")
        self.ok_stat = self._param_row(control_panel, "Total OK", "0")
        self.ng_stat = self._param_row(control_panel, "Total NG", "0")
        tk.Button(control_panel, text="Reset Counter", font=btn_font,
                  command=self.reset_count).pack(fill="x", padx=10, pady=6)

        raw_frame = tk.Frame(self.root, bg="black")
        raw_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        raw_frame.grid_propagate(False)
        tk.Label(raw_frame, text="CAMERA (RAW)", font=tkfont.Font(size=12, weight="bold"),
                 bg="black", fg="white", anchor="w").pack(fill="x", padx=8, pady=4)
        self.raw_label = tk.Label(raw_frame, bg=BG_IMAGE)
        self.raw_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        result_frame = tk.Frame(self.root, bg=BG_IMAGE)
        result_frame.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)
        result_frame.grid_propagate(False)
        header = tk.Frame(result_frame, bg=BG_IMAGE)
        header.pack(fill="x", padx=8, pady=4, anchor="w")
        tk.Label(header, text="AI INSPECTION RESULT", font=tkfont.Font(size=12, weight="bold"),
                 bg=BG_IMAGE, fg=GREEN).pack(anchor="w")
        self.count_label = tk.Label(header, text="Products Detected: 0", font=tkfont.Font(size=10),
                                    bg=BG_IMAGE, fg=GREEN)
        self.count_label.pack(anchor="w")
        self.result_label_img = tk.Label(result_frame, bg=BG_IMAGE)
        self.result_label_img.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=tkfont.Font(size=9, weight="bold"),
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=8, pady=(10, 2))

    def _param_row(self, parent, label, value):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=1)
        tk.Label(row, text=label, bg=BG_PANEL, font=tkfont.Font(size=9), width=15, anchor="w").pack(side="left")
        val_label = tk.Label(row, text=value, bg="white", relief="sunken", font=tkfont.Font(size=9), anchor="w")
        val_label.pack(side="left", fill="x", expand=True)
        return val_label

    def refresh_sample_count(self):
        n = len(self.reference.get("samples", []))
        status = "" if n >= MIN_OK_SAMPLES else f"  (can toi thieu {MIN_OK_SAMPLES})"
        self.sample_count_label.config(text=f"Da thu thap: {n} mau{status}")

    def display_image(self, label_widget, cv_img):
        label_widget.update_idletasks()
        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        if w <= 1 or h <= 1: w, h = 500, 500
        img_h, img_w = cv_img.shape[:2]
        scale = min(w / img_w, h / img_h)
        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        resized = cv2.resize(cv_img, (new_w, new_h))
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        y_off, x_off = (h - new_h) // 2, (w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

    def search_device(self):
        """Quét các camera đang cắm vào máy (index 0-4), liệt kê để chọn."""
        found = []
        for i in range(2):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    found.append(i)
                cap_test.release()
        self.available_devices = found
        if found:
            self.device_combo["values"] = [f"Camera {i}" for i in found]
            print(f"Tim thay {len(found)} camera: {found}")
        else:
            self.device_combo["values"] = []
            print("Khong tim thay camera nao!")

    def open_device(self):
        """Mở camera đang được chọn trong dropdown."""
        if not self.available_devices:
            print("Chua tim thay device nao - bam Search Device truoc!")
            return
        sel = self.device_combo.current()
        if sel < 0:
            sel = 0
            self.device_combo.current(0)
        cam_index = self.available_devices[sel]

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Tắt tự động chỉnh sáng
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Set exposure thấp (Chống mờ)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.device_status_label.config(text=f"Trang thai: da mo Camera {cam_index}", fg="#080")
        self.trigger_btn.config(state="normal")
        print(f"Da mo Camera {cam_index}")

    def close_device(self):
        """Giải phóng camera - nhường quyền truy cập cho ứng dụng khác."""
        self.set_trigger_mode()  # dừng continuous mode nếu đang chạy
        self.mode_var.set("trigger")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "DEVICE CLOSED", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        self.display_image(self.raw_label, blank)
        self.device_status_label.config(text="Trang thai: da dong", fg="#a00")
        self.trigger_btn.config(state="disabled")
        print("Da dong camera.")

    def set_continuous_mode(self):
        """Tự động chạy kiểm tra liên tục theo chu kỳ, không cần bấm SPACE."""
        if self.cap is None:
            print("Chua mo camera - khong the bat Continuous mode!")
            self.mode_var.set("trigger")
            return
        self.continuous_running = True
        self._continuous_tick()
        print("Da bat Continuous mode - tu dong kiem tra moi 1.5s.")

    def set_trigger_mode(self):
        self.continuous_running = False
        if self.continuous_job is not None:
            self.root.after_cancel(self.continuous_job)
            self.continuous_job = None

    def _continuous_tick(self):
        if not self.continuous_running:
            return
        self.run_inspection()
        self.continuous_job = self.root.after(1500, self._continuous_tick)

    def update_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_image(self.raw_label, frame)
        self.root.after(30, self.update_camera)

    def _get_first_roi(self, frame):
        results = self.model(frame, imgsz=320, conf=0.6, iou=0.45, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0: return None
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]): continue
            w_box = x2 - x1
            h_box = y2 - y1
            pad = max(15, int(0.05 * max(w_box, h_box)))
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            return frame[y1p:y2p, x1p:x2p]
        return None

    def add_ok_sample(self):
        if self.current_frame is None: return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print("Khong phat hien san pham nao trong khung hinh de them mau!")
            return

        contour = extract_aligned_contour(roi)
        if contour is None:
            print("Khong tach duoc contour - Kiem tra lai mau HSV hoac anh sang!")
            return

        features = get_shape_features(contour)
        self.reference.setdefault("samples", []).append(features)
        save_reference(self.reference)
        self.ref_stats = compute_stats(self.reference)
        self.refresh_sample_count()
        print(f"Da them mau OK: {features}")

    def clear_samples(self):
        self.reference = {"samples": []}
        save_reference(self.reference)
        self.ref_stats = None
        self.refresh_sample_count()
        print("Da xoa toan bo mau tham chieu.")

    def test_sample_debug(self):
        """Test 1 sản phẩm mà KHÔNG lưu vào bộ mẫu - in đầy đủ TẤT CẢ chỉ số dùng để quyết định,
        kể cả norm_area/centroid_shift (trước đây bị thiếu trong log)."""
        if self.current_frame is None:
            return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print(">> Khong phat hien san pham nao de test!")
            return

        contour = extract_aligned_contour(roi)
        if contour is None:
            print(">> Khong tach duoc contour!")
            return

        features = get_shape_features(contour)

        if self.ref_stats is None:
            print(f">> [TEST] Chua du mau tham chieu. Dac trung do duoc: {features}")
            return

        print(f"\n===== [TEST DEBUG] =====")
        keys = ["norm_area", "centroid_shift", "solidity", "perimeter_ratio",
                "max_defect_depth", "mean_defect_depth"]
        for key in keys:
            if key not in self.ref_stats:
                continue
            mean, std = self.ref_stats[key]["mean"], self.ref_stats[key]["std"]
            z = abs(features[key] - mean) / std
            print(f"  {key:18s}: gia_tri={features[key]:.4f}  mean_OK={mean:.4f}  "
                  f"std_OK={std:.4f}  z-score={z:.2f}")
        result, reasons = classify_by_shape(features, self.ref_stats)
        print(f"  => KET LUAN: {result}  | ly do: {', '.join(reasons) if reasons else '-'}")
        print(f"=========================\n")

    def run_inspection(self):
        if self.current_frame is None or self.cap is None:
            return

        frame = self.current_frame.copy()
        results = self.model(frame, imgsz=320, conf=0.6, iou=0.45, verbose=False)
        boxes = results[0].boxes
        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        if len(boxes) == 0:
            self.ng_count += 1
            self.ng_stat.config(text=str(self.ng_count))
            cv2.putText(display_frame, "NG - KHONG DETECT", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.display_image(self.result_label_img, display_frame)
            self.count_label.config(text="Products Detected: 0 | Batch: 0 OK / 1 NG")
            return

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                color = (0, 165, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, "OUT OF ZONE", (x1, min(frame.shape[0] - 5, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                continue

            pad_w = int((x2 - x1) * 0.08)
            pad_h = int((y2 - y1) * 0.08)
            pad = max(15, pad_w, pad_h)

            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0: continue

            contour = extract_aligned_contour(roi)
            sharpness = measure_sharpness(roi)

            if contour is None:
                result, reasons = "NG", ["khong tach duoc contour"]
                features = None
            else:
                features = get_shape_features(contour)
                result, reasons = classify_by_shape(features, self.ref_stats)

            if result == "NG":
                self.ng_count += 1; batch_ng += 1; color = (0, 0, 255)
            elif result == "OK":
                self.ok_count += 1; batch_ok += 1; color = (0, 255, 0)
            else:
                color = (0, 165, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{idx + 1}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            label_text = result if features else "NO CONTOUR"
            cv2.putText(display_frame, label_text, (x1, min(frame.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            blur_note = " [MO]" if sharpness < 80 else ""
            feat_note = (f"solidity={features['solidity']:.3f} defect_depth={features['max_defect_depth']:.2f} "
                         f"norm_area={features['norm_area']:.3f} centroid_shift={features['centroid_shift']:.3f}"
                         ) if features else ""
            print(f"San pham #{idx + 1}: {result} | {feat_note} | sharpness={sharpness:.0f}{blur_note} | ly do: {', '.join(reasons) if reasons else '-'}")

        self.display_image(self.result_label_img, display_frame)
        self.count_label.config(text=f"Products Detected: {len(boxes)} | Batch: {batch_ok} OK / {batch_ng} NG")
        self.ok_stat.config(text=str(self.ok_count))
        self.ng_stat.config(text=str(self.ng_count))

    def reset_count(self):
        self.ok_count = 0
        self.ng_count = 0
        self.ok_stat.config(text="0")
        self.ng_stat.config(text="0")


if __name__ == "__main__":
    root = tk.Tk()
    app = MachineVisionApp(root)
    root.bind('<space>', lambda e: app.run_inspection())
    root.mainloop()