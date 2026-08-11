import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from tkinter import ttk
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# ============================================================
# CAU HINH CHUNG
# ============================================================
REFERENCE_PATH_CAM1 = "reference_stats_cam1.json"
REFERENCE_PATH_CAM2 = "reference_stats_cam2.json"
MIN_OK_SAMPLES = 20
Z_SCORE_THRESHOLD = 5.0
DEFECT_DEPTH_THRESHOLD = 3.0

# --- SAN (FLOOR) TOI THIEU CHO STD ---
FEATURE_MIN_STD = {
    "solidity": 0.008,
    "perimeter_ratio": 0.012,
}

# --- RADIAL PROFILE ---
RADIAL_BINS = 72
RADIAL_MIN_STD = 0.012
RADIAL_Z_THRESHOLD = 4.0
RADIAL_MIN_CONSEC_BINS = 3
RADIAL_MAX_PHASE_SHIFT = 4

CANVAS_W, CANVAS_H = 260, 340

# Vung kiem tra an toan
ZONE_X_MIN, ZONE_X_MAX = 0.20, 0.80
ZONE_Y_MIN, ZONE_Y_MAX = 0.20, 0.80

# Nguong mau HSV rieng cho 2 cam
HSV_CONFIG = {
    "cam1": {
        "lower": np.array([60, 179, 189]),
        "upper": np.array([179, 255, 255])
    },
    "cam2": {
        "lower": np.array([70, 40, 40]),
        "upper": np.array([100, 255, 255])
    }
}


# ============================================================
# BUOC 1: XU LY ANH (LAY CONTOUR)
# ============================================================
def is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX

def extract_main_contour(roi_bgr, lower_color, upper_color):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, None

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 80:
        return None, mask, None

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [main_contour], -1, 255, -1)
    hole_contour = None
    if hierarchy is not None:
        max_hole_area = 0
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # La contour con (lo ben trong)
                cv2.drawContours(clean_mask, [contours[i]], -1, 0, -1)
                a = cv2.contourArea(contours[i])
                if a > max_hole_area:
                    max_hole_area = a
                    hole_contour = contours[i]

    return main_contour, clean_mask, hole_contour

def measure_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
# RADIAL PROFILE
# ============================================================
def rotate_contour_upright(contour, hole_contour=None):
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if angle < -45:
        angle = 90 + angle
    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    pts = contour.reshape(-1, 1, 2).astype(np.float64)
    rotated = cv2.transform(pts, M)

    xs, ys = rotated[:, 0, 0], rotated[:, 0, 1]
    y_min, y_max = ys.min(), ys.max()
    mid_y = (y_min + y_max) / 2

    if hole_contour is not None and len(hole_contour) > 0:
        hole_pts = hole_contour.reshape(-1, 1, 2).astype(np.float64)
        hole_rot = cv2.transform(hole_pts, M)
        hole_cy = hole_rot[:, 0, 1].mean()
        flip = hole_cy > mid_y
    else:
        min_y_idx = np.argmin(ys)
        flip = ys[min_y_idx] > mid_y

    if flip:
        cx_box, cy_box = (xs.min() + xs.max()) / 2, mid_y
        rotated[:, 0, 0] = 2 * cx_box - rotated[:, 0, 0]
        rotated[:, 0, 1] = 2 * cy_box - rotated[:, 0, 1]

    return rotated.astype(np.float32)

def get_radial_profile(rotated_contour, n_bins=RADIAL_BINS):
    pts = rotated_contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return None

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    hull = cv2.convexHull(rotated_contour.astype(np.int32))
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    scale = np.sqrt(hull_area / np.pi)

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    bin_idx = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins

    profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b, r in zip(bin_idx, radii):
        if r > profile[b]:
            profile[b] = r
        counts[b] += 1

    valid = counts > 0
    if not np.any(valid):
        return None
    if not np.all(valid):
        idxs = np.arange(n_bins)
        valid_idxs = idxs[valid]
        valid_vals = profile[valid]
        ext_idxs = np.concatenate([valid_idxs - n_bins, valid_idxs, valid_idxs + n_bins])
        ext_vals = np.concatenate([valid_vals, valid_vals, valid_vals])
        profile = np.interp(idxs, ext_idxs, ext_vals)

    return profile / scale

def align_profile_phase(test_profile, ref_profile, max_shift=RADIAL_MAX_PHASE_SHIFT):
    best_shift, best_err = 0, None
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(test_profile, shift)
        err = np.sum((shifted - ref_profile) ** 2)
        if best_err is None or err < best_err:
            best_err, best_shift = err, shift
    return np.roll(test_profile, best_shift), best_shift

def _longest_circular_run(flags):
    n = len(flags)
    if np.all(flags):
        return n, 0
    false_idx = np.where(~flags)[0]
    start = false_idx[0]
    rolled = np.roll(flags, -start)
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    for i, v in enumerate(rolled):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, (best_start + start) % n

def classify_by_radial(test_profile, radial_stats):
    if radial_stats is None or test_profile is None:
        return []

    mean = np.array(radial_stats["mean"])
    std = np.array(radial_stats["std"])
    aligned, shift = align_profile_phase(test_profile, mean)

    deficit = (mean - aligned) / std
    flagged = deficit > RADIAL_Z_THRESHOLD

    if not np.any(flagged):
        return []

    run_len, run_start = _longest_circular_run(flagged)
    if run_len < RADIAL_MIN_CONSEC_BINS:
        return []

    n = len(mean)
    angle_deg = run_start * (360.0 / n)
    max_deficit = deficit[flagged].max()
    return [f"thieu nhua cuc bo: {run_len} lat lien tiep quanh goc {angle_deg:.0f}° (z={max_deficit:.1f})"]

def classify_defect(features, stats, radial_profile, radial_stats):
    if stats is None:
        return "CHUA DU MAU", []
    result, reasons = classify_by_shape(features, stats)
    reasons = list(reasons) + classify_by_radial(radial_profile, radial_stats)
    result = "NG" if reasons else "OK"
    return result, reasons


# ============================================================
# BUOC 2: TRICH XUAT DAC TRUNG HINH HOC
# ============================================================
def get_shape_features(contour):
    epsilon = 0.002 * cv2.arcLength(contour, True)
    smoothed = cv2.approxPolyDP(contour, epsilon, True)

    area = cv2.contourArea(smoothed)
    perimeter = cv2.arcLength(smoothed, True)

    hull = cv2.convexHull(smoothed)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = area / hull_area if hull_area > 0 else 0
    perimeter_ratio = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    defects_count = 0
    max_defect_depth = 0.0
    hull_idx = cv2.convexHull(smoothed, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        hull_idx = np.sort(hull_idx, axis=0)
        try:
            defects = cv2.convexityDefects(smoothed, hull_idx)
            if defects is not None:
                scale = np.sqrt(area) if area > 0 else 1
                for i in range(defects.shape[0]):
                    _, _, _, d = defects[i, 0]
                    depth = (d / 256.0) / scale * 100
                    if depth > 1.5:
                        defects_count += 1
                        max_defect_depth = max(max_defect_depth, depth)
        except cv2.error:
            pass

    return {
        "area": float(area),
        "solidity": float(solidity),
        "perimeter_ratio": float(perimeter_ratio),
        "defects_count": int(defects_count),
        "max_defect_depth": float(max_defect_depth),
    }


# ============================================================
# BUOC 3: Z-SCORE VA BO LOC LOI NHO
# ============================================================
def load_reference(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {"samples": []}

def save_reference(ref, file_path):
    with open(file_path, "w") as f:
        json.dump(ref, f, indent=2)

def compute_stats(ref):
    samples = ref.get("samples", [])
    if len(samples) < 2:
        return None
    stats = {}
    keys = ["solidity", "perimeter_ratio", "defects_count", "max_defect_depth", "area"]
    for k in keys:
        vals = np.array([s[k] for s in samples])
        raw_std = float(vals.std())
        floor = FEATURE_MIN_STD.get(k, 1e-4)
        stats[k] = {"mean": float(vals.mean()), "std": max(raw_std, floor, 1e-4),
                    "raw_std": raw_std}
    return stats

def compute_radial_stats(ref):
    samples = ref.get("samples", [])
    profiles = [s["radial_profile"] for s in samples if s.get("radial_profile")]
    if len(profiles) < 2:
        return None
    arr = np.array(profiles)
    mean = arr.mean(axis=0)
    std = np.maximum(arr.std(axis=0), RADIAL_MIN_STD)
    return {"mean": mean.tolist(), "std": std.tolist()}

def classify_by_shape(features, stats):
    if stats is None:
        return "CHUA DU MAU", []

    reasons = []

    for key in ["solidity", "perimeter_ratio"]:
        mean, std = stats[key]["mean"], stats[key]["std"]
        z = abs(features[key] - mean) / std
        if z > Z_SCORE_THRESHOLD:
            reasons.append(f"{key} lệch {z:.1f} lần (gt={features[key]:.4f}, mean={mean:.4f})")

    mean_depth = stats["max_defect_depth"]["mean"]
    std_depth = stats["max_defect_depth"]["std"]
    depth_limit = max(mean_depth + Z_SCORE_THRESHOLD * std_depth, DEFECT_DEPTH_THRESHOLD)
    if features["max_defect_depth"] > depth_limit:
        reasons.append(f"vết lõm sâu bất thường ({features['max_defect_depth']:.2f})")

    if "area" in stats:
        mean_area = stats["area"]["mean"]
        if features["area"] < mean_area * 0.90:
            pct_loss = (1 - features["area"] / mean_area) * 100
            reasons.append(f"diện tích giảm {pct_loss:.1f}% so với chuẩn")

    return "NG" if reasons else "OK", reasons


# ============================================================
# BUOC 4: GUI VA LOGIC CHINH
# ============================================================
BG_DARK = "#3c3f41"
BG_PANEL = "#e8e8e8"
BG_IMAGE = "#4a4a4a"
GREEN = "#00ff00"

class MachineVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Vision Inspection System - Dual Cam")
        self.root.geometry("1700x720")
        self.root.configure(bg=BG_DARK)

        # SỬA LỖI ĐƯỜNG DẪN MODEL
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")
        if not os.path.exists(model_path):
            print(f"Canh bao: Khong tim thay model tai {model_path}! Hay dat file best.pt cung thu muc.")
            model_path = "best.pt"
        self.model = YOLO(model_path)

        # Load dữ liệu tham chiếu riêng cho từng camera
        self.ref_cam1 = load_reference(REFERENCE_PATH_CAM1)
        self.ref_cam2 = load_reference(REFERENCE_PATH_CAM2)

        self.ref_stats_cam1 = compute_stats(self.ref_cam1)
        self.ref_stats_cam2 = compute_stats(self.ref_cam2)

        self.ref_radial_stats_cam1 = compute_radial_stats(self.ref_cam1)
        self.ref_radial_stats_cam2 = compute_radial_stats(self.ref_cam2)

        # Biến camera
        self.cap1 = None
        self.cap2 = None
        self.available_devices = []
        self.current_frame1 = None
        self.current_frame2 = None

        self.ok_count = 0
        self.ng_count = 0
        self.inspection_state = "idle"

        self.setup_ui()
        self.search_device()  # Tự động quét camera khi khởi động
        self.update_camera()
        self.refresh_sample_count()

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1, uniform="img")
        self.root.grid_columnconfigure(2, weight=1, uniform="img")
        self.root.grid_columnconfigure(3, weight=1, uniform="img")
        self.root.grid_rowconfigure(0, weight=1)

        control_panel = tk.Frame(self.root, bg=BG_PANEL, width=230)
        control_panel.grid(row=0, column=0, sticky="ns")
        control_panel.grid_propagate(False)

        btn_font = tkfont.Font(family="Arial", size=10)

        # --- INITIALIZATION ---
        self._section_label(control_panel, "Initialization")
        tk.Button(control_panel, text="Search Device", font=btn_font, command=self.search_device).pack(fill="x", padx=10, pady=2)

        tk.Label(control_panel, text="Camera 1:", bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=12, pady=(4, 0))
        self.device_combo1 = ttk.Combobox(control_panel, font=btn_font, state="readonly")
        self.device_combo1.pack(fill="x", padx=10, pady=2)

        tk.Label(control_panel, text="Camera 2:", bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=12, pady=(4, 0))
        self.device_combo2 = ttk.Combobox(control_panel, font=btn_font, state="readonly")
        self.device_combo2.pack(fill="x", padx=10, pady=2)

        row = tk.Frame(control_panel, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=4)
        tk.Button(row, text="Open Device", font=btn_font, command=self.open_device).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Close Device", font=btn_font, command=self.close_device).pack(side="left", expand=True, fill="x")

        self.device_status_label = tk.Label(control_panel, text="Trạng thái: chưa mở", bg=BG_PANEL, font=tkfont.Font(size=8), fg="#a00")
        self.device_status_label.pack(anchor="w", padx=10, pady=(0, 5))

        # --- IMAGE ACQUISITION ---
        self._section_label(control_panel, "Image Acquisition")
        self.mode_var = tk.StringVar(value="trigger")
        tk.Radiobutton(control_panel, text="Continuous", variable=self.mode_var, value="continuous", bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=10)
        tk.Radiobutton(control_panel, text="Trigger Mode", variable=self.mode_var, value="trigger", bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=10)

        self.trigger_btn1 = tk.Button(control_panel, text="Trigger Cam 1 (SPACE)", font=btn_font, bg="#cfe8ff", command=self.trigger_cam1)
        self.trigger_btn1.pack(fill="x", padx=10, pady=(8, 2))
        self.trigger_btn2 = tk.Button(control_panel, text="Trigger Cam 2", font=btn_font, bg="#cfe8ff", command=self.trigger_cam2)
        self.trigger_btn2.pack(fill="x", padx=10, pady=(2, 2))

        # --- SAMPLES ---
        self._section_label(control_panel, "Bo mau tham chieu (OK)")
        self.sample_count_label = tk.Label(control_panel, text="Da thu thap: 0 mau", bg=BG_PANEL, font=btn_font, fg="#333")
        self.sample_count_label.pack(anchor="w", padx=10)

        tk.Button(control_panel, text="+ Them mau OK (tu cam1)", font=btn_font, bg="#d4f7d4", command=self.add_ok_sample_cam1).pack(fill="x", padx=10, pady=(4, 2))
        tk.Button(control_panel, text="+ Them mau OK (tu cam2)", font=btn_font, bg="#d4f7d4", command=self.add_ok_sample_cam2).pack(fill="x", padx=10, pady=(2, 2))
        tk.Button(control_panel, text="Xoa het mau", font=btn_font, bg="#f7d4d4", command=self.clear_samples).pack(fill="x", padx=10, pady=2)
        tk.Button(control_panel, text="Test mau (khong luu)", font=btn_font, bg="#fff3cd", command=self.test_sample_debug).pack(fill="x", padx=10, pady=(4, 2))

        # --- PARAMETERS & STATISTICS ---
        self._section_label(control_panel, "Parameters")
        self._param_row(control_panel, "Z-score Threshold", str(Z_SCORE_THRESHOLD))
        self._param_row(control_panel, "Conf Threshold", "0.5")

        self._section_label(control_panel, "Statistics")
        self.ok_stat = self._param_row(control_panel, "Total OK", "0")
        self.ng_stat = self._param_row(control_panel, "Total NG", "0")
        tk.Button(control_panel, text="Reset Counter", font=btn_font, command=self.reset_count).pack(fill="x", padx=10, pady=6)

        # --- CAMERA FEEDS ---
        raw_frame1 = tk.Frame(self.root, bg="black")
        raw_frame1.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        raw_frame1.grid_propagate(False)
        tk.Label(raw_frame1, text="CAMERA 1 (RAW)", font=tkfont.Font(size=12, weight="bold"), bg="black", fg="white", anchor="w").pack(fill="x", padx=8, pady=4)
        self.raw_label1 = tk.Label(raw_frame1, bg=BG_IMAGE)
        self.raw_label1.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        raw_frame2 = tk.Frame(self.root, bg="black")
        raw_frame2.grid(row=0, column=2, sticky="nsew", padx=(2, 2), pady=4)
        raw_frame2.grid_propagate(False)
        tk.Label(raw_frame2, text="CAMERA 2 (RAW)", font=tkfont.Font(size=12, weight="bold"), bg="black", fg="white", anchor="w").pack(fill="x", padx=8, pady=4)
        self.raw_label2 = tk.Label(raw_frame2, bg=BG_IMAGE)
        self.raw_label2.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        result_frame = tk.Frame(self.root, bg=BG_IMAGE)
        result_frame.grid(row=0, column=3, sticky="nsew", padx=(2, 4), pady=4)
        result_frame.grid_propagate(False)
        header = tk.Frame(result_frame, bg=BG_IMAGE)
        header.pack(fill="x", padx=8, pady=4, anchor="w")
        tk.Label(header, text="AI INSPECTION RESULT", font=tkfont.Font(size=12, weight="bold"), bg=BG_IMAGE, fg=GREEN).pack(anchor="w")
        self.count_label = tk.Label(header, text="Products Detected: 0", font=tkfont.Font(size=10), bg=BG_IMAGE, fg=GREEN)
        self.count_label.pack(anchor="w")
        self.result_label_img = tk.Label(result_frame, bg=BG_IMAGE)
        self.result_label_img.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=tkfont.Font(size=9, weight="bold"), bg=BG_PANEL, fg="#333").pack(anchor="w", padx=8, pady=(10, 2))

    def _param_row(self, parent, label, value):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=1)
        tk.Label(row, text=label, bg=BG_PANEL, font=tkfont.Font(size=9), width=15, anchor="w").pack(side="left")
        val_label = tk.Label(row, text=value, bg="white", relief="sunken", font=tkfont.Font(size=9), anchor="w")
        val_label.pack(side="left", fill="x", expand=True)
        return val_label

    def refresh_sample_count(self):
        n1 = len(self.ref_cam1.get("samples", []))
        n2 = len(self.ref_cam2.get("samples", []))
        total = n1 + n2
        status = "" if total >= MIN_OK_SAMPLES * 2 else f"  (can toi thieu {MIN_OK_SAMPLES*2} - {MIN_OK_SAMPLES} mau/cam)"
        self.sample_count_label.config(text=f"Da thu thap: {total} mau (Cam1:{n1}, Cam2:{n2}){status}")

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
        found = []
        for i in range(5):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    found.append(i)
                cap_test.release()
        self.available_devices = found
        if found:
            self.device_combo1["values"] = [f"Camera {i}" for i in found]
            self.device_combo2["values"] = [f"Camera {i}" for i in found]
            if len(found) > 0:
                self.device_combo1.current(0)
                self.device_combo2.current(min(1, len(found) - 1))
            print(f"Tìm thấy {len(found)} camera: {found}")
        else:
            self.device_combo1["values"] = []
            self.device_combo2["values"] = []
            print("Không tìm thấy camera nào!")

    def _open_single_camera(self, index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Đã bật lại các dòng cấu hình camera cố định
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            return cap
        return None

    def open_device(self):
        if not self.available_devices:
            print("Chưa tìm thấy thiết bị - hãy bấm Search Device trước!")
            return

        try:
            idx1 = int(self.device_combo1.get().split(" ")[1]) if self.device_combo1.get() else None
            idx2 = int(self.device_combo2.get().split(" ")[1]) if self.device_combo2.get() else None
        except:
            print("Lỗi parse index camera. Kiểm tra lại combobox.")
            return

        self.close_device()

        if idx1 is not None:
            self.cap1 = self._open_single_camera(idx1)
        if idx2 is not None:
            self.cap2 = self._open_single_camera(idx2)

        status_text = "Trạng thái: "
        if self.cap1 and self.cap2:
            status_text += "Đã mở cả 2 camera"
        elif self.cap1:
            status_text += "Đã mở Cam 1 (Cam 2 lỗi)"
        elif self.cap2:
            status_text += "Đã mở Cam 2 (Cam 1 lỗi)"
        else:
            status_text += "Mở thất bại!"

        self.device_status_label.config(text=status_text, fg="#080" if self.cap1 or self.cap2 else "#a00")
        self.trigger_btn1.config(state="normal" if self.cap1 else "disabled")
        self.trigger_btn2.config(state="normal" if self.cap2 else "disabled")

    def close_device(self):
        if self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        if self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None
        self.current_frame1 = None
        self.current_frame2 = None
        self.device_status_label.config(text="Trạng thái: đã đóng", fg="#a00")
        self.trigger_btn1.config(state="disabled")
        self.trigger_btn2.config(state="disabled")

    def update_camera(self):
        if self.cap1 is not None and self.cap1.isOpened():
            ret1, frame1 = self.cap1.read()
            if ret1:
                self.current_frame1 = frame1.copy()
                self.display_image(self.raw_label1, frame1)
        if self.cap2 is not None and self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.current_frame2 = frame2.copy()
                self.display_image(self.raw_label2, frame2)
        self.root.after(30, self.update_camera)

    def _get_first_roi(self, frame, lower_color, upper_color):
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0: return None
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]): continue
            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            return frame[y1p:y2p, x1p:x2p]
        return None

    def _inspect_frame(self, frame, lower_color, upper_color, stats, radial_stats):
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes
        detections = []
        if len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                detections.append({"bbox": (x1, y1, x2, y2), "result": "ZONE", "reasons": []})
                continue

            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            contour, _, hole_contour = extract_main_contour(roi, lower_color, upper_color)
            if contour is None:
                detections.append({"bbox": (x1, y1, x2, y2), "result": "NG", "reasons": ["khong tach duoc contour"]})
                continue

            features = get_shape_features(contour)
            profile = get_radial_profile(rotate_contour_upright(contour, hole_contour))
            result, reasons = classify_defect(features, stats, profile, radial_stats)
            detections.append({"bbox": (x1, y1, x2, y2), "result": result, "reasons": reasons})

        return detections

    def _update_result(self, detections, cam_id):
        frame = self.current_frame1 if cam_id == 1 else self.current_frame2
        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            result = det["result"]

            if result == "ZONE":
                color = (0, 165, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, "OUT OF ZONE", (x1, min(frame.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                continue

            if result == "NG":
                self.ng_count += 1
                batch_ng += 1
                color = (0, 0, 255)
            else:
                self.ok_count += 1
                batch_ok += 1
                color = (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{idx + 1}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, result, (x1, min(frame.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Cam{cam_id} #{idx + 1}: {result} | ly do: {', '.join(det['reasons']) if det['reasons'] else '-'}")

        self.ok_stat.config(text=str(self.ok_count))
        self.ng_stat.config(text=str(self.ng_count))
        self.display_image(self.result_label_img, display_frame)
        self.count_label.config(text=f"Cam{cam_id} - Products Detected: {len(detections)} | Batch: {batch_ok} OK / {batch_ng} NG")

    def trigger_cam1(self):
        if self.current_frame1 is None: return
        detections = self._inspect_frame(
            self.current_frame1,
            HSV_CONFIG["cam1"]["lower"], HSV_CONFIG["cam1"]["upper"],
            self.ref_stats_cam1, self.ref_radial_stats_cam1
        )
        self._update_result(detections, 1)

    def trigger_cam2(self):
        if self.current_frame2 is None: return
        # SỬA LỖI: Truyền đúng dữ liệu thống kê cho Cam2
        detections = self._inspect_frame(
            self.current_frame2,
            HSV_CONFIG["cam2"]["lower"], HSV_CONFIG["cam2"]["upper"],
            self.ref_stats_cam2, self.ref_radial_stats_cam2
        )
        self._update_result(detections, 2)

    def _add_sample_from_roi(self, roi, lower, upper, src, ref_data, ref_path, stats, radial_stats):
        contour, _, hole_contour = extract_main_contour(roi, lower, upper)
        if contour is None:
            print(f"Không tách được contour từ {src}")
            return
        features = get_shape_features(contour)
        profile = get_radial_profile(rotate_contour_upright(contour, hole_contour))
        features["radial_profile"] = profile.tolist() if profile is not None else None

        ref_data.setdefault("samples", []).append(features)
        save_reference(ref_data, ref_path)

        # Cập nhật lại stats cho camera tương ứng
        if src == "cam1":
            self.ref_stats_cam1 = compute_stats(ref_data)
            self.ref_radial_stats_cam1 = compute_radial_stats(ref_data)
        else:
            self.ref_stats_cam2 = compute_stats(ref_data)
            self.ref_radial_stats_cam2 = compute_radial_stats(ref_data)

        self.refresh_sample_count()
        print(f"Đã thêm mẫu OK từ {src}: solidity={features['solidity']:.4f}")

    def add_ok_sample_cam1(self):
        if self.current_frame1 is None: return
        roi = self._get_first_roi(self.current_frame1, HSV_CONFIG["cam1"]["lower"], HSV_CONFIG["cam1"]["upper"])
        if roi is None:
            print("Không phát hiện sản phẩm trên camera 1")
            return
        self._add_sample_from_roi(roi, HSV_CONFIG["cam1"]["lower"], HSV_CONFIG["cam1"]["upper"],
                                   "cam1", self.ref_cam1, REFERENCE_PATH_CAM1,
                                   self.ref_stats_cam1, self.ref_radial_stats_cam1)

    def add_ok_sample_cam2(self):
        if self.current_frame2 is None: return
        roi = self._get_first_roi(self.current_frame2, HSV_CONFIG["cam2"]["lower"], HSV_CONFIG["cam2"]["upper"])
        if roi is None:
            print("Không phát hiện sản phẩm trên camera 2")
            return
        self._add_sample_from_roi(roi, HSV_CONFIG["cam2"]["lower"], HSV_CONFIG["cam2"]["upper"],
                                   "cam2", self.ref_cam2, REFERENCE_PATH_CAM2,
                                   self.ref_stats_cam2, self.ref_radial_stats_cam2)

    def clear_samples(self):
        # Xóa dữ liệu trong bộ nhớ và ghi đè file rỗng cho cả 2 cam
        self.ref_cam1 = {"samples": []}
        self.ref_cam2 = {"samples": []}
        save_reference(self.ref_cam1, REFERENCE_PATH_CAM1)
        save_reference(self.ref_cam2, REFERENCE_PATH_CAM2)

        self.ref_stats_cam1 = None
        self.ref_stats_cam2 = None
        self.ref_radial_stats_cam1 = None
        self.ref_radial_stats_cam2 = None
        self.refresh_sample_count()
        print("Đã xóa toàn bộ mẫu tham chiếu của cả 2 camera.")

    def test_sample_debug(self):
        # Debug trên Cam 1
        if self.current_frame1 is None: return
        roi = self._get_first_roi(self.current_frame1, HSV_CONFIG["cam1"]["lower"], HSV_CONFIG["cam1"]["upper"])
        if roi is None:
            print("Không phát hiện sản phẩm trên camera 1")
            return
        contour, _, hole_contour = extract_main_contour(roi, HSV_CONFIG["cam1"]["lower"], HSV_CONFIG["cam1"]["upper"])
        if contour is None:
            print(">> Không tách được contour!")
            return
        features = get_shape_features(contour)
        profile = get_radial_profile(rotate_contour_upright(contour, hole_contour))

        # SỬA LỖI: Test debug phải dùng Stats Cam1 mới đúng
        result, reasons = classify_defect(features, self.ref_stats_cam1, profile, self.ref_radial_stats_cam1)

        print("\n===== [TEST DEBUG] =====")
        if self.ref_stats_cam1:
            for key in ["solidity", "perimeter_ratio"]:
                mean, std = self.ref_stats_cam1[key]["mean"], self.ref_stats_cam1[key]["std"]
                raw_std = self.ref_stats_cam1[key]["raw_std"]
                z = abs(features[key] - mean) / std
                print(f"  {key}: gt={features[key]:.4f} mean={mean:.4f} std_dung={std:.4f} (std_thuc={raw_std:.4f}) -> z={z:.2f}")
        print(f"  max_defect_depth = {features['max_defect_depth']:.2f}")
        print(f"  area = {features['area']:.0f}")
        if self.ref_radial_stats_cam1 and profile is not None:
            mean_arr = np.array(self.ref_radial_stats_cam1["mean"])
            std_arr = np.array(self.ref_radial_stats_cam1["std"])
            aligned, shift = align_profile_phase(profile, mean_arr)
            deficit = (mean_arr - aligned) / std_arr
            top = np.argsort(deficit)[::-1][:5]
            print(f"  radial: da can chinh lech {shift} lat, top-5 goc hut nhieu nhat:")
            for b in top:
                angle_deg = b * (360.0 / len(mean_arr))
                print(f"    goc {angle_deg:5.0f}° : z_hut={deficit[b]:.2f}")
        else:
            print("  radial: chua co du lieu")
        print(f"  => KET LUAN: {result} | ly do: {', '.join(reasons) if reasons else '-'}")
        print("=========================\n")

    def reset_count(self):
        self.ok_count = 0
        self.ng_count = 0
        self.ok_stat.config(text="0")
        self.ng_stat.config(text="0")


if __name__ == "__main__":
    root = tk.Tk()
    app = MachineVisionApp(root)
    root.bind('<space>', lambda e: app.trigger_cam1())
    root.bind('<Return>', lambda e: app.trigger_cam2())
    root.mainloop()