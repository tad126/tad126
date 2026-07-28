import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# ============================================================
# CAU HINH CHUNG
# ============================================================
REFERENCE_PATH = "reference_stats.json"   # Luu thong ke OK
MIN_OK_SAMPLES = 20
Z_SCORE_THRESHOLD = 5.0
DEFECT_DEPTH_THRESHOLD = 3.0

# --- SAN (FLOOR) TOI THIEU CHO STD CUA TUNG DAC TRUNG ---
# Neu 20 mau OK duoc chup gan nhu lien tiep, cung dieu kien (ban tinh, sang khong doi),
# std thuc te se RAT NHO -> chi mot rung/lech tu nhien tren line cung du vuot z-score,
# gay bao NG gia dung nhu ban gap. Cac gia tri sau la muc dao dong tu nhien toi thieu
# CHAP NHAN DUOC, dua tren kinh nghiem AOI thuc te - HAY DIEU CHINH lai bang cach chay
# "Test mau (khong luu)" tren nhieu san pham OK that tren line va xem gia tri z in ra.
FEATURE_MIN_STD = {
    "solidity": 0.008,
    "perimeter_ratio": 0.012,
}

# Khong gian chuan de so sanh pixel cho loi nho (neu muon dung)
CANVAS_W, CANVAS_H = 260, 340

# Vung kiem tra an toan
ZONE_X_MIN, ZONE_X_MAX = 0.20, 0.80
ZONE_Y_MIN, ZONE_Y_MAX = 0.20, 0.80

# Nguong mau HSV
LOWER_COLOR = np.array([60, 179, 189])
UPPER_COLOR = np.array([179, 255, 255])


# ============================================================
# BUOC 1: XU LY ANH (LAY CONTOUR)
# ============================================================
def is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


def extract_main_contour(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # QUAN TRONG: RETR_CCOMP de giu lai cac lo ben trong (vd. lo tron tren san pham)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 80:
        return None, mask

    # Tao mask sach: Chi giu lai san pham, to den cac lo ben trong
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [main_contour], -1, 255, -1)
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # La contour con (lo ben trong)
                cv2.drawContours(clean_mask, [contours[i]], -1, 0, -1)

    return main_contour, clean_mask


def measure_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
# BUOC 2: TRICH XUAT DAC TRUNG HINH HOC (Z-Score)
# ============================================================
def get_shape_features(contour):
    # Lam min contour 1 LAN DUY NHAT de giam rang cua gay z-score gia
    # (ban goc goi approxPolyDP 2 lan lien tiep tren cung 1 contour, lam mat chi tiet
    #  hinh hoc that su - da gop lai thanh 1 lan de dam bao dac trung phan anh dung hinh dang)
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
# BUOC 3: Z-SCORE VA BO LOC LOI NHO (AREA RATIO)
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
    keys = ["solidity", "perimeter_ratio", "defects_count", "max_defect_depth", "area"]
    for k in keys:
        vals = np.array([s[k] for s in samples])
        raw_std = float(vals.std())
        floor = FEATURE_MIN_STD.get(k, 1e-4)
        # AP DUNG SAN TOI THIEU: neu du lieu tham chieu qua "sach" (std thuc te qua nho),
        # dung floor de tranh z-score vot nguong chi vi 1 xe dich rat nho, tu nhien.
        stats[k] = {"mean": float(vals.mean()), "std": max(raw_std, floor, 1e-4),
                    "raw_std": raw_std}
    return stats

def classify_by_shape(features, stats):
    if stats is None:
        return "CHUA DU MAU", []

    reasons = []

    # 1. Z-score cho solidity va perimeter_ratio (da co san chong z-score gia)
    for key in ["solidity", "perimeter_ratio"]:
        mean, std = stats[key]["mean"], stats[key]["std"]
        z = abs(features[key] - mean) / std
        if z > Z_SCORE_THRESHOLD:
            reasons.append(f"{key} lệch {z:.1f} lần (gt={features[key]:.4f}, mean={mean:.4f})")

    # 2. Z-score cho max_defect_depth
    mean_depth = stats["max_defect_depth"]["mean"]
    std_depth = stats["max_defect_depth"]["std"]
    depth_limit = max(mean_depth + Z_SCORE_THRESHOLD * std_depth, DEFECT_DEPTH_THRESHOLD)
    if features["max_defect_depth"] > depth_limit:
        reasons.append(f"vết lõm sâu bất thường ({features['max_defect_depth']:.2f})")

    # 3. So sanh dien tich tuyet doi (Area) de bat loi nho (giam >10% la NG)
    if "area" in stats:
        mean_area = stats["area"]["mean"]
        if features["area"] < mean_area * 0.90:
            pct_loss = (1 - features["area"] / mean_area) * 100
            reasons.append(f"diện tích giảm {pct_loss:.1f}% so với chuẩn")

    result = "NG" if reasons else "OK"
    return result, reasons


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
        self.root.title("Machine Vision Inspection System")
        self.root.geometry("1400x700")
        self.root.configure(bg=BG_DARK)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")
        if not os.path.exists(model_path):
            print(f"Canh bao: Khong tim thay model tai {model_path}! Hay dat file best.pt cung thu muc.")
            model_path = "best.pt"
        self.model = YOLO(model_path)

        self.reference = load_reference()
        self.ref_stats = compute_stats(self.reference)

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Dieu chinh theo anh sang thuc te
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.current_frame = None
        self.ok_count = 0
        self.ng_count = 0

        self.setup_ui()
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
        tk.Button(control_panel, text="Search Device", font=btn_font).pack(fill="x", padx=10, pady=2)
        row = tk.Frame(control_panel, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=2)
        tk.Button(row, text="Open Device", font=btn_font).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Close Device", font=btn_font).pack(side="left", expand=True, fill="x")

        self._section_label(control_panel, "Image Acquisition")
        self.mode_var = tk.StringVar(value="trigger")
        tk.Radiobutton(control_panel, text="Continuous", variable=self.mode_var, value="continuous",
                        bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=10)
        tk.Radiobutton(control_panel, text="Trigger Mode", variable=self.mode_var, value="trigger",
                        bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=10)

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

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.display_image(self.raw_label, frame)
        self.root.after(30, self.update_camera)

    def _get_first_roi(self, frame):
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

    def add_ok_sample(self):
        if self.current_frame is None: return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print("Khong phat hien san pham nao de them mau!")
            return

        contour, _ = extract_main_contour(roi)
        if contour is None:
            print("Khong tach duoc contour - Kiem tra HSV!")
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
        """Kiem tra 1 san pham va IN RA z-score chi tiet, KHONG luu ket qua.
        Dung de hieu chinh FEATURE_MIN_STD / Z_SCORE_THRESHOLD cho dung thuc te."""
        if self.current_frame is None:
            return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print(">> Khong phat hien san pham nao de test!")
            return

        contour, _ = extract_main_contour(roi)
        if contour is None:
            print(">> Khong tach duoc contour!")
            return

        features = get_shape_features(contour)
        result, reasons = classify_by_shape(features, self.ref_stats)

        print("\n===== [TEST DEBUG] =====")
        if self.ref_stats:
            for key in ["solidity", "perimeter_ratio"]:
                mean, std = self.ref_stats[key]["mean"], self.ref_stats[key]["std"]
                raw_std = self.ref_stats[key]["raw_std"]
                z = abs(features[key] - mean) / std
                print(f"  {key}: gt={features[key]:.4f} mean={mean:.4f} "
                      f"std_dung={std:.4f} (std_thuc={raw_std:.4f}) -> z={z:.2f}")
        print(f"  max_defect_depth = {features['max_defect_depth']:.2f}")
        print(f"  area = {features['area']:.0f}")
        print(f"  => KET LUAN: {result} | ly do: {', '.join(reasons) if reasons else '-'}")
        print("=========================\n")

    def run_inspection(self):
        if self.current_frame is None: return

        frame = self.current_frame.copy()
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes
        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        if len(boxes) == 0:
            self.ng_count += 1
            self.ng_stat.config(text=str(self.ng_count))
            cv2.putText(display_frame, "NG - KHONG DETECT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.display_image(self.result_label_img, display_frame)
            self.count_label.config(text="Products Detected: 0 | Batch: 0 OK / 1 NG")
            return

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                color = (0, 165, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, "OUT OF ZONE", (x1, min(frame.shape[0]-5, y2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                continue

            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0: continue

            contour, _ = extract_main_contour(roi)
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
            feat_note = (f"solidity={features['solidity']:.3f} defect_depth={features['max_defect_depth']:.2f}") if features else ""
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
