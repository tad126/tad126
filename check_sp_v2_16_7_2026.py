import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ============================================================
#  CẤU HÌNH
# ============================================================
MODEL_PATH = "C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt"
REFERENCE_DIR = "reference_masks"      # nơi lưu các mẫu OK (dạng binary mask đã chuẩn hoá)
CROP_SIZE = (224, 224)                 # kích thước chuẩn hoá mask để so sánh

MIN_OK_SAMPLES = 5

# Vùng kiểm tra trung tâm (0-1 theo tỷ lệ khung hình) - tránh rìa khung bị méo ống kính/ánh sáng yếu
ZONE_X_MIN, ZONE_X_MAX = 0.15, 0.85
ZONE_Y_MIN, ZONE_Y_MAX = 0.15, 0.85

BLUR_THRESHOLD = 80          # dưới ngưỡng này coi là ảnh mờ, cảnh báo
HU_THRESHOLD = 0.35          # ngưỡng khoảng cách Hu Moments (matchShapes) - càng nhỏ càng giống
MISSING_PIXEL_THRESHOLD = 0.18  # tỷ lệ diện tích khác biệt tối đa cho phép (0-1)

CANNY_LOW, CANNY_HIGH = 50, 150

os.makedirs(REFERENCE_DIR, exist_ok=True)


# ============================================================
#  BƯỚC: Đo độ nét ảnh (Kiểm tra Blur)
# ============================================================
def measure_sharpness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
#  BƯỚC: Rotate ROI - căn chỉnh thẳng theo góc vật thể (ước lượng thô bằng Otsu)
# ============================================================
def rotate_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi_bgr

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 50:
        return roi_bgr

    rect = cv2.minAreaRect(main_contour)
    angle = rect[2]
    (h, w) = roi_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(roi_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ============================================================
#  BƯỚC: CLAHE -> Gaussian Blur -> Canny -> Morphology Close -> Largest Contour -> Binary Mask
# ============================================================
def build_binary_mask(rotated_roi):
    """
    Trả về (mask_224x224, contour_dùng_để_matchShapes, ok_flag).
    mask là ảnh nhị phân đã chuẩn hoá kích thước CROP_SIZE, letterbox giữ tỷ lệ.
    """
    gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)

    # CLAHE: cân bằng sáng cục bộ, giảm phụ thuộc ánh sáng không đều
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian Blur: giảm nhiễu trước khi tách biên
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny: tách biên
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Morphology Close: nối các đoạn biên đứt quãng thành đường viền liền
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Largest Contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 80:
        return None, None, False

    # Tạo Binary Mask: tô đặc contour lớn nhất trên canvas cùng kích thước ROI
    raw_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(raw_mask, [main_contour], -1, 255, thickness=-1)

    # Chuẩn hoá kích thước (letterbox, giữ tỷ lệ) để so sánh công bằng giữa các mẫu
    mask_std = letterbox_resize_gray(raw_mask, CROP_SIZE)

    # Lấy lại contour TỪ mask đã chuẩn hoá để dùng cho matchShapes (đồng nhất tỷ lệ giữa test/reference)
    contours_std, _ = cv2.findContours(mask_std, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_std:
        return None, None, False
    contour_std = max(contours_std, key=cv2.contourArea)

    return mask_std, contour_std, True


def letterbox_resize_gray(img, size):
    h, w = img.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off, x_off = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


# ============================================================
#  BƯỚC: Match Shape (Hu Moments) + Area Difference (Missing Pixel) -> Decision Fusion
# ============================================================
def rotations_of_mask(mask):
    return [
        mask,
        cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(mask, cv2.ROTATE_180),
        cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def missing_pixel_ratio(mask_test, mask_ref):
    xor = cv2.bitwise_xor(mask_test, mask_ref)
    diff_pixels = cv2.countNonZero(xor)
    ref_pixels = max(cv2.countNonZero(mask_ref), 1)
    return diff_pixels / ref_pixels


def compare_with_references(mask_test, contour_test, reference_masks):
    """
    So sánh với TOÀN BỘ mẫu tham chiếu đã thu thập, lấy kết quả khớp nhất (gần giống nhất).
    - Hu Moments (matchShapes): có tính bất biến xoay -> không cần thử nhiều hướng.
    - Missing Pixel: cần align đúng hướng -> thử cả 4 hướng xoay, lấy hướng khớp nhất.
    """
    if not reference_masks:
        return None

    best_hu = 999.0
    best_missing = 1.0

    test_rotations = rotations_of_mask(mask_test)

    for ref_mask, ref_contour in reference_masks:
        # --- Hu Moments (rotation-invariant, không cần thử nhiều hướng) ---
        try:
            hu = cv2.matchShapes(contour_test, ref_contour, cv2.CONTOURS_MATCH_I1, 0)
        except cv2.error:
            hu = 999.0
        best_hu = min(best_hu, hu)

        # --- Missing Pixel: thử cả 4 hướng, lấy hướng khớp nhất ---
        for rotated_test in test_rotations:
            ratio = missing_pixel_ratio(rotated_test, ref_mask)
            best_missing = min(best_missing, ratio)

    return best_hu, best_missing


def decision_fusion(hu_dist, missing_ratio):
    """NG nếu MỘT TRONG HAI tín hiệu vượt ngưỡng - ưu tiên an toàn (không bỏ sót lỗi)."""
    reasons = []
    if hu_dist > HU_THRESHOLD:
        reasons.append(f"Hu Moments lệch ({hu_dist:.3f} > {HU_THRESHOLD})")
    if missing_ratio > MISSING_PIXEL_THRESHOLD:
        reasons.append(f"Vùng khác biệt lớn ({missing_ratio*100:.1f}% > {MISSING_PIXEL_THRESHOLD*100:.0f}%)")
    result = "NG" if reasons else "OK"
    return result, reasons


def is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


# ============================================================
#  Quản lý bộ mẫu tham chiếu (nhiều ảnh OK, lưu dạng mask trên đĩa)
# ============================================================
def list_reference_files():
    if not os.path.isdir(REFERENCE_DIR):
        return []
    return sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".png")])


def load_reference_masks():
    """Trả về list các (mask, contour) đã load từ đĩa."""
    result = []
    for fname in list_reference_files():
        mask = cv2.imread(os.path.join(REFERENCE_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        result.append((mask, contour))
    return result


def save_reference_mask(mask):
    idx = len(list_reference_files())
    path = os.path.join(REFERENCE_DIR, f"sample_{idx:03d}.png")
    cv2.imwrite(path, mask)


def clear_reference_masks():
    for fname in list_reference_files():
        os.remove(os.path.join(REFERENCE_DIR, fname))


# ============================================================
#  PIPELINE TỔNG: ROI (BGR) -> (result, reasons, sharpness, mask, contour)
# ============================================================
def process_roi(roi_bgr, reference_masks):
    sharpness = measure_sharpness(roi_bgr)

    rotated = rotate_roi(roi_bgr)
    mask, contour, ok_flag = build_binary_mask(rotated)

    if not ok_flag:
        return "NG", ["Không tách được contour (kiểm tra ánh sáng / focus)"], sharpness, None

    if not reference_masks:
        return "CHƯA ĐỦ MẪU", [], sharpness, mask

    hu_dist, missing_ratio = compare_with_references(mask, contour, reference_masks)
    result, reasons = decision_fusion(hu_dist, missing_ratio)

    if sharpness < BLUR_THRESHOLD:
        reasons.append(f"ảnh khá mờ (sharpness={sharpness:.0f})")

    return result, reasons, sharpness, mask, hu_dist, missing_ratio


# ============================================================
#  GIAO DIỆN TKINTER
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

        self.model = YOLO(MODEL_PATH)
        self.reference_masks = load_reference_masks()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.current_frame = None
        self.ok_count = 0
        self.ng_count = 0

        self.setup_ui()
        self.update_camera()
        self.refresh_sample_count()

    # ---------------- UI ----------------
    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1, uniform="img")
        self.root.grid_columnconfigure(2, weight=1, uniform="img")
        self.root.grid_rowconfigure(0, weight=1)

        control_panel = tk.Frame(self.root, bg=BG_PANEL, width=240)
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

        self._section_label(control_panel, "Parameters")
        self._param_row(control_panel, "Hu Threshold", str(HU_THRESHOLD))
        self._param_row(control_panel, "Missing Px Thresh", f"{MISSING_PIXEL_THRESHOLD*100:.0f}%")
        self._param_row(control_panel, "Blur Threshold", str(BLUR_THRESHOLD))
        self._param_row(control_panel, "Conf Threshold", "0.5")

        self._section_label(control_panel, "Statistics")
        self.ok_stat = self._param_row(control_panel, "Total OK", "0")
        self.ng_stat = self._param_row(control_panel, "Total NG", "0")
        tk.Button(control_panel, text="Reset Counter", font=btn_font,
                  command=self.reset_count).pack(fill="x", padx=10, pady=6)

        # Khung camera gốc
        raw_frame = tk.Frame(self.root, bg="black")
        raw_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        raw_frame.grid_propagate(False)
        tk.Label(raw_frame, text="CAMERA (RAW)", font=tkfont.Font(size=12, weight="bold"),
                 bg="black", fg="white", anchor="w").pack(fill="x", padx=8, pady=4)
        self.raw_label = tk.Label(raw_frame, bg=BG_IMAGE)
        self.raw_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # Khung kết quả
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
        n = len(self.reference_masks)
        status = "" if n >= MIN_OK_SAMPLES else f"  (can toi thieu {MIN_OK_SAMPLES})"
        self.sample_count_label.config(text=f"Da thu thap: {n} mau{status}")

    # ---------------- Camera / hiển thị ----------------
    def display_image(self, label_widget, cv_img):
        label_widget.update_idletasks()
        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 500, 500
        img_h, img_w = cv_img.shape[:2]
        scale = min(w / img_w, h / img_h)
        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        resized = cv2.resize(cv_img, (new_w, new_h))
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        y_off, x_off = (h - new_h) // 2, (w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
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
        if len(boxes) == 0:
            return None
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 5
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        return frame[y1p:y2p, x1p:x2p]

    # ---------------- Thu thập mẫu OK ----------------
    def add_ok_sample(self):
        if self.current_frame is None:
            return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print("Khong phat hien san pham nao de them mau!")
            return

        rotated = rotate_roi(roi)
        mask, contour, ok_flag = build_binary_mask(rotated)
        if not ok_flag:
            print("Khong tach duoc contour - kiem tra anh sang / khoang cach camera!")
            return

        save_reference_mask(mask)
        self.reference_masks = load_reference_masks()
        self.refresh_sample_count()
        print(f"Da them mau OK. Tong so mau: {len(self.reference_masks)}")

    def clear_samples(self):
        clear_reference_masks()
        self.reference_masks = []
        self.refresh_sample_count()
        print("Da xoa toan bo mau tham chieu.")

    # ---------------- Kiểm tra chính ----------------
    def run_inspection(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        frame_h, frame_w = frame.shape[:2]
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes

        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        # Vẽ vùng kiểm tra để tiện quan sát
        zx1, zy1 = int(ZONE_X_MIN * frame_w), int(ZONE_Y_MIN * frame_h)
        zx2, zy2 = int(ZONE_X_MAX * frame_w), int(ZONE_Y_MAX * frame_h)
        cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 1)

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(display_frame, "NGOAI VUNG", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                continue

            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame_w, x2 + pad), min(frame_h, y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            output = process_roi(roi, self.reference_masks)
            if len(output) == 4:
                result, reasons, sharpness, mask = output
                hu_dist, missing_ratio = None, None
            else:
                result, reasons, sharpness, mask, hu_dist, missing_ratio = output

            if result == "NG":
                self.ng_count += 1
                batch_ng += 1
                color = (0, 0, 255)
            elif result == "OK":
                self.ok_count += 1
                batch_ok += 1
                color = (0, 255, 0)
            else:  # CHUA DU MAU
                color = (0, 165, 255)

            metric_txt = ""
            if hu_dist is not None:
                metric_txt = f"Hu:{hu_dist:.2f} Miss:{missing_ratio*100:.0f}%"

            print(f"San pham #{idx + 1}: {result} | {metric_txt} | sharpness={sharpness:.0f} "
                  f"| ly do: {', '.join(reasons) if reasons else '-'}")

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{idx + 1}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"{result} {metric_txt}",
                        (x1, min(frame_h - 5, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        self.display_image(self.result_label_img, display_frame)
        self.count_label.config(text=f"Products Detected: {len(boxes)}   |   Batch: {batch_ok} OK / {batch_ng} NG")
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
