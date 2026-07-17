import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

REFERENCE_PATH = "reference_stats.json"   # nơi lưu thống kê các mẫu OK đã thu thập
CROP_SIZE = (224, 224)
LOWER_COLOR = np.array([70, 40, 40])
UPPER_COLOR = np.array([100, 255, 255])

MIN_OK_SAMPLES = 20          # cần tối thiểu bấy nhiêu mẫu OK mới đủ tin cậy để tính thống kê
Z_SCORE_THRESHOLD = 3.0     # số độ lệch chuẩn cho phép trước khi coi là bất thường (NG)
DEFECT_DEPTH_THRESHOLD = 3.0  # ngưỡng riêng cho độ sâu vết lõm (đơn vị đã chuẩn hoá theo kích thước vật)

# Vùng kiểm tra trung tâm (tỷ lệ 0-1 theo chiều rộng/cao khung hình).
# Sản phẩm nằm NGOÀI vùng này sẽ không bị chấm OK/NG - vì rìa khung hình
# thường bị méo ống kính, ánh sáng yếu hơn, dễ cho ra đặc trưng sai lệch.
ZONE_X_MIN, ZONE_X_MAX = 0.20, 0.80
ZONE_Y_MIN, ZONE_Y_MAX = 0.20, 0.80


# ============================================================
#  BƯỚC 1: Tách vật thể ra khỏi nền bằng màu (HSV) -> lấy contour
# ============================================================
def is_in_inspection_zone(x1, y1, x2, y2, frame_w, frame_h):
    """Kiểm tra tâm bounding box có nằm trong vùng kiểm tra trung tâm không."""
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    return ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX


def extract_main_contour(roi_bgr):
    # 1. Làm mịn ảnh để giảm nhiễu răng cưa
    blurred = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

    # 2. Dùng kernel (7,7) để khử nhiễu hạt và nối các vùng đứt gãy
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. SỬA QUAN TRỌNG: Dùng RETR_CCOMP để lấy cả các lỗ bên trong (hierarchy)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    # Lấy contour ngoài cùng có diện tích lớn nhất
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 80:
        return None, mask

    # 4. Tạo mask sạch: Chỉ giữ lại sản phẩm, tô đen các lỗ bên trong
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [main_contour], -1, 255, -1)

    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # Nếu là contour con (nằm bên trong sản phẩm) -> tô màu đen để loại bỏ khỏi contour
            if h[3] != -1:
                cv2.drawContours(clean_mask, [contours[i]], -1, 0, -1)

    return main_contour, clean_mask


def measure_sharpness(img_bgr):
    """Đo độ nét ảnh (Variance of Laplacian). Càng thấp càng mờ. Dưới ~80 là khá mờ."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
#  BƯỚC 2: Tính đặc trưng HÌNH HỌC của contour
#  Các đặc trưng này KHÔNG phụ thuộc vào góc xoay của vật thể,
#  nên không cần lo việc "align đúng hướng" như cách làm cũ (SSIM).
# ============================================================
def get_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    # Solidity: sứt mẻ làm mất phần lồi -> solidity giảm rõ rệt


    # Perimeter ratio: vết sứt mẻ tạo viền lởm chởm -> chu vi tăng so với hull
    defects_count = 0
    max_defect_depth = 0.0
    # Thêm biến mới
    depth_values = []

    # Convexity defects: đây là "dấu hiệu toán học" trực tiếp của vết lõm/sứt mẻ

    defects_count = 0
    max_defect_depth = 0.0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = area / hull_area if hull_area > 0 else 0
    perimeter_ratio = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    defects_count = 0
    max_defect_depth = 0.0

    # CHUẨN HÓA DIỆN TÍCH: Lấy bbox của contour
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    norm_area = area / bbox_area if bbox_area > 0 else 0

    # TÍNH ĐỘ LỆCH TÂM KHỐI (Để bắt sứt mẻ nặng và vỡ lệch tâm):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx_contour = M["m10"] / M["m00"]
        cy_contour = M["m01"] / M["m00"]
        # Tâm của Bbox
        cx_bbox = x + w / 2
        cy_bbox = y + h / 2
        # Khoảng cách lệch tâm chuẩn hóa theo kích thước bbox
        centroid_shift = np.sqrt((cx_contour - cx_bbox) ** 2 + (cy_contour - cy_bbox) ** 2) / max(w, h)
    else:
        centroid_shift = 0.0

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
                    if depth > 1.5:
                        defects_count += 1
                        max_defect_depth = max(max_defect_depth, depth)
        except cv2.error:
            pass
    if hull_idx is not None and len(hull_idx) > 3:
        hull_idx = np.sort(hull_idx, axis=0)
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                # Chuẩn hoá độ sâu theo kích thước vật (sqrt(area)) để không phụ thuộc khoảng cách camera
                scale = np.sqrt(area) if area > 0 else 1
                for i in range(defects.shape[0]):
                    _, _, _, d = defects[i, 0]
                    depth = (d / 256.0) / scale * 100  # đưa về đơn vị % kích thước vật, dễ so sánh
                    depth_values.append(depth)
                    defects_count += 1
                    max_defect_depth = max(max_defect_depth, depth)
        except cv2.error:
            pass

    return {
        "area": float(area),
        "norm_area": float(norm_area),  # Thêm mới
        "centroid_shift": float(centroid_shift),  # Thêm mới
        "mean_defect_depth": float(np.mean(depth_values)) if depth_values else 0.0,
        "solidity": float(solidity),
        "perimeter_ratio": float(perimeter_ratio),
        "defects_count": int(defects_count),
        "max_defect_depth": float(max_defect_depth),
    }


# ============================================================
#  BƯỚC 3: Quản lý bộ thống kê tham chiếu (nhiều mẫu OK, không phải 1 ảnh duy nhất)
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
    # Bổ sung "mean_defect_depth" vào danh sách keys
    keys = ["norm_area", "solidity", "perimeter_ratio", "defects_count", "max_defect_depth", "mean_defect_depth"]
    for k in keys:
        vals = np.array([s[k] for s in samples])
        stats[k] = {"mean": float(vals.mean()), "std": float(max(vals.std(), 1e-4))}
    return stats


def classify_by_shape(features, stats):
    if stats is None:
        return "CHƯA ĐỦ MẪU", []

    reasons = []
    MIN_STD_SOLIDITY = 0.02
    MIN_STD_PERIMETER = 0.03
    MIN_STD_DEPTH = 0.8

    # CẬP NHẬT: Kiểm tra tỷ lệ diện tích (norm_area)
    if "norm_area" in stats:
        mean_norm_area = stats["norm_area"]["mean"]
        # Nếu tỷ lệ diện tích hao hụt hơn 12% -> Báo NG (Vì nó đã chuẩn hóa nên ngưỡng này rất ổn định)
        if features["norm_area"] < mean_norm_area * 0.88:
            reasons.append(f"tỷ lệ diện tích giảm {(1 - features['norm_area'] / mean_norm_area) * 100:.1f}%")
            return "NG", reasons

    # CẬP NHẬT: Kiểm tra độ lệch tâm khối (Để bắt sứt mẻ 1 bên cạnh nặng)
    if "centroid_shift" in stats:
        mean_shift = stats["centroid_shift"]["mean"]
        std_shift = stats["centroid_shift"]["std"]
        effective_std_shift = max(std_shift, 0.005)  # Dung sai tối thiểu

        # 1. Cộng thêm một lượng "dung sai cứng" (tolerance) để bỏ qua nhiễu nhỏ do xoay vật
        SAFE_TOLERANCE = 0.03  # 3% kích thước vật, bạn có thể test tăng/giảm

        # 2. Điều kiện NG: Phải vượt quá cả Z-score và vượt quá dung sai cứng
        shift_z_score = (features["centroid_shift"] - mean_shift) / effective_std_shift
        if features["centroid_shift"] > (mean_shift + 3 * effective_std_shift) and \
                features["centroid_shift"] > (mean_shift + SAFE_TOLERANCE):
            reasons.append(f"khối tâm lệch {features['centroid_shift']:.3f}")

    for key in ["solidity", "perimeter_ratio"]:
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        effective_std = std
        if key == "solidity":
            effective_std = max(std, MIN_STD_SOLIDITY)
        elif key == "perimeter_ratio":
            effective_std = max(std, MIN_STD_PERIMETER)

        z = abs(features[key] - mean) / effective_std
        if z > Z_SCORE_THRESHOLD:
            reasons.append(f"{key} lệch {z:.1f} lần")

    mean_depth = stats["max_defect_depth"]["mean"]
    std_depth = stats["max_defect_depth"]["std"]
    effective_std_depth = max(std_depth, MIN_STD_DEPTH)
    depth_limit = max(mean_depth + Z_SCORE_THRESHOLD * effective_std_depth, DEFECT_DEPTH_THRESHOLD)

    mean_depth = stats["max_defect_depth"]["mean"]
    std_depth = stats["max_defect_depth"]["std"]
    effective_std_depth = max(std_depth, MIN_STD_DEPTH)
    depth_limit = max(mean_depth + Z_SCORE_THRESHOLD * effective_std_depth, DEFECT_DEPTH_THRESHOLD)

    if features["max_defect_depth"] > depth_limit:
        reasons.append(f"vết lõm sâu {features['max_defect_depth']:.2f}")

    # BỔ SUNG: Bắt vết sứt NÔNG nhưng DÀY ĐẶC (dùng mean_defect_depth)
    if "mean_defect_depth" in stats:
        mean_md = stats["mean_defect_depth"]["mean"]
        std_md = stats["mean_defect_depth"]["std"]
        # Nếu độ sâu trung bình của các vết lõm tăng đột biến (ví dụ > 2 lần chuẩn)
        if features["mean_defect_depth"] > mean_md + 2 * max(std_md, 0.3):
            reasons.append(f"độ sâu vết lõm trung bình tăng {features['mean_defect_depth']:.2f}")

    result = "NG" if reasons else "OK"
    return result, reasons


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

        self.model = YOLO("C:/Downloads/check sp_v2-20260712T152639Z-2-001/check sp_v2/weights/best.pt")

        self.reference = load_reference()
        self.ref_stats = compute_stats(self.reference)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

        # ===== Khu vực THU THẬP MẪU OK (thay cho "1 ảnh template" cũ) =====
        self._section_label(control_panel, "Bo mau tham chieu (OK)")
        self.sample_count_label = tk.Label(control_panel, text="Da thu thap: 0 mau",
                                             bg=BG_PANEL, font=btn_font, fg="#333")
        self.sample_count_label.pack(anchor="w", padx=10)

        tk.Button(control_panel, text="+ Them mau OK (tu khung trai)", font=btn_font, bg="#d4f7d4",
                  command=self.add_ok_sample).pack(fill="x", padx=10, pady=(4, 2))
        tk.Button(control_panel, text="Xoa het mau", font=btn_font, bg="#f7d4d4",
                  command=self.clear_samples).pack(fill="x", padx=10, pady=2)

        self._section_label(control_panel, "Parameters")
        self._param_row(control_panel, "Z-score Threshold", str(Z_SCORE_THRESHOLD))
        self._param_row(control_panel, "Conf Threshold", "0.5")

        self._section_label(control_panel, "Statistics")
        self.ok_stat = self._param_row(control_panel, "Total OK", "0")
        self.ng_stat = self._param_row(control_panel, "Total NG", "0")
        tk.Button(control_panel, text="Reset Counter", font=btn_font,
                  command=self.reset_count).pack(fill="x", padx=10, pady=6)

        # ===== KHUNG ẢNH GỐC =====
        raw_frame = tk.Frame(self.root, bg="black")
        raw_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        raw_frame.grid_propagate(False)
        tk.Label(raw_frame, text="CAMERA (RAW)", font=tkfont.Font(size=12, weight="bold"),
                 bg="black", fg="white", anchor="w").pack(fill="x", padx=8, pady=4)
        self.raw_label = tk.Label(raw_frame, bg=BG_IMAGE)
        self.raw_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # ===== KHUNG ẢNH KẾT QUẢ =====
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

    # ---- Hiển thị ảnh giữ tỷ lệ, không méo ----
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
        """Lấy ROI của sản phẩm đầu tiên YOLO detect được trong frame."""
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                continue
            w_box = x2 - x1
            h_box = y2 - y1
            # Tính pad theo tỷ lệ 5% kích thước sản phẩm, nhưng tối thiểu là 15 pixel
            pad = max(15, int(0.05 * max(w_box, h_box)))
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            return frame[y1p:y2p, x1p:x2p]
        return None

    def add_ok_sample(self):
        """Thêm 1 mẫu OK vào bộ tham chiếu, dựa trên sản phẩm đang hiện ở khung camera trái."""
        if self.current_frame is None:
            return
        roi = self._get_first_roi(self.current_frame.copy())
        if roi is None or roi.size == 0:
            print("Khong phat hien san pham nao trong khung hinh de them mau!")
            return

        contour, _ = extract_main_contour(roi)
        if contour is None:
            print("Khong tach duoc contour san pham - kiem tra lai mau HSV hoac anh sang!")
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

    def run_inspection(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes
        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        # --- BỔ SUNG TỪ ĐÂY ---
        if len(boxes) == 0:
            self.ng_count += 1
            self.ng_stat.config(text=str(self.ng_count))
            cv2.putText(display_frame, "NG - KHONG DETECT SAN PHAM", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self.display_image(self.result_label_img, display_frame)
            self.count_label.config(text="Products Detected: 0 | Batch: 0 OK / 1 NG")
            return

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not is_in_inspection_zone(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                color = (0, 165, 255)  # orange: detected, but intentionally not inspected
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, "OUT OF ZONE",
                            (x1, min(frame.shape[0] - 5, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                print(f"San pham #{idx + 1}: OUT OF ZONE - khong cham OK/NG")
                continue
            pad_w = int((x2 - x1) * 0.08)
            pad_h = int((y2 - y1) * 0.08)
            pad = max(15, pad_w, pad_h)

            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]

            if roi.size == 0:
                continue

            contour, _ = extract_main_contour(roi)
            sharpness = measure_sharpness(roi)

            if contour is None:
                result, reasons = "NG", ["khong tach duoc contour (kiem tra anh sang/mau)"]
                features = None
            else:
                features = get_shape_features(contour)
                result, reasons = classify_by_shape(features, self.ref_stats)

            if result == "NG":
                self.ng_count += 1
                batch_ng += 1
                color = (0, 0, 255)
            elif result == "OK":
                self.ok_count += 1
                batch_ok += 1
                color = (0, 255, 0)
            else:  # "CHUA DU MAU"
                color = (0, 165, 255)

            blur_note = " [MO]" if sharpness < 80 else ""
            feat_note = ""
            if features:
                feat_note = (f"solidity={features['solidity']:.3f} "
                              f"defect_depth={features['max_defect_depth']:.2f}")
            print(f"San pham #{idx + 1}: {result} | {feat_note} | sharpness={sharpness:.0f}{blur_note} "
                  f"| ly do: {', '.join(reasons) if reasons else '-'}")

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{idx + 1}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            label_text = result if features else "NO CONTOUR"
            cv2.putText(display_frame, label_text, (x1, min(frame.shape[0] - 5, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
