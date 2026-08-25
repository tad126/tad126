import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

TEMPLATE_PATH = "C:/Users/ADMIN/Pictures/check/saved_frame.jpg"
SSIM_THRESHOLD = 0.85
CROP_SIZE = (224, 224)
LOWER_COLOR = np.array([70, 40, 40])
UPPER_COLOR = np.array([100, 255, 255])


def get_straight_crop(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 50:
        return None
    rect = cv2.minAreaRect(main_contour)
    box_pts = np.array(cv2.boxPoints(rect), dtype="float32")
    (w, h) = rect[1]
    w, h = int(w), int(h)
    if w == 0 or h == 0:
        return None
    dst_pts = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box_pts, dst_pts)
    warped = cv2.warpPerspective(roi_bgr, M, (w, h))
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def letterbox_resize(img, size):
    h, w = img.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off, x_off = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def compare_with_template(straight_crop, template_img, threshold=SSIM_THRESHOLD):
    resized_test = letterbox_resize(straight_crop, CROP_SIZE)
    resized_template = letterbox_resize(template_img, CROP_SIZE)
    gray_test = cv2.cvtColor(resized_test, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_template, gray_test, full=True)
    result = "OK" if score >= threshold else "NG"
    return result, score


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
        self.template_img = cv2.imread(TEMPLATE_PATH)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.current_frame = None
        self.live_mode = True
        self.ok_count = 0
        self.ng_count = 0

        self.setup_ui()
        self.update_camera()

    def setup_ui(self):
        # ===== Layout tổng: dùng grid 3 cột =====
        # cột 0: control panel (fix width)
        # cột 1 và cột 2: 2 khung ảnh (weight bằng nhau => luôn bằng nhau)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1, uniform="img")
        self.root.grid_columnconfigure(2, weight=1, uniform="img")
        self.root.grid_rowconfigure(0, weight=1)

        # ===== PANEL ĐIỀU KHIỂN =====
        control_panel = tk.Frame(self.root, bg=BG_PANEL, width=220)
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
                        bg=BG_PANEL, font=btn_font, command=self.set_continuous).pack(anchor="w", padx=10)
        tk.Radiobutton(control_panel, text="Trigger Mode", variable=self.mode_var, value="trigger",
                        bg=BG_PANEL, font=btn_font).pack(anchor="w", padx=10)

        row2 = tk.Frame(control_panel, bg=BG_PANEL)
        row2.pack(fill="x", padx=10, pady=2)
        tk.Button(row2, text="Start", font=btn_font, command=self.start_live).pack(side="left", expand=True, fill="x")
        tk.Button(row2, text="Stop", font=btn_font, command=self.stop_live).pack(side="left", expand=True, fill="x")

        self.trigger_btn = tk.Button(control_panel, text="Trigger Once (SPACE)", font=btn_font,
                                       bg="#cfe8ff", command=self.run_inspection)
        self.trigger_btn.pack(fill="x", padx=10, pady=(8, 2))

        self._section_label(control_panel, "Picture Storage")
        tk.Button(control_panel, text="Save as JPG", font=btn_font, command=self.save_image).pack(fill="x", padx=10, pady=2)

        self._section_label(control_panel, "Parameters")
        self._param_row(control_panel, "SSIM Threshold", str(SSIM_THRESHOLD))
        self._param_row(control_panel, "Conf Threshold", "0.5")

        self._section_label(control_panel, "Statistics")
        self.ok_stat = self._param_row(control_panel, "Total OK", "0")
        self.ng_stat = self._param_row(control_panel, "Total NG", "0")
        tk.Button(control_panel, text="Reset Counter", font=btn_font,
                  command=self.reset_count).pack(fill="x", padx=10, pady=6)

        # ===== KHUNG ẢNH GỐC (cột 1) =====
        raw_frame = tk.Frame(self.root, bg="black")
        raw_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=4)
        raw_frame.grid_propagate(False)

        raw_header = tk.Label(raw_frame, text="CAMERA (RAW)", font=tkfont.Font(size=12, weight="bold"),
                                bg="black", fg="white", anchor="w")
        raw_header.pack(fill="x", padx=8, pady=4)

        self.raw_label = tk.Label(raw_frame, bg=BG_IMAGE)
        self.raw_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # ===== KHUNG ẢNH KẾT QUẢ (cột 2) =====
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
        tk.Button(control_panel, text="Lưu làm Template OK", font=btn_font, bg="#d4f7d4",
                  command=self.save_as_template).pack(fill="x", padx=10, pady=2)
    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=tkfont.Font(size=9, weight="bold"),
                 bg=BG_PANEL, fg="#333").pack(anchor="w", padx=8, pady=(10, 2))

    def _param_row(self, parent, label, value):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=10, pady=1)
        tk.Label(row, text=label, bg=BG_PANEL, font=tkfont.Font(size=9), width=13, anchor="w").pack(side="left")
        val_label = tk.Label(row, text=value, bg="white", relief="sunken", font=tkfont.Font(size=9), anchor="w")
        val_label.pack(side="left", fill="x", expand=True)
        return val_label

    def set_continuous(self):
        self.live_mode = True

    def start_live(self):
        self.live_mode = True

    def stop_live(self):
        self.live_mode = False

    def save_image(self):
        if self.current_frame is not None:
            cv2.imwrite("C:/Users/ADMIN/Pictures/check/saved_frame.jpg", self.current_frame)
            print("Đã lưu ảnh: saved_frame.jpg")

    # ---- Hàm hiển thị ảnh CHUẨN, giữ tỷ lệ, không méo ----
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

        canvas = np.full((h, w, 3), 40, dtype=np.uint8)  # nền xám tối thay vì đen tuyền
        y_off, x_off = (h - new_h) // 2, (w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.display_image(self.raw_label, frame)  # LUÔN update, bỏ điều kiện if self.live_mode
        self.root.after(30, self.update_camera)

    def save_as_template(self):
        """Lưu vùng sản phẩm ĐÃ CROP + ALIGN làm ảnh template chuẩn OK"""
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("Không phát hiện sản phẩm nào để lưu làm template!")
            return

        # Lấy sản phẩm đầu tiên detect được (đảm bảo trong khung chỉ đặt 1 sản phẩm OK chuẩn)
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 5
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        roi = frame[y1p:y2p, x1p:x2p]

        straight_crop = get_straight_crop(roi)
        if straight_crop is None:
            straight_crop = roi

        cv2.imwrite(TEMPLATE_PATH, straight_crop)
        print(f"Đã lưu template mới: {TEMPLATE_PATH}, kích thước {straight_crop.shape}")

        # Load lại ngay để dùng cho lần kiểm tra tiếp theo
        self.template_img = cv2.imread(TEMPLATE_PATH)
    def run_inspection(self):
        if self.current_frame is None or self.template_img is None:
            return

        frame = self.current_frame.copy()
        # KHÔNG cần set live_mode = False nữa
        # KHÔNG cần self.display_image(self.raw_label, frame) nữa

        results = self.model(frame, imgsz=320, conf=0.5, iou=0.45, verbose=False)
        boxes = results[0].boxes

        display_frame = frame.copy()
        batch_ok, batch_ng = 0, 0

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            straight_crop = get_straight_crop(roi)
            if straight_crop is None:
                straight_crop = roi

            result, score = compare_with_template(straight_crop, self.template_img)

            if result == "NG":
                self.ng_count += 1
                batch_ng += 1
                color = (0, 0, 255)
            else:
                self.ok_count += 1
                batch_ok += 1
                color = (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{idx + 1}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"{result} {score:.2f}", (x1, min(frame.shape[0] - 5, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Chỉ khung RESULT mới hiển thị ảnh đã đóng băng (có vẽ box)
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