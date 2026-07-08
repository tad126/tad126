import cv2
import numpy as np
from ultralytics import YOLO
import os

# ============ CONFIG ============
MODEL_PATH = "C:/Downloads/results-20260706T170355Z-3-001/results\DO AN TOT NGHIEP_v1/weights/best.pt"              # đổi sang "best_ncnn_model" khi chạy trên Pi4
TEMPLATE_PATH = "C:/Users/ADMIN/Pictures/IMG_8429.JPG"  # ảnh mẫu sản phẩm OK chuẩn
CONF_THRESHOLD = 0.5
SHAPE_THRESHOLD = 0.15              # ngưỡng so sánh contour, cần tự test để tìm giá trị phù hợp
SAVE_DIR = "captured_results"

# Khoảng màu HSV của sản phẩm - CẦN chỉnh lại theo giá trị bạn đã dò được bằng trackbar
LOWER_COLOR = np.array([70, 50, 50])
UPPER_COLOR = np.array([100, 255, 255])

os.makedirs(SAVE_DIR, exist_ok=True)

# ============ LOAD MODEL & TEMPLATE ============
print("Đang load model...")
model = YOLO(MODEL_PATH)

def get_contour_from_color(img, lower, upper):
    """Tách contour bằng màu HSV thay vì grayscale threshold"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour, mask

# Load template contour
template_contour = None
if os.path.exists(TEMPLATE_PATH):
    template_img = cv2.imread(TEMPLATE_PATH)
    template_contour, _ = get_contour_from_color(template_img, LOWER_COLOR, UPPER_COLOR)
    if template_contour is not None:
        print("Đã load template thành công.")
    else:
        print("CẢNH BÁO: không tìm được contour trong ảnh template.")
else:
    print(f"CẢNH BÁO: chưa có file template tại '{TEMPLATE_PATH}'.")


# ============ HÀM SO SÁNH SHAPE (đã đổi sang dùng màu HSV) ============
def check_shape(crop_img, template_contour, threshold=SHAPE_THRESHOLD):
    if template_contour is None:
        return "SKIP", -1

    main_contour, mask = get_contour_from_color(crop_img, LOWER_COLOR, UPPER_COLOR)

    if main_contour is None:
        return "NG", 999.0  # không tách được vật thể -> nghi ngờ lỗi hoặc sai màu

    # Align góc xoay trước khi so sánh
    rect = cv2.minAreaRect(main_contour)
    angle = rect[2]
    (h, w) = crop_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(crop_img, M, (w, h))

    main_contour_rot, _ = get_contour_from_color(rotated, LOWER_COLOR, UPPER_COLOR)
    if main_contour_rot is None:
        main_contour_rot = main_contour  # fallback nếu xoay xong bị mất contour

    similarity = cv2.matchShapes(main_contour_rot, template_contour, cv2.CONTOURS_MATCH_I1, 0)
    result = "NG" if similarity > threshold else "OK"
    return result, similarity


# ============ HÀM XỬ LÝ CHÍNH KHI CHỤP ẢNH ============
def process_frame(frame):
    results = model(frame, imgsz=320, conf=CONF_THRESHOLD, iou=0.45, verbose=False)

    if len(results[0].boxes) == 0:
        print(">> Không phát hiện sản phẩm nào trong ảnh.")
        return frame, []

    output_frame = frame.copy()
    all_results = []

    for i, box in enumerate(results[0].boxes):
        yolo_label = model.names[int(box.cls[0])]
        yolo_conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]

        shape_result, similarity = check_shape(crop, template_contour)

        if shape_result == "SKIP":
            final_result = yolo_label
        else:
            final_result = "NG" if (yolo_label == "NG" or shape_result == "NG") else "OK"

        color = (0, 0, 255) if final_result == "NG" else (0, 255, 0)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{final_result} (YOLO:{yolo_label} {yolo_conf:.2f}, Shape:{shape_result} {similarity:.3f})"
        cv2.putText(output_frame, text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"Sản phẩm #{i+1}: YOLO={yolo_label}({yolo_conf:.2f}) | Shape={shape_result}({similarity:.3f}) => KẾT QUẢ: {final_result}")

        all_results.append(final_result)

    return output_frame, all_results


# ============ VÒNG LẶP CHÍNH ============
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n=== Camera đã sẵn sàng ===")
print("Bấm 'q' để chụp ảnh và xử lý.")
print("Bấm 'ESC' để thoát.\n")

capture_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Live Camera - Press 'q' to capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print(f"\n--- Đã chụp ảnh #{capture_count + 1} ---")
        result_frame, results_list = process_frame(frame)

        cv2.imshow("Ket Qua", result_frame)

        save_path = os.path.join(SAVE_DIR, f"capture_{capture_count}.jpg")
        cv2.imwrite(save_path, result_frame)
        print(f"Đã lưu ảnh kết quả tại: {save_path}")

        capture_count += 1

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()