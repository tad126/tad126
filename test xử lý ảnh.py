import cv2
import numpy as np
from matplotlib import pyplot as plt

# (Tùy chỉnh) Hãy dùng tool check màu của bạn để chỉnh lại 2 dòng này cho sát màu sản phẩm
LOWER_COLOR = np.array([82, 127, 92])
UPPER_COLOR = np.array([179, 255, 255])

img = cv2.imread("C:/Users/ADMIN/Pictures/z8078682672461_be17f415eb2f84acadff6207a65a0200.jpg")

# --- BƯỚC 1: Tạo Mask HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_hsv = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
kernel_hsv = np.ones((3, 3), np.uint8)
mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_hsv)
mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel_hsv)

# === CỰC KỲ QUAN TRỌNG: Giãn nở Mask để bù phần bị chói sáng ===
dilate_kernel = np.ones((7, 7), np.uint8)
mask_hsv = cv2.dilate(mask_hsv, dilate_kernel, iterations=1)
# ================================================================
# (Bạn có thể bỏ comment dòng dưới để chạy thử xem Mask có đúng không)
# cv2.imshow("Debug Mask", mask_hsv); cv2.waitKey(0)

# --- BƯỚC 2: Xử lý Canny ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)
gray_blur = cv2.medianBlur(gray_clahe, 5)
edges = cv2.Canny(gray_blur, 70, 180)

# --- BƯỚC 3: Kết hợp ---
edges_clean = cv2.bitwise_and(edges, edges, mask=mask_hsv)

# --- BƯỚC 4: Nối mạch ---
kernel_repair = np.ones((3, 3), np.uint8)
edges_clean = cv2.dilate(edges_clean, kernel_repair, iterations=1)

# --- BƯỚC 5: Tìm Contour và Xoay ảnh ---
contours, _ = cv2.findContours(edges_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 350]
main_contour = max(valid_contours, key=cv2.contourArea)

# --- XOAY ẢNH (ALIGN) ---
rect = cv2.minAreaRect(main_contour)
angle = rect[2]
if angle < -45: angle = 90 + angle
center = tuple(rect[0])
(h, w) = img.shape[:2]  # Lấy kích thước ảnh gốc

# --- BẮT ĐẦU SỬA LỖI ---
# 1. Tính kích thước khung hình mới sau khi xoay
abs_cos = abs(np.cos(np.deg2rad(angle)))
abs_sin = abs(np.sin(np.deg2rad(angle)))
new_w = int(h * abs_sin + w * abs_cos)
new_h = int(h * abs_cos + w * abs_sin)

# 2. Tạo ma trận xoay từ tâm của contour
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# 3. Dịch chuyển nội dung ảnh để nó nằm chính giữa khung hình mới
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

# 4. Thực hiện xoay với kích thước mới (new_w, new_h) thay vì (w, h)
rotated_roi = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
# --- KẾT THÚC SỬA LỖI ---

# --- BƯỚC 6: Tìm lại Contour trên ảnh xoay ---
gray_rot = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)

hsv_rot = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2HSV)
mask_hsv_rot = cv2.inRange(hsv_rot, LOWER_COLOR, UPPER_COLOR)
mask_hsv_rot = cv2.morphologyEx(mask_hsv_rot, cv2.MORPH_CLOSE, kernel_hsv)
mask_hsv_rot = cv2.morphologyEx(mask_hsv_rot, cv2.MORPH_OPEN, kernel_hsv)

# === QUAN TRỌNG: Dilate Mask cho lần xoay này luôn ===
mask_hsv_rot = cv2.dilate(mask_hsv_rot, dilate_kernel, iterations=2)
# =======================================================

edges_rot = cv2.Canny(gray_rot, 50, 180)
edges_rot_clean = cv2.bitwise_and(edges_rot, edges_rot, mask=mask_hsv_rot)
edges_rot_clean = cv2.dilate(edges_rot_clean, kernel_repair, iterations=1)

c_rot, _ = cv2.findContours(edges_rot_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

valid_rot = [cnt for cnt in c_rot if cv2.contourArea(cnt) > 350]
final_contour = max(valid_rot, key=cv2.contourArea)

img_color = img.copy()
if contours:
    main_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img_color, [main_contour], -1, (0, 0, 255), 2)


plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(mask_hsv, cmap='gray'), plt.title('Mask (Da dilate)'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(edges_rot_clean, cmap='gray'), plt.title('Sau Rotate + Mask'), plt.xticks([]), plt.yticks([])
cv2.imshow("Contour Check", img_color)
plt.show()
