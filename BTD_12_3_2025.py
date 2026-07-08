import cv2
import numpy as np

img = cv2.imread("C:/Users/ADMIN/Pictures/IMG_8429.JPG")

# Chuyển sang HSV để lọc theo màu (chính xác hơn grayscale threshold)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Định nghĩa khoảng màu xanh ngọc (teal/cyan) của sản phẩm
# Cần tinh chỉnh giá trị này dựa theo màu thực tế sản phẩm của bạn
lower_teal = np.array([70, 50, 50])
upper_teal = np.array([100, 255, 255])

mask = cv2.inRange(hsv, lower_teal, upper_teal)

# Làm sạch mask: loại nhiễu nhỏ, lấp lỗ hổng
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # lấp lỗ nhỏ bên trong vật thể
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # xóa nhiễu nhỏ bên ngoài

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ contour lớn nhất (chính là sản phẩm) lên ảnh gốc để kiểm tra
img_color = img.copy()
if contours:
    main_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img_color, [main_contour], -1, (0, 0, 255), 2)

cv2.imshow("Mask", mask)
cv2.imshow("Contour Check", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()