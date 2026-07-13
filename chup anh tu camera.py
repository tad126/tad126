import cv2
import os
from datetime import datetime

# Tạo thư mục lưu ảnh
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)

# Mở camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

print("Nhấn X để chụp ảnh")
print("Nhấn Q để thoát")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được camera")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Nhấn X để chụp
    if key == ord('x'):
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        filepath = os.path.join(save_folder, filename)

        cv2.imwrite(filepath, frame)
        print(f"Đã lưu: {filepath}")

    # Nhấn Q để thoát
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()