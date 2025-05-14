import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import os

# Tạo thư mục nếu chưa có
os.makedirs('images', exist_ok=True)

# Khởi tạo mô hình
coco_model = YOLO('yolov8n.pt')  # Phát hiện phương tiện
plate_model = YOLO('license_plate_detector.pt')  # Phát hiện biển số
ocr_reader = easyocr.Reader(['en'])  # OCR tiếng Anh

# Đọc ảnh
img = cv2.imread('./anh1.jpg')
vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Phát hiện phương tiện
coco_detections = coco_model(img)[0]
for det in coco_detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    if int(class_id) in vehicles:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Phát hiện biển số
plate_detections = plate_model(img)[0]
for i, det in enumerate(plate_detections.boxes.data.tolist()):
    x1, y1, x2, y2, score, class_id = det

    # Cắt ảnh biển số
    plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Resize để dễ đọc hơn
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Threshold
    _, thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY_INV)

    # Cắt làm 2 phần để đọc 2 dòng
    h, w = thresh.shape
    upper_half = thresh[0:int(h/2), :]
    lower_half = thresh[int(h/2):, :]

    # OCR từng dòng
    upper_text = ocr_reader.readtext(upper_half, detail=0)
    lower_text = ocr_reader.readtext(lower_half, detail=0)

    # Ghép text
    plate_text = ''
    if upper_text:
        plate_text += upper_text[0].strip()
    if lower_text:
        plate_text += ' ' + lower_text[0].strip()

    # Vẽ lên ảnh gốc
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, plate_text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị và lưu ảnh threshold
    cv2.imshow(f'Thresh Plate {i}', thresh)
    cv2.imwrite(f'images/thresh_plate_{i}.png', thresh)

# Lưu ảnh kết quả
cv2.imwrite('images/result_detected.png', img)

# Hiển thị ảnh kết quả
cv2.imshow('Detected Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
