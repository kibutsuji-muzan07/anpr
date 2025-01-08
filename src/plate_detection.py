from ultralytics import YOLO
import cv2
from util import *

img = 'Images/car_10.jpg'
frame = cv2.imread(img, cv2.IMREAD_COLOR)
plate_detector = YOLO('C:/Users/bhand/Documents/Codes/anpr/license_plate_detector.pt')
plate = plate_detector(img)[0]
# print(plate.boxes.data.tolist())


for license_plate in plate.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

 # crop license plate
license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

 # process license plate
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

# license_plate_crop_gray = cv2.equalizeHist(license_plate_crop_gray)

_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 55, 255, cv2.THRESH_BINARY)


# license_plate_crop_thresh = cv2.adaptiveThreshold(
#     license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 2
# )

# license_plate_crop_thresh = cv2.GaussianBlur(license_plate_crop_thresh, (5, 5), 0)

# # Morphological Operations
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# license_plate_crop_thresh = cv2.morphologyEx(license_plate_crop_thresh, cv2.MORPH_CLOSE, kernel)

license_number, score = read_license_plate(license_plate_crop_thresh)

print(license_number)
cv2.imshow('YOLOv8 Detection', license_plate_crop_thresh)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()
