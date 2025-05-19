import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from ultralytics import NAS
from ultralytics import YOLO

model = YOLO("/home/skrudfhr02/hdd/rokkie/python_code/yolo_project/Model/yolo_nas_l.pt")

# Dataset folder path
image_folder = "/home/skrudfhr02/hdd/rokkie/python_code/yolo_project/Dataset/M1003"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg") and f.startswith("img")])

# Result video path
result_path = "/home/skrudfhr02/hdd/rokkie/python_code/yolo_project/Result/New_M1003.mp4"

# Read the first image to get frame size
first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
height, width, _ = first_image.shape

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(result_path, fourcc, 30.0, (width, height))

# 클래스별 색상 리스트 (필요에 따라 색상 추가 가능)
class_colors = [
    (255, 0, 0),    # 클래스 0: 빨강
    (0, 255, 0),    # 클래스 1: 초록
    (0, 0, 255),    # 클래스 2: 파랑
    (255, 255, 0),  # 클래스 3: 노랑
    (255, 0, 255),  # 클래스 4: 마젠타
    (0, 255, 255),  # 클래스 5: 시안
    (128, 0, 128),  # 클래스 6: 보라
    (255, 165, 0),  # 클래스 7: 주황
    (0, 128, 128),  # 클래스 8: 청록
    (128, 128, 0),  # 클래스 9: 올리브
]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Run inference
    results = model(image_path)
    image = cv2.imread(image_path)

    # Draw results
    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf[0].item()
        cls = int(det.cls[0].item())
        label = model.names[cls]
        color = class_colors[cls % len(class_colors)]  # 클래스별 색상 선택
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)  # 두께 1로
        cv2.putText(
            image,
            f"{label}:{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # fontScale을 0.7로
            color,
            1,    # thickness를 1로
            cv2.LINE_AA  # 안티앨리어싱 적용
        )

    # Write frame to video
    video_writer.write(image)

video_writer.release()
print(f"Video saved to {result_path}")