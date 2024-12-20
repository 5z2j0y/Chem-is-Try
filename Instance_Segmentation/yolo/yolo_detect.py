import cv2
import random
from ultralytics import YOLO
import numpy as np

model = YOLO(r"models\yolo\examples\yolov11s-seg.pt")
video_path = 0
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # ...existing code...
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            cls = int(box.cls)  # 将 cls 转换为整数
            confidence = float(box.conf)  # 将 confidence 转换为浮点数
            color = colors[cls]
            label = f"{yolo_classes[cls]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # ...existing code...
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()