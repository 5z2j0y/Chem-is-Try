import cv2
from ultralytics import YOLO
import random

# 加载模型
model = YOLO(r"models\yolo\examples\yolo11n-pose.pt", verbose=False) 

conf = 0.5

# 打开视频捕捉
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

        # 进行YOLO预测
    results = model.predict(frame, conf=conf)
    
    for result in results:
        # 绘制结果
        frame = result.plot()
            
    cv2.imshow('YOLO Segmentation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()