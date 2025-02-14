import cv2
from ultralytics import YOLO
import random

# 加载模型
model = YOLO(r"models\yolo\examples\vessels-seg.pt", verbose=False) 

# 定义一个函数来生成随机颜色
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 字典来存储类别和颜色的映射
category_colors = {}

# 打开视频捕捉
video_cap = cv2.VideoCapture(1)

while video_cap.isOpened():
    success, frame = video_cap.read()
    if not success:
        break

    # 进行物体检测
    results = model(frame)

    # 显示结果
    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            score = box.conf.item()
            label = model.names[cls_id]

            # 获取颜色，如果类别没有颜色则生成一个新的
            if label not in category_colors:
                category_colors[label] = get_random_color()
            color = category_colors[label]

            # 获取边框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video_cap.release()
cv2.destroyAllWindows()