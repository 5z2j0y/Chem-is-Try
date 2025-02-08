import cv2
from ultralytics import YOLO
import random

# 加载模型
model = YOLO(r"models\yolo\examples\vessels-seg.pt", verbose=False) 
# model = YOLO("path/to/best.pt")  # 加载自定义训练模型

# 定义一个函数来生成随机颜色
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 字典来存储类别和颜色的映射
category_colors = {}

# 打开视频捕捉
video_cap = cv2.VideoCapture(1)

# 设置视频分辨率为1920*1080
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while video_cap.isOpened():
    success, frame = video_cap.read()
    if not success:
        break
    
    # 计算裁切的起始位置以获取中心区域
    h, w = frame.shape[:2]
    start_x = w//2 - 540  # (1920-1080)/2 = 420
    frame = frame[:, start_x:start_x+1080]  # 裁切成1080*1080
    frame = cv2.resize(frame, (640, 640))  # 缩放成640*640
    # 进行物体检测
    results = model(frame)

    # 显示结果
    annotated_frame = frame.copy()
    max_conf = 0
    max_box = None
    max_label = None
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            score = box.conf.item()
            label = model.names[cls_id]

            # 更新最高置信度的物体信息
            if score > max_conf:
                max_conf = score
                max_box = box
                max_label = label

            # 获取颜色，如果类别没有颜色则生成一个新的
            if label not in category_colors:
                category_colors[label] = get_random_color()
            color = category_colors[label]

            # 获取边框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 如果找到了物体，裁切并显示最高置信度的物体
    if max_box is not None:
        x1, y1, x2, y2 = map(int, max_box.xyxy[0].tolist())
        # 确保坐标在有效范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        # 裁切物体区域
        object_crop = frame[y1:y2, x1:x2]
        if object_crop.size > 0:  # 确保裁切区域有效
            # 调整裁切图像大小，保持长宽比
            max_size = 200
            h, w = object_crop.shape[:2]
            ratio = min(max_size/w, max_size/h)
            new_size = (int(w*ratio), int(h*ratio))
            object_crop = cv2.resize(object_crop, new_size)
            # 显示裁切的物体
            cv2.imshow(f'Highest Confidence Object ({max_label})', object_crop)

    cv2.imshow('Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video_cap.release()
cv2.destroyAllWindows()