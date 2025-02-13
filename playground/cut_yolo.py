import cv2
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('models/yolo/vessels-nano-box.pt')

# 打开摄像头
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面")
        break
        
    # 执行YOLO检测
    results = model(frame)
    
    # 处理检测结果
    if len(results) > 0 and len(results[0].boxes) > 0:
        # 获取得分最高的检测框
        boxes = results[0].boxes
        box = boxes[0]  # 取第一个检测结果
        
        # 获取边界框坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # 确保坐标为整数并在图像范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        
        # 裁剪量筒区域
        cropped_cylinder_region = frame[y1:y2, x1:x2]
        
        # 在原始帧上画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 如果裁剪区域有效，显示它
        if cropped_cylinder_region.size > 0:
            cv2.imshow('Cropped Cylinder Region', cropped_cylinder_region)
    
    # 显示原始画面
    cv2.imshow('Original Frame with Detection', frame)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()