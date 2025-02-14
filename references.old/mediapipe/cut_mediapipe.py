import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe
BaseOptions = mp.tasks.BaseOptions
Detection = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 配置模型选项
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/mediapipe/vessels.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    max_results=1,  # 只获取最可能的一个检测结果
    score_threshold=0.5  # 设置检测阈值
)

# 创建检测器并打开摄像头
with Detection.create_from_options(options) as detector:
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
            
        # 转换图像格式为 MediaPipe 要求的格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # 执行检测
        detection_result = detector.detect(mp_image)
        
        # 处理检测结果
        if detection_result.detections:
            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            
            # 获取边界框坐标
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # 裁剪量筒区域
            cropped_cylinder_region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # 可视化：在原始帧上画框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
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