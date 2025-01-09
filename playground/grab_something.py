import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import threading
from queue import Queue
import time

# 初始化手势检测器
base_options = python.BaseOptions(model_asset_path=r'models\hand_landmark\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
hand_detector = vision.HandLandmarker.create_from_options(options)

# 初始化YOLO模型
yolo_model = YOLO(r"models\yolo\examples\yolov11s-seg.pt")

# 创建共享队列
frame_queue = Queue(maxsize=2)
hand_result_queue = Queue(maxsize=2)
seg_result_queue = Queue(maxsize=2)

def process_hand_detection():
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
            
        frame = frame_queue.get()
        if frame is None:
            break
            
        # 手势检测处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = hand_detector.detect(mp_image)
        hand_result_queue.put(detection_result)

def process_segmentation():
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
            
        frame = frame_queue.get()
        if frame is None:
            break
            
        # YOLO分割处理
        results = yolo_model.predict(frame, conf=0.5)
        seg_result_queue.put(results[0])

def draw_hand_landmarks(image, hand_landmarks):
    for hand_landmarks in hand_landmarks:
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
        connections = mp.solutions.hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (
                int(hand_landmarks[start_idx].x * image.shape[1]),
                int(hand_landmarks[start_idx].y * image.shape[0])
            )
            end_point = (
                int(hand_landmarks[end_idx].x * image.shape[1]),
                int(hand_landmarks[end_idx].y * image.shape[0])
            )
            
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    return image

def is_grabbing(hand_landmarks, boxes, img_width, img_height, class_names):
    for box in boxes:
        box_cls_id = int(box.cls[0])
        # 跳过 person 类别（通常是第0类）
        if box_cls_id == 0:
            continue
        label = class_names[box_cls_id]
        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        for landmarks_of_one_hand in hand_landmarks:
            count = 0
            for lm in landmarks_of_one_hand:
                lx, ly = int(lm.x * img_width), int(lm.y * img_height)
                if x1 <= lx <= x2 and y1 <= ly <= y2:
                    count += 1
                    if count >= 3:
                        print(f"抓取了 {label}")
                        return True
    return False

def main():
    # 创建并启动处理线程
    hand_thread = threading.Thread(target=process_hand_detection)
    seg_thread = threading.Thread(target=process_segmentation)
    hand_thread.start()
    seg_thread.start()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    class_names = []
    with open(r"coco.names", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将当前帧放入队列
        frame_queue.put(frame.copy())
        frame_queue.put(frame.copy())  # 为两个处理线程各放入一份

        # 获取处理结果
        if not hand_result_queue.empty():
            hand_result = hand_result_queue.get()
            if hand_result.hand_landmarks:
                frame = draw_hand_landmarks(frame, hand_result.hand_landmarks)

        if not seg_result_queue.empty():
            seg_result = seg_result_queue.get()
            # 将分割结果叠加到原图上
            seg_frame = seg_result.plot()
            frame = cv2.addWeighted(frame, 0.5, seg_frame, 0.5, 0)
            if hand_result.hand_landmarks:
                is_grabbing(hand_result.hand_landmarks, seg_result.boxes, frame.shape[1], frame.shape[0], class_names)

        cv2.imshow('Combined Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 终止处理线程
    frame_queue.put(None)
    frame_queue.put(None)
    hand_thread.join()
    seg_thread.join()

if __name__ == '__main__':
    main()