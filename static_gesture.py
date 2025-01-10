# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from utils.draw_hand import draw_landmarks

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=r'models\hand_landmark\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
        
    # 转换BGR为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为MediaPipe图像格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # 检测手势
    detection_result = detector.detect(mp_image)
    
    # 转换回BGR用于显示
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 绘制手部关键点和连接线
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            landmark_point = []
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmark_point.append((x, y))
            
            # 调用draw_landmarks方法
            image = draw_landmarks(image, landmark_point)
    
    # 显示结果
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
