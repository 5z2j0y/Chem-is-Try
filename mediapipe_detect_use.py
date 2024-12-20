import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import random

model_path = 'models\mediapipe\examples\efficientdet_lite2.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    score_threshold=0.5,
    running_mode=VisionRunningMode.VIDEO)

# 定义一个函数来生成随机颜色
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 字典来存储类别和颜色的映射
category_colors = {}

with ObjectDetector.create_from_options(options) as detector:
    # 检测器已创建，可以开始检测视频
    video_cap = cv2.VideoCapture(0)
    frame_index = 0
    video_file_fps = video_cap.get(cv2.CAP_PROP_FPS)

    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(1000 * frame_index / video_file_fps)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # 绘制检测结果
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            
            # 获取类别和置信度
            category = detection.categories[0].category_name
            score = detection.categories[0].score
            
            # 获取颜色，如果类别没有颜色则生成一个新的
            if category not in category_colors:
                category_colors[category] = get_random_color()
            color = category_colors[category]
            
            # 绘制边框和置信度
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, f'{category}: {score:.2f}', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    video_cap.release()
    cv2.destroyAllWindows()
