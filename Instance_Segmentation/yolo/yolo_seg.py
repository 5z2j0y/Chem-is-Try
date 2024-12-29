import cv2
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO(r"models\yolo\examples\vessels-seg.pt")
video_path = r"test.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

conf = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
