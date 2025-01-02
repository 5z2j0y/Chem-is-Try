from ultralytics import YOLO

# 加载预训练模型
model = YOLO(r'models\yolo\examples\yolov11s.pt')

# 训练模型
model.train(data=r'process_data\test\trashcan\mydata_pc.yaml', epochs=100
            , imgsz=320)
