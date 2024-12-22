from ultralytics import YOLO

# 加载预训练模型
model = YOLO(r'models\yolo\examples\yolov11s-seg.pt')

# 训练模型
model.train(data=r'datasets\new_vessels\mydata.yaml', epochs=2
            , imgsz=640)

