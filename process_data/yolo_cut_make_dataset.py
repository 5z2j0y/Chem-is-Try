import cv2
import os
import glob
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO(r'models/yolo/vessels-nano-box.pt')

# 设置参数
CONFIDENCE_THRESHOLD = 0.3
TARGET_CLASS = 0  # 可以手动修改这个值来指定要检测的类别
INPUT_DIR = r'images\20250209_png'  # 输入文件夹路径
OUTPUT_DIR = r'images\landmark\origin_cropped_img_4'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取输入文件夹中的所有image*.png文件
image_files = glob.glob(os.path.join(INPUT_DIR, 'image*.png'))

for image_path in image_files:
    # 从文件名中提取编号 - 修改这部分
    base_name = os.path.basename(image_path)  # 获取文件名（不含路径）
    number = base_name.replace('image', '').replace('.png', '')
    
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片：{image_path}")
        continue
        
    # 执行YOLO检测
    results = model(frame)
    
    # 处理检测结果
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        
        # 筛选指定类别且置信度大于阈值的检测框
        valid_boxes = []
        for box in boxes:
            if box.cls.cpu().numpy()[0] == TARGET_CLASS and box.conf.cpu().numpy()[0] > CONFIDENCE_THRESHOLD:
                valid_boxes.append(box)
        
        if valid_boxes:
            # 按照置信度排序
            valid_boxes.sort(key=lambda x: x.conf.cpu().numpy()[0], reverse=True)
            best_box = valid_boxes[0]
            
            # 获取边界框坐标
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            # 确保坐标为整数并在图像范围内
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            # 裁剪量筒区域
            cropped_cylinder_region = frame[y1:y2, x1:x2]
            
            # 保存裁剪后的图片
            output_filename = os.path.join(OUTPUT_DIR, f'image_cropped{number}.png')
            if cropped_cylinder_region.size > 0:
                cv2.imwrite(output_filename, cropped_cylinder_region)
                print(f"已保存裁剪图片：{output_filename}")
        else:
            print(f"图片 {image_path} 未找到符合条件的检测框")
    else:
        print(f"图片 {image_path} 未检测到任何目标")