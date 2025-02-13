import os
import cv2
import json
import random
import base64
import numpy as np
import albumentations as A

def create_aug_folder(source_dir):
    aug_dir = os.path.join(source_dir, "coco_aug_img_label")
    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)
    return aug_dir

def get_image_number(filename):
    # 从文件名提取数字，例如从 "image1.jpg" 提取 "1"
    return ''.join(filter(str.isdigit, filename))

def convert_points_to_bbox(points, img_width, img_height):
    """将labelme的points转换为归一化的YOLO格式bbox"""
    x_min, y_min = points[0]
    x_max, y_max = points[1]
    
    # 归一化坐标
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]

def convert_bbox_to_points(bbox, img_width, img_height):
    """将归一化的YOLO格式bbox转换回labelme的points格式"""
    x_center, y_center, width, height = bbox
    
    # 转换回像素坐标
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    
    return [[x_min, y_min], [x_max, y_max]]

def image_to_base64(image):
    # 将图像转换为base64编码
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def augment_data(input_dir, aug_per_image=3):
    aug_dir = create_aug_folder(input_dir)
    
    # 定义增强pipeline
    transform = A.Compose([
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomShadow(p=0.5),
            A.RandomFog(p=0.5),
            A.RandomSunFlare(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.Sharpen(p=0.5),
            A.Emboss(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        json_file = base_name + '.json'
        json_path = os.path.join(input_dir, json_file)
        
        if not os.path.exists(json_path):
            continue
            
        # 读取图片和JSON标注
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # 在读取JSON数据后，获取图像尺寸
        image_height, image_width = image.shape[:2]
        
        # 对每个图像进行多次增强
        for aug_idx in range(aug_per_image):
            bboxes = []
            class_labels = []
            original_labels = []
            
            # 从JSON中提取标注信息
            for shape in json_data['shapes']:
                # 传入图像尺寸进行归一化
                bbox = convert_points_to_bbox(shape['points'], image_width, image_height)
                bboxes.append(bbox)
                class_labels.append(0)  # 使用统一的类别ID
                original_labels.append(shape['label'])
            
            # 应用增强
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            
            # 准备新的文件名
            number = get_image_number(img_file)
            new_img_name = f"image_aug{number}_{aug_idx}{os.path.splitext(img_file)[1]}"
            new_json_name = f"image_aug{number}_{aug_idx}.json"
            
            new_img_path = os.path.join(aug_dir, new_img_name)
            new_json_path = os.path.join(aug_dir, new_json_name)
            
            # 保存增强后的图像
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_img_path, aug_image_bgr)
            
            # 创建新的JSON数据
            new_json_data = {
                "version": "5.5.0",
                "flags": {},
                "shapes": [],
                "imagePath": new_img_name,
                "imageData": image_to_base64(aug_image_bgr),
                "imageHeight": aug_image.shape[0],
                "imageWidth": aug_image.shape[1]
            }
            
            # 转换增强后的边界框为labelme格式
            for bbox, label in zip(aug_bboxes, original_labels):
                points = convert_bbox_to_points(bbox, aug_image.shape[1], aug_image.shape[0])
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                }
                new_json_data["shapes"].append(shape)
            
            # 保存新的JSON文件
            with open(new_json_path, 'w') as f:
                json.dump(new_json_data, f, indent=2)

if __name__ == "__main__":
    input_directory = r"E:\github_projects\Chem-is-Try\images\rect_not_seperated\coco_img_label\datasets\train"
    augment_data(input_directory)
    print("数据增强完成！")
