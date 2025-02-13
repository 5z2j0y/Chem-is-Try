import os
import cv2
import random
import numpy as np
import albumentations as A
def create_aug_folder(source_dir):
    aug_dir = os.path.join(source_dir, "yolo_aug_img_txt")
    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)
    return aug_dir

def get_image_number(filename):
    # 从文件名提取数字，例如从 "image1.jpg" 提取 "1"
    return ''.join(filter(str.isdigit, filename))

def process_yolo_annotation(bbox, augmentation):
    # 将YOLO格式转换为albumentations格式 (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
    x_center, y_center, width, height = bbox
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    
    transformed_bbox = augmentation['bboxes'][0]
    x_min, y_min, x_max, y_max = transformed_bbox
    
    # 转换回YOLO格式
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

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
        txt_file = base_name + '.txt'
        txt_path = os.path.join(input_dir, txt_file)
        
        if not os.path.exists(txt_path):
            continue
            
        # 读取图片和标注
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open(txt_path, 'r') as f:
            annotations = f.readlines()
        
        # 对每个图像进行多次增强
        for aug_idx in range(aug_per_image):
            bboxes = []
            class_labels = []
            
            for ann in annotations:
                class_id, *bbox = map(float, ann.strip().split())
                bboxes.append(bbox)
                class_labels.append(class_id)
            
            # 应用增强
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']
            
            # 保存增强后的图像
            number = get_image_number(img_file)
            new_img_name = f"image_aug{number}_{aug_idx}{os.path.splitext(img_file)[1]}"
            new_txt_name = f"image_aug{number}_{aug_idx}.txt"
            
            new_img_path = os.path.join(aug_dir, new_img_name)
            new_txt_path = os.path.join(aug_dir, new_txt_name)
            
            # 保存图像
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_img_path, aug_image)
            
            # 保存标注
            with open(new_txt_path, 'w') as f:
                for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                    f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")

if __name__ == "__main__":
    input_directory = r"images\rect_not_seperated\yolo_img_txt"
    augment_data(input_directory)
    print("数据增强完成！")
