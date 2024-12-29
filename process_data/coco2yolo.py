import json
import os
from PIL import Image
from tqdm import tqdm  # 添加进度条库

def coco_to_yolo(json_path, images_dir, labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for image in tqdm(data['images'], desc="Processing images"):  # 添加进度条
        image_id = image['id']
        file_name = image['file_name']
        image_path = os.path.join(images_dir, file_name)
        
        with Image.open(image_path) as img:
            width, height = img.size

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
        
        label_file = os.path.join(labels_dir, os.path.splitext(file_name)[0] + '.txt')
        with open(label_file, 'w') as lf:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height
                lf.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

if __name__ == "__main__":
    json_path = r'process_data\test\valid\labels.json'
    images_dir = r'process_data\test\valid\images'
    labels_dir = r'process_data\test\valid\yolo_labels'
    coco_to_yolo(json_path, images_dir, labels_dir)


