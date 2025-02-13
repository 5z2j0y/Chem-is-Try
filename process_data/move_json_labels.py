import os
import shutil
from pathlib import Path

def move_json_files(image_dir, json_dir):
    """
    将JSON标注文件移动到对应的图片文件夹
    
    Args:
        image_dir (str): 图片文件夹路径
        json_dir (str): JSON标注文件夹路径
    """
    # 确保路径存在
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    
    if not image_dir.exists() or not json_dir.exists():
        print("图片文件夹或JSON文件夹不存在！")
        return
    
    # 获取所有图片文件
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(image_dir.rglob(ext))
    
    # 处理每个图片文件
    for image_path in image_files:
        # 获取不带扩展名的文件名
        image_stem = image_path.stem
        # 构建对应的JSON文件路径
        json_path = json_dir / f"{image_stem}.json"
        
        if json_path.exists():
            # 复制JSON文件到图片所在文件夹
            destination = image_path.parent / json_path.name
            try:
                shutil.copy2(json_path, destination)
                print(f"成功：复制 {json_path.name} 到 {destination.parent}")
            except Exception as e:
                print(f"错误：复制 {json_path.name} 失败 - {str(e)}")
        else:
            print(f"警告：未找到对应的JSON文件 {json_path.name}")

if __name__ == "__main__":
    # 设置图片文件夹和JSON文件夹的路径
    JSON_DIR= r"images\rect_not_seperated\labelme_json"  # 替换为你的图片文件夹路径
    IMAGE_DIR= r"images\rect_not_seperated\coco_img_label\datasets\train"    # 替换为你的JSON文件夹路径
    
    move_json_files(IMAGE_DIR, JSON_DIR)