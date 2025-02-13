import os
import random
import shutil
from pathlib import Path

def divide_dataset(source_dir, valid_dir, valid_ratio=0.2):
    """
    将数据集按比例划分为训练集和验证集
    
    Args:
        source_dir (str): 源数据目录，包含图片和标注文件
        valid_dir (str): 验证集目录
        valid_ratio (float): 验证集比例，默认0.2，即20%
    """
    # 确保目标目录存在
    valid_img_dir = os.path.join(valid_dir, 'images')
    valid_label_dir = os.path.join(valid_dir, 'labels')
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_label_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = []
    for ext in ['.jpg', '.png']:
        image_files.extend(list(Path(source_dir).glob(f'*{ext}')))
    
    # 计算验证集数量
    valid_count = int(len(image_files) * valid_ratio)
    
    # 随机选择验证集文件
    valid_files = random.sample(image_files, valid_count)
    
    # 移动文件
    for img_path in valid_files:
        # 获取对应的标注文件路径
        txt_path = img_path.with_suffix('.txt')
        
        if not txt_path.exists():
            print(f"警告: 找不到对应的标注文件 {txt_path}")
            # continue
            
        # 移动图片
        shutil.move(str(img_path), os.path.join(valid_img_dir, img_path.name))
        
        # 移动标注文件
        # shutil.move(str(txt_path), os.path.join(valid_label_dir, txt_path.name))
        
        print(f"已移动: {img_path.name} 和对应的标注文件")
    
    print(f"\n完成划分!")
    print(f"验证集数量: {valid_count}")
    print(f"训练集数量: {len(image_files) - valid_count}")

if __name__ == "__main__":
    # 设置路径和划分比例
    SOURCE_DIR = r"images\rect_not_seperated\origin_img"  # 替换为你的源数据目录
    VALID_DIR = r"images\rect_not_seperated\coco_img_label\datasets\valid"    # 替换为你的验证集目录
    VALID_RATIO = 0.3  # 验证集比例为20%
    
    divide_dataset(SOURCE_DIR, VALID_DIR, VALID_RATIO)