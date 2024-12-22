import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, val_ratio=0.2):
    # 获取所有图片文件
    files = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(files)
    
    # 计算验证集的数量
    val_count = int(len(files) * val_ratio)
    
    # 分配文件到验证集
    val_files = files[:val_count]
    train_files = files[val_count:]
    
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 移动文件到验证集文件夹
    for file in val_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))
        txt_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(source_dir, txt_file)):
            shutil.move(os.path.join(source_dir, txt_file), os.path.join(val_dir, txt_file))
    
    # 移动文件到训练集文件夹
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
        txt_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(source_dir, txt_file)):
            shutil.move(os.path.join(source_dir, txt_file), os.path.join(train_dir, txt_file))

# 示例用法
source_dir = r'process_data\test\new_vessels'
train_dir = r'process_data\test\new_vessels\train'
val_dir = r'process_data\test\new_vessels\valid'
split_data(source_dir, train_dir, val_dir)
