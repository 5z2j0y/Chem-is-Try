import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 定义类别颜色映射
LABEL_COLORS = {
    'graduated cylinder': 'red',
    'beaker': 'blue',
    'volumetric flask': 'green'
}

def load_and_draw(img_path, json_path):
    """加载图片和JSON标注，并在图片上绘制边界框"""
    # 读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 读取JSON标注
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    return img, json_data['shapes']

def visualize_annotations(input_dir, num_images=20):
    """可视化指定数量的随机图片及其标注"""
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机抽取指定数量的图片
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
    num_images = len(image_files)
    
    # 创建4x5的子图布局
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Random Sample Annotation Visualization', fontsize=16)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], color=color, label=label, linewidth=2)
                      for label, color in LABEL_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.99, 0.98))
    
    # 展平axes数组以便遍历
    axes_flat = axes.flatten()
    
    # 处理每张图片
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        json_path = os.path.join(input_dir, os.path.splitext(img_file)[0] + '.json')
        
        if not os.path.exists(json_path):
            continue
            
        # 加载图片和标注
        img, shapes = load_and_draw(img_path, json_path)
        
        # 在当前子图上显示图片
        ax = axes_flat[idx]
        ax.imshow(img)
        
        # 绘制所有边界框
        for shape in shapes:
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 获取类别对应的颜色
                color = LABEL_COLORS.get(shape['label'], 'yellow')  # 默认黄色
                
                # 计算矩形的参数
                width = x2 - x1
                height = y2 - y1
                
                # 创建矩形patch
                rect = Rectangle((x1, y1), width, height,
                               fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                
                # 添加标签文本
                ax.text(x1, y1-5, shape['label'], 
                       color=color, fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # 设置子图标题
        ax.set_title(f'{img_file}', fontsize=8)
        ax.axis('off')
    
    # 隐藏空白子图
    for idx in range(num_images, 20):
        axes_flat[idx].axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    input_directory = r"images\rect_not_seperated\coco_img_label\datasets\train"
    visualize_annotations(input_directory)
    print("可视化检查完成！")
