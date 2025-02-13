import json
import os
from pathlib import Path

def convert_labelme_to_yolo(json_path):
    # 类别映射
    class_map = {
        'graduated cylinder': 0,
        'beaker': 1,
        'volumetric flask': 2
    }
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图片尺寸
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # 处理每个标注
    yolo_annotations = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            continue
            
        points = shape['points']
        # LabelMe标注格式为[[x1,y1], [x2,y2]]
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # 计算YOLO格式的中心点和宽高
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        # 确保值在0-1之间
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)
        
        # YOLO格式：<class> <x_center> <y_center> <width> <height>
        yolo_line = f"{class_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def main():
    # 指定JSON文件所在目录
    json_dir = Path(r"images\rect_not_seperated")
    
    # 创建yolo_labels目录
    yolo_dir = json_dir / 'yolo_labels'
    yolo_dir.mkdir(exist_ok=True)
    
    # 处理所有JSON文件
    for json_path in json_dir.glob('*.json'):
        try:
            # 转换标注
            yolo_annotations = convert_labelme_to_yolo(json_path)
            
            # 保存为同名txt文件
            txt_path = yolo_dir / (json_path.stem + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))
                
            print(f"成功转换: {json_path.name}")
        except Exception as e:
            print(f"处理 {json_path.name} 时出错: {str(e)}")

if __name__ == '__main__':
    main()
