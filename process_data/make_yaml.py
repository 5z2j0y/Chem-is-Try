import os
import yaml

# 定义类别和标签映射
label_map = {
    "beaker": 0,
    "volumetric flask": 1,
    "graduated cylinder": 2
}

# 指定图片和txt文件所在的文件夹
data_folder = r'process_data\test\trashcan'

# 获取所有图片和txt文件
images = [f for f in os.listdir(data_folder) if f.endswith('.jpg') or f.endswith('.png')]
annotations = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

# 生成yaml文件内容
data = {
    'train': os.path.join(data_folder, 'train'),
    'val': os.path.join(data_folder, 'val'),
    'nc': len(label_map),
    'names': list(label_map.keys())
}

# 保存yaml文件
with open('mydata.yaml', 'w') as file:
    yaml.dump(data, file)

print("YAML文件已生成")