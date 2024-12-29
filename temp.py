import os

# 指定标签文件所在的文件夹路径
labels_dir = r'datasets\trashcan\valid'

# 遍历文件夹中的所有.txt标签文件
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        label_path = os.path.join(labels_dir, filename)
        
        # 打开并读取标签文件
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        # 处理每一行标签
        updated_lines = []
        for line in lines:
            # 分割每一行的内容
            parts = line.strip().split()
            class_id = int(parts[0])
            
            # 将class_id减去1
            if class_id > 0:
                parts[0] = str(class_id - 1)
            
            # 将更新后的标签行加入列表
            updated_lines.append(" ".join(parts) + '\n')
        
        # 将修改后的标签写回文件
        with open(label_path, 'w') as file:
            file.writelines(updated_lines)

print("标签更新完成！")
