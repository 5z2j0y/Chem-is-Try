from PIL import Image
import os
from datetime import datetime

# 设置输入和输出文件夹路径
input_folder = r"images\.old"  # 替换为你的输入文件夹路径
output_folder = r"images\20250209"  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取今天的日期作为文件名前缀
today = datetime.now().strftime("%Y%m%d")

# 遍历输入文件夹中的所有jpg文件
for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.lower().endswith('.jpg'):
        # 打开图片
        with Image.open(os.path.join(input_folder, filename)) as img:
            # 计算裁切区域
            width, height = img.size
            left = (width - height) // 2  # 从中间开始裁切
            top = 0
            right = left + height
            bottom = height

            # 裁切图片
            cropped_img = img.crop((left, top, right, bottom))
            
            # 生成新文件名
            new_filename = f"{today}_{idx+1:03d}.jpg"
            
            # 保存裁切后的图片
            cropped_img.save(os.path.join(output_folder, new_filename))

print("图片裁切完成！")
