import os
from PIL import Image

def convert_and_rename_images(input_folder, output_folder, start_number=2000):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有jpg文件
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    # 转换和重命名
    for index, jpg_file in enumerate(jpg_files, start=start_number):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, jpg_file)
        output_path = os.path.join(output_folder, f'image{index}.png')
        
        # 打开图片并转换
        try:
            with Image.open(input_path) as img:
                # 转换为RGB模式（以防是RGBA）
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                # 保存为PNG
                img.save(output_path, 'PNG')
                print(f'已转换: {jpg_file} -> image{index}.png')
        except Exception as e:
            print(f'处理 {jpg_file} 时出错: {str(e)}')

if __name__ == '__main__':
    # 设置输入和输出文件夹路径
    input_folder = r"images\20250209"
    output_folder = r"images\20250209_png"
    
    # 执行转换
    convert_and_rename_images(input_folder, output_folder)
