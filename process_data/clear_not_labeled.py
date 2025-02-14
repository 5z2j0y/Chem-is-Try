import os
import glob

def clear_unlabeled_images(directory):
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(directory, "*.png"))
    image_files.extend(glob.glob(os.path.join(directory, "*.jpg")))
    
    removed_count = 0
    
    for image_path in image_files:
        # 构建对应的json文件路径
        json_path = os.path.splitext(image_path)[0] + '.json'
        
        # 如果不存在对应的json文件，删除图片
        if not os.path.exists(json_path):
            try:
                os.remove(image_path)
                print(f"已删除: {image_path}")
                removed_count += 1
            except Exception as e:
                print(f"删除失败 {image_path}: {str(e)}")
    
    print(f"\n清理完成! 总共删除了 {removed_count} 个未标注的图片文件。")
    print(f"还剩下 {len(image_files) - removed_count} 个文件。")

if __name__ == "__main__":
    # 指定要处理的目录路径
    target_directory = r"images\low_quality_rect\origin_img"
    
    if os.path.exists(target_directory):
        print(f"开始处理目录: {target_directory}")
        clear_unlabeled_images(target_directory)
    else:
        print("错误: 指定的目录不存在!")
