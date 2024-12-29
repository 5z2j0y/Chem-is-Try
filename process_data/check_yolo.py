import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def get_color_map(num_classes):
    random.seed(42)
    return {i: (random.random(), random.random(), random.random()) for i in range(num_classes)}

def visualize_yolo(images_dir, labels_dir, num_images=20, num_classes=15):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)
    selected_images = image_files[:num_images]

    color_map = get_color_map(num_classes)

    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    axes = axes.flatten()

    for ax, image_file in zip(axes, selected_images):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        with Image.open(image_path) as img:
            ax.imshow(img)
            width, height = img.size

            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    for line in lf:
                        category_id, x_center, y_center, w, h = map(float, line.strip().split())
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        x = x_center - w / 2
                        y = y_center - h / 2

                        color = color_map[int(category_id)]
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

        ax.axis('off')
        ax.set_title(image_file)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images_dir = r'process_data\test\train\images'
    labels_dir = r'process_data\test\train\yolo_labels'
    visualize_yolo(images_dir, labels_dir)