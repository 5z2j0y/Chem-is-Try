import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载模型 (确保路径正确)
model = torch.load(r'models\yolo\vessels-nano-box.pt')['model']
model.eval() # 设置为评估模式，不进行梯度计算

# **[修改]** 检查是否有 CUDA (GPU)，如果有则使用 GPU，否则使用 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).half() # **[修改]** 将模型移动到 GPU (或 CPU) 并转换为半精度


# 2. 选择要可视化的层 (根据您第一步的观察结果修改)
target_layer_name = 'model.23.cv3.2.2' # 请替换为您实际选择的层名

target_layer = None
# 遍历模型的 named_modules 找到目标层
for name, module in model.named_modules():
    if name == target_layer_name:
        target_layer = module
        break

if target_layer is None:
    raise ValueError(f"未找到名为 {target_layer_name} 的层")

# 3. 定义特征提取hook
output_features = None # 用于存储目标层输出的特征图

def feature_hook(module, input, output):
    global output_features
    output_features = output # 将层的输出保存在全局变量中

# 注册hook到目标层
hook = target_layer.register_forward_hook(feature_hook)

# 4. 图像预处理
img_path = r'images\rect_not_seperated\origin_img\image2.png' # 您的图像路径
img_pil = Image.open(img_path)

# YOLOv5 预处理通常包括 Resize, Normalize 等。需要根据您模型训练时的预处理方式来确定。
# 这里假设是标准的 YOLOv5 预处理，您可能需要根据您的实际情况调整
transform = transforms.Compose([
    transforms.Resize((640, 640)), # 假设输入尺寸是 640x640，根据您的模型调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 预训练模型的归一化参数
])

img_tensor = transform(img_pil).unsqueeze(0).to(device).half() # **[修改]** 将输入张量移动到 GPU (或 CPU) 并转换为半精度


# 5. 前向传播并获取特征图
with torch.no_grad(): # 禁用梯度计算
    model(img_tensor) #  前向传播，hook 会自动保存目标层的输出

hook.remove() # 移除hook，避免影响后续操作

if output_features is None:
    raise ValueError("未能获取目标层的特征输出")

# 6. 生成热力图 (代码保持不变)
# 通常取特征图在通道维度上的平均值，然后进行归一化
heatmap = output_features.squeeze().float().mean(dim=0).cpu().numpy() # **[修改]** 将特征图转换为 float() 以便后续计算，因为 numpy 默认使用 float64

# 归一化到 [0, 1]
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

# 7. 热力图叠加到原图 (修改了这部分)
heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img_pil.width, img_pil.height), Image.Resampling.BILINEAR)) # 调整热力图大小到原始图像大小,  **[修改]** 使用 Image.Resampling.BILINEAR 替换 Image.LINEAR
heatmap_colormap = plt.cm.jet(heatmap_resized)[:,:,:3] # 使用 jet 颜色映射
heatmap_image = Image.fromarray(np.uint8(heatmap_colormap * 255))

# 将热力图叠加到原始图像上 (代码保持不变)
overlaid_image = Image.blend(img_pil.convert('RGB'), heatmap_image, alpha=0.5)

# 8. 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_pil)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlaid_image)
plt.title(f'Heatmap from Layer: {target_layer_name}')
plt.axis('off')

plt.tight_layout()
plt.show()

# 如果您想保存结果，可以使用 plt.savefig('heatmap_visualization.png')