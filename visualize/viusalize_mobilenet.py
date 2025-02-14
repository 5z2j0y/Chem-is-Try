import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. 加载模型
model = load_model(r"models\mobile_net_v3\vessel_pre_box_mobilenet_large.keras")
model.summary()

# 2. 加载和预处理图像
image_path = r'images\rect_not_seperated\origin_img\image2.png'
original_image = Image.open(image_path).convert('RGB')
resized_image = original_image.resize((224, 224), Image.BILINEAR) # 确保与训练时的图像尺寸一致
image_array = np.array(resized_image) / 255.0
input_image = np.expand_dims(image_array, axis=0) # 扩展维度以匹配模型输入

# 3. 确定 Grad-CAM 的目标层 (需要根据您的模型结构调整)
# 运行 model.summary() 查看模型结构，选择合适的卷积层
target_layer_name = 'expanded_conv_14_project' # <---  请根据 model.summary() 的输出结果修改

# 4. 构建 Grad-CAM 模型
target_layer = model.get_layer(target_layer_name)
grad_model = tf.keras.models.Model(
    inputs=[model.inputs], # 模型的原始输入
    outputs=[target_layer.output, model.output[1]] # 目标层的输出 和 模型的类别预测输出 (class_output)
)

# 5. 计算梯度和生成 Grad-CAM
with tf.GradientTape() as tape:
    conv_output, class_prediction = grad_model(input_image) # 前向传播，获取目标层输出和类别预测
    class_index = np.argmax(class_prediction[0]) # 获取预测类别索引 (这里取概率最高的类别)
    loss = class_prediction[0][class_index] # 获取预测类别的分数 (loss)

output = conv_output[0] # 目标层输出的特征图 (去除 batch 维度)
grads = tape.gradient(loss, conv_output)[0] # 计算 loss 对 conv_output 的梯度 (去除 batch 维度)

# 全局平均池化梯度，得到特征图权重
pooled_grads = tf.reduce_mean(grads, axis=(0, 1)).numpy()

# 特征图加权
heatmap = output @ pooled_grads[..., tf.newaxis] #  等价于 output * pooled_grads.reshape(1,1,-1)  但更高效
heatmap = tf.squeeze(heatmap).numpy() # 去除维度

# ReLU 激活
heatmap = np.maximum(heatmap, 0) / np.max(heatmap) # ReLU 并归一化到 [0, 1]

# 6. 热力图后处理和叠加
import cv2
heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height)) # 调整热力图大小到原始图像尺寸
heatmap_uint8 = np.uint8(255 * heatmap_resized) # 转换为 0-255 的 uint8 格式
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET) # 应用 JET 彩色映射

# 将热力图强度调整到 0.5 左右，并叠加到原始图像上
superimposed_image = cv2.addWeighted(np.array(original_image), 0.5, heatmap_color, 0.5, 0)


# 7. 可视化结果
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap='jet') # 显示原始热力图 (调整大小后)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(superimposed_image)
plt.title('Superimposed Image')
plt.axis('off')

plt.tight_layout()
plt.show()