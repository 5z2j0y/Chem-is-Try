import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.src.saving.legacy import load_model_legacy

model = load_model_legacy("models/mobile_net_v3/optimized_mobilenetv3_detector.h5")


# 获取输入图像尺寸
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)  # 加载并调整图片尺寸
    img = img_to_array(img) / 255.0  # 归一化
    img = np.expand_dims(img, axis=0)  # 添加 batch 维度
    return img

image_path = r"images\rect_not_seperated\origin_img\image2.png"  # 替换成你的图片路径
img = preprocess_image(image_path, IMAGE_SIZE)

# 获取特征提取层
feature_extractor = Model(inputs=model.input, outputs=model.get_layer("Conv_1").output)  # 选择最后的卷积层
features = feature_extractor.predict(img)  # 提取特征

# 可视化特征图
def visualize_feature_maps(feature_maps, num_cols=8):
    num_filters = feature_maps.shape[-1]  # 通道数
    num_rows = num_filters // num_cols  # 计算行数

    plt.figure(figsize=(15, num_rows * 2))
    for i in range(num_filters):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.show()

visualize_feature_maps(features)

def compute_gradcam(model, img, layer_name="Conv_1"):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = tf.reduce_max(predictions[:, 4:])  # 仅关注类别部分

    grads = tape.gradient(loss, conv_outputs)  # 计算梯度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # 全局平均池化

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]  # 计算权重
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # 归一化

    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.5):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, IMAGE_SIZE)

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # 归一化到 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 伪彩色映射

    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)  # 叠加

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

heatmap = compute_gradcam(model, img)
overlay_heatmap(image_path, heatmap)
