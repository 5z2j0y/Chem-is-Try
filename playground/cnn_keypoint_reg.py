import tensorflow as tf
from tensorflow.keras import layers, models

# 构建 Backbone 网络
def build_backbone():
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), # 输入图像尺寸，可以根据您的需求调整
        include_top=False,      # 不包含顶部的全连接层 (我们不需要分类层)
        weights='imagenet'     # 使用 ImageNet 预训练权重 (可选，但通常能提高性能)
    )
    # 获取 Backbone 网络的输出特征图
    backbone_output = backbone.output
    return backbone, backbone_output

backbone_model, backbone_output = build_backbone()

# 构建关键点回归头
def build_keypoint_regression_head(backbone_output, num_keypoints):
    # 卷积层 1
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(backbone_output)
    # 卷积层 2
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    # 关键点坐标预测层 (输出 num_keypoints * 2 个通道)
    keypoints_output = layers.Conv2D(num_keypoints * 2, kernel_size=1, activation=None)(x) # 线性激活，直接输出坐标值

    # 将输出 reshape 成 (batch_size, num_keypoints, 2) 的形状，方便后续计算损失
    keypoints_output = layers.Reshape((-1, 2))(keypoints_output) # -1 表示自动推断维度， 2 表示 x, y 坐标

    return keypoints_output

num_keypoints = 5 # 假设我们要预测 5 个关键点 (例如 顶部边缘点2个，底部边缘点2个，刻度线中心点1个)
keypoints_output = build_keypoint_regression_head(backbone_output, num_keypoints)

# 获取 Backbone 网络的输入层
backbone_input = backbone_model.input

# 构建完整的模型
keypoint_model = models.Model(inputs=backbone_input, outputs=keypoints_output)

# 打印模型结构 (可选)
keypoint_model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # 使用 Adam 优化器，学习率可以调整
loss_function = tf.keras.losses.MeanSquaredError() # 使用均方误差损失函数
metrics = ['MeanAbsoluteError'] # 可以添加评估指标，例如平均绝对误差

keypoint_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)