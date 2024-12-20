# 我的电脑不知为何安装不了mediapipe_model_maker，所以我只能在kaggle上运行这个代码
# emo了呜呜
# 详见https://www.kaggle.com/code/codingbearhsun/chem-is-try/

import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

# 设置数据集路径
train_dataset_path = r"/kaggle/input/trash-3/trash3/train"
validation_dataset_path = r"/kaggle/input/trash-3/trash3/valid"

# 设置缓存路径
cache_dir_train = "/kaggle/working/cache/train"
cache_dir_valid = "/kaggle/working/cache/valid"

# 加载数据集
train_data = object_detector.Dataset.from_coco_folder(
    train_dataset_path, cache_dir=cache_dir_train
)
validation_data = object_detector.Dataset.from_coco_folder(
    validation_dataset_path, cache_dir=cache_dir_valid
)

# 检查数据集大小
print(f"Train dataset size: {train_data.size}")
print(f"Validation dataset size: {validation_data.size}")

spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(
    export_dir='exported_model', 
    learning_rate= 0.05, 
    batch_size=16, 
    epochs=50,
    cosine_decay_epochs=50,
    cosine_decay_alpha=0.1,
)
# 创建模型
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

# 训练模型
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# 评价模型性能
loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"COCO metrics: {coco_metrics}")

# 导出模型
model.export_model()