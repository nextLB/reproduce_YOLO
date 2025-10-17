
"""
    YOLOV1复现项目的路径配置
"""

# 2025.10.17 (V1.0)            --- by next, 初步实现了YOLOv1的路径配置文件









import os

# 数据集的路径配置
DATA_PATH = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'VOC2007_train_and_val')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'VOC2007_test')


