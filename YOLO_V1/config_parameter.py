

"""
    YOLOV1复现项目的参数配置
"""

# 数据集中的图像参数配置
IMAGE_SIZE = (448, 448)
S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset


# 训练集与验证集的比例配置
RATIO = 0.3


# 训练时的参数配置
MAX_EPOCHS = 10
BATCH_SIZE = 2
NUM_WORKERS = 2
LEARNING_RATE = 0.00001
EPSILON = 0.0000001



