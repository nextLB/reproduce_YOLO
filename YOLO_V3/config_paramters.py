"""
    复现YOLOv3时的全局参数配置文件
"""

# 数据集参数配置
IMAGE_SIZE = 640
GRID_SIZE = 32
RATIO = 0.3  # 验证集与训练集的划分比例
RANDOM_CROP_RATIO = 0.5  # 构建数据集时对于数据进行随机裁剪的比例
RANDOM_CROP_SCOPE = (0.3, 1)       # 构建数据集时对于数据进行随机裁剪的比例范围


# 训练参数配置
BATCH_SIZE = 4
NUM_WORKERS = 4

# LEARNING_RATE_INIT = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
# LEARNING_RATE_FINAL = 0.01  # final OneCycleLR learning rate (LEARNING_RATE_INIT * LEARNING_RATE_FINAL)
# MOMENTUM = 0.937    # SGD momentum/Adam beta1
# WEIGHT_DECAY = 0.0005  # optimizer weight decay 5e-4
# WARMUP_EPOCHS = 3.0  # warmup epochs (fractions ok)
# WARMUP_MOMENTUM = 0.8  # warmup initial momentum
# WARMUP_BIAS_LEARNING_RATE = 0.1  # warmup initial bias lr




