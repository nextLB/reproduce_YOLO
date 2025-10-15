"""
    复现YOLOv3时的全局参数配置文件
"""

# 数据集参数配置
IMAGE_SIZE = 640
GRID_SIZE = [32, 16, 8]  # 对应不同尺度的网格大小
RATIO = 0.15  # 验证集与训练集的划分比例
RANDOM_CROP_RATIO = 0.5  # 构建数据集时对于数据进行随机裁剪的比例
RANDOM_AFFINE_RATIO = 0.5  # 构建数据集时对于数据进行仿射变换的比例
RANDOM_COLOR_RATIO = 0.5  # 对数据集进行随机颜色调整的比例
RANDOM_LOAD_IMAGE_RATIO = 0.5  # 构建数据集时对于数据以不同的加载方式进行加载的比例
RANDOM_CROP_SCOPE = (0.3, 1)       # 构建数据集时对于数据进行随机裁剪的比例范围
TARGETS_SIZE = 200      # 构建目标数据集时的检测目标的长度
DEGREES = 10         # 对数据集进行仿射变换时的旋转角度范围设定
TRANSLATE = 0.1       # 对数据集进行仿射变换时的平移范围设定
SCALE = 0.1    # 对数据集进行仿射变换时的缩放范围设定
SHEAR = 10  # 对数据集进行仿射变换时的剪切范围设定



# 模型构建时的参数配置
ANCHORS = [
            [(10, 13), (16, 30), (33, 23)],   # P3/8
            [(30, 61), (62, 45), (59, 119)],  # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]
NUM_CLASSES = 80



# 训练参数配置
MAX_EPOCHS = 30
BATCH_SIZE = 2
NUM_WORKERS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.937    # SGD momentum/Adam beta1
WEIGHT_DECAY = 0.0005  # optimizer weight decay 5e-4


# LEARNING_RATE_INIT = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
# LEARNING_RATE_FINAL = 0.01  # final OneCycleLR learning rate (LEARNING_RATE_INIT * LEARNING_RATE_FINAL)


# WARMUP_EPOCHS = 3.0  # warmup epochs (fractions ok)
# WARMUP_MOMENTUM = 0.8  # warmup initial momentum
# WARMUP_BIAS_LEARNING_RATE = 0.1  # warmup initial bias lr




