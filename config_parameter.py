"""
    复现项目所用到的相关的参数配置
"""


# 构建数据集时所需要用到的参数配置
IMAGE_SIZE = (448, 448)
RATIO = 0.3     # 训练集与验证集的比例设置
BATCH_SIZE = 4
NUM_WORKERS = 4
RANDOM_CROP_RATIO = 0.5  # 构建数据集时对于数据进行随机裁剪的比例
RANDOM_AFFINE_RATIO = 0.5  # 构建数据集时对于数据进行仿射变换的比例
RANDOM_COLOR_RATIO = 0.5  # 对数据集进行随机颜色调整的比例
RANDOM_CROP_SCOPE = (0.3, 1)       # 构建数据集时对于数据进行随机裁剪的比例范围
AFFINE_ROTATION = (-30, 30)        # 进行仿射变换时的旋转角度范围 (度)
AFFINE_TRANSLATION = (-0.2, 0.2)     # 进行仿射变换时的平移范围 (相对于图像尺寸的比例)
AFFINE_SCALE = (0.8, 1.2)       # 进行仿射变换时的缩放范围
AFFINE_SHEAR = (-10, 10)       # 进行仿射变换时的剪切角度范围 (度)
AFFINE_FLIP = 0.5        # 进行仿射变换时的水平翻转的概率
AFFINE_BORDER = (128, 128, 128)      # 进行仿射变换时的边界填充值
MEAN = [0.485, 0.456, 0.406]        # 图像数据进行归一化时的均值
STD = [0.229, 0.224, 0.225]         # 图像数据进行归一化时的方差
CLASS_NUMBER = 20      # Number of classes in the dataset
TARGETS_SIZE = 100



# 模型训练时的参数配置
MAX_EPOCHS = 100
LEARNING_RATE = 0.0001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005



