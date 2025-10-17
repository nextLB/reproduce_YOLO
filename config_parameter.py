"""
    复现项目所用到的相关的参数配置
"""


# 构建数据集时所需要用到的参数配置
RATIO = 0.3     # 训练集与验证集的比例设置
BATCH_SIZE = 4
NUM_WORKERS = 4
RANDOM_CROP_RATIO = 0.5  # 构建数据集时对于数据进行随机裁剪的比例
RANDOM_AFFINE_RATIO = 0.5  # 构建数据集时对于数据进行仿射变换的比例
RANDOM_COLOR_RATIO = 0.5  # 对数据集进行随机颜色调整的比例
RANDOM_CROP_SCOPE = (0.3, 1)       # 构建数据集时对于数据进行随机裁剪的比例范围






