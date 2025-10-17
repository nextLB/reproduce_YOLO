

# 数据集中的图像参数配置
IMAGE_SIZE = (640, 640)
S = 10       # Divide each image into a SxS grid
B = 5       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset


# 训练集与验证集的比例配置
RATIO = 0.3


# 训练时的参数配置
MAX_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 0.00001
EPSILON = 0.0000001

