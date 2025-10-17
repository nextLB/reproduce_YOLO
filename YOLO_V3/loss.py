"""
    YOLOV3的损失函数具体实现的程序
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLOV3的损失函数
# 2025.10.17 (V1.1)            --- by next, 使用了更通用的接口形式更新了YOLOv3的损失函数



import torch
import torch.nn as nn
import torch.nn.functional as F


ANCHORS = [
            [(10, 13), (16, 30), (33, 23)],   # P3/8
            [(30, 61), (62, 45), (59, 119)],  # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]
NUM_CLASSES = 20
IMAGE_SIZE = 640




class YOLOv3Loss(nn.Module):
    def __init__(self):
        super(YOLOv3Loss, self).__init__()
        self.anchors = ANCHORS
        self.numClasses = NUM_CLASSES
        self.imageSize = IMAGE_SIZE
        # 参数一：'none' - 不进行降维
        # 含义：返回与输入相同形状的损失张量，每个位置都保留独立的损失值
        # 使用场景：当你需要对不同位置的损失进行加权或选择性处理时
        # 参数二：'mean' - 求平均值（默认）
        # 含义：返回所有元素损失的平均值
        # 使用场景：大多数标准训练场景
        # 参数三：'sum' - 求和
        # 含义：返回所有元素损失的总和
        # 使用场景：当你想要总损失而不是平均损失时
        self.mseLoss = nn.MSELoss(reduction='none')
        self.bceLoss = nn.BCELoss(reduction='none')


        # 损失权重的设定
        self.lambdaCoord = 5.0
        self.lambdaObj = 1.0
        self.lambdaNoobj = 0.5
        self.lambdaClass = 1.0




