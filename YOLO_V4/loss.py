"""
    YOLOV4损失函数的实现
"""



# 2025.10.21 (V1.1)            --- by next, 实现了YOLOV4模型与真实标签值之间的计算

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOV4Loss(nn.Module):


