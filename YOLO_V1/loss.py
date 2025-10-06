
"""
    YOLOv1 损失函数的实现
"""


import torch
from torch import nn as nn
from torch.nn import functional as F



class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lCoord = 5
        self.lNoobj = 0.5



