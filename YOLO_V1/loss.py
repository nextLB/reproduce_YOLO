
"""
    YOLOv1 损失函数的实现
"""


import torch
from torch import nn as nn
from torch.nn import functional as F
from utils import get_iou, bbox_attr



class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lCoord = 5
        self.lNoobj = 0.5


    def forward(self, p, a):
        # Calculate IOU of each predicted bbox against the ground truth
        iou = get_iou(p, a)     # (batch, S, S, B, B)
        maxIou = torch.max(iou, dim=-1)[0]      # (batch, S, S, B)

        # Get masks
        bboxMask = bbox_attr(a, 4) > 0.0
        pTemplate = bbox_attr(p, 4) > 0.0
        obj_i = bboxMask[..., 0:1]


