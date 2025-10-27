"""
    YOLOV5模型的构建
"""

# 2025.10.25 (V1.1)            --- by next, 初步构建了YOLOV5的模型架构


#############################################
#############################################

# 关于YOLOV5的主要架构组成如下

# 首先进行图像预处理：Mosaic数据增强    --->    自适应锚框计算     --->    自适应图像缩放
# 之后是Backbone(CSPDarknet53):Conv卷积块     --->    C3模块    --->    SPPF模块
# 然后是Neck(FPN+PAN):FPN路径自顶向下    --->    PAN模块自底向上   ---> 多尺度特征融合
# 最后是Head检测头:大目标检测P5/32  --->    中目标检测(P4/16)    --->    小目标检测P3/8


#############################################
#############################################



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


# 基础卷积模块
class ConvBlock(nn.Module):
    def __init__(self, inCannels: int, outChannels: int, kernelSize: int, stride: int, groups: int):
        super(ConvBlock, self).__init__()
        padding = (kernelSize - 1) // 2
        self.conv = nn.Conv2d(inCannels, outChannels, kernelSize, stride, padding, groups=groups, bias=False)




