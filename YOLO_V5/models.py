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
        self.batchNorm = nn.BatchNorm2d(outChannels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batchNorm(self.conv(x)))


# 瓶颈层模块
class BottleneckBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, useShortcut: bool , groups: int, expansion: float):
        super(BottleneckBlock, self).__init__()
        hiddenChannels = int(outChannels * expansion)
        self.conv1 = ConvBlock(inChannels, hiddenChannels, 1, 1, 1)
        self.conv2 = ConvBlock(hiddenChannels, outChannels, 3, 1, groups)
        self.useAdd = useShortcut and inChannels == outChannels

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        if self.useAdd:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


# C3模块
class C3Block(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, numBottlenecks: int, useShortcut: bool, groups: int, expansion: float):
        super(C3Block, self).__init__()
        hiddenChannels = int(outChannels * expansion)
        self.conv1 = ConvBlock(inChannels, hiddenChannels, 1, 1, 1)
        self.conv2 = ConvBlock(inChannels, hiddenChannels, 1, 1, 1)
        self.conv3 = ConvBlock(2 * hiddenChannels, outChannels, 1, 1, 1)

        self.bottleneckBlocks = nn.Sequential(
            *[BottleneckBlock(hiddenChannels, hiddenChannels, useShortcut, groups, expansion=1.0)
              for _ in range(numBottlenecks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.bottleneckBlocks(self.conv2(x))
        return self.conv3(torch.cat((x1, x2), dim=1))


# SPPF模块
class SPPFBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int):
        super(SPPFBlock, self).__init__()
        hiddenChannels = inChannels // 2
        self.conv1 = ConvBlock(inChannels, hiddenChannels, 1, 1, 1)
        self.conv2 = ConvBlock(hiddenChannels * 4, outChannels,1, 1, 1)

        self.maxPool = nn.MaxPool2d(kernelSize=kernelSize, stride=1, padding=kernelSize//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.maxPool(x)
        y2 = self.maxPool(y1)
        y3 = self.maxPool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


        