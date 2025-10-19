"""
    YOLOV4模型的构建
"""

# 2025.10.18 (V1.1)            --- by next, 初步构建了YOLOV4的模型架构



# YOLOV4的主要架构组件为: CSPDarnet53骨干网络、SPP模块、PANet颈部网络和YOLO检测头



import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



# Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 卷积块： conv + BN + Mish
class ConvBnMish(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride=1, padding=0):
        super(ConvBnMish, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(outChannels)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnMish(channels, channels, 1)
        self.conv2 = ConvBnMish(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


# CSP块
class CspBlock(nn.Module):
    def __init__(self, inChannels, outChannels, numBlocks):
        super(CspBlock, self).__init__()
        self.downSample = ConvBnMish(inChannels, outChannels, )

