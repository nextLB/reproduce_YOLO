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




