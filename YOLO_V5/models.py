"""
    YOLOV5模型的构建
"""

# 2025.10.25 (V1.1)            --- by next, 初步构建了YOLOV5的模型架构


#############################################
#############################################

# 关于YOLOV5的主要架构组成如下

# 首先进行图像预处理：Mosaic数据增强    --->    自适应锚框计算     --->    自适应图像缩放
# 之后是Backbone(CSPDarknet53):Conv卷积块     --->    C3模块    --->    SPPF模块
# 然后是Neck(FPN+PAN):


#############################################
#############################################



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math






