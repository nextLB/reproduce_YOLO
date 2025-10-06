"""
    YOLO模型的架构程序文件
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config_parameter


#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config_parameter.B * 5 + config_parameter.C

        # Load backbone ResNet





