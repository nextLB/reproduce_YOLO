"""
    YOLO模型的架构程序文件
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLOv1的模型架构文件
# 2025.10.17 (V1.0)            --- by next, 基本没变


import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights



S = 7
B = 2
C = 20



#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = B * 5 + C


        # # Load backbone ResNet
        # self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.backbone.requires_grad_(False)      # Freeze backbone weights
        #
        # # Delete last two layers and attach detection layers
        # self.backbone.avgpool = nn.Identity()
        # self.backbone.fc = nn.Identity()
        #
        # self.backboneReshape = Reshape(2048, 14, 14)
        # self.detectionNet = DetectionNet(2048)




        # # Load backbone ResNet34
        # self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        # self.backbone.requires_grad_(False)      # Freeze backbone weights
        #
        # # Delete last two layers and attach detection layers
        # self.backbone.avgpool = nn.Identity()
        # self.backbone.fc = nn.Identity()
        #
        # self.backboneReshape = Reshape(512, 14, 14)
        # self.detectionNet = DetectionNet(512)



        # Load backbone ResNet18 - 更小的模型
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.requires_grad_(False)      # Freeze backbone weights

        # Delete last two layers and attach detection layers
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.backboneReshape = Reshape(512, 14, 14) # ResNet18的输出通道数也是512
        self.detectionNet = DetectionNet(512)       # 输入通道数保持512


        # 特征恒等映射层 用于捕捉训练过程中的特征图像
        self.backbone_before_identity = nn.Identity()
        self.reshape_after_identity = nn.Identity()
        self.detectionNet_after_identity = nn.Identity()

    def forward(self, x):
        # backbone
        x = self.backbone_before_identity(x)
        x = self.backbone(x)

        # reshape
        x = self.backboneReshape(x)
        x = self.reshape_after_identity(x)

        # detection net
        out = self.detectionNet(x)
        out = self.detectionNet_after_identity(out)
        return out


#############################
#       Helper Modules      #
#############################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, inChannels):
        super().__init__()

        self.innerChannels = 512
        self.depth = 5 * B + C
        self.model = nn.Sequential(
            nn.Conv2d(inChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, stride=2, padding=1),
            # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * self.innerChannels, 4096),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, S * S * self.depth)

        )

    def forward(self, x):
        x = self.model.forward(x)
        out = torch.reshape(x, (-1, S, S, self.depth))
        return out




