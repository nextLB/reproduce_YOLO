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
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)      # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)
        )

    def forward(self, x):
        out = self.model.forward(x)
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

        self.innerChannels = 1024
        self.depth = 5 * config_parameter.B + config_parameter.C
        self.model = nn.Sequential(
            nn.Conv2d(inChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, stride=2, padding=1),      # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(self.innerChannels, self.innerChannels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * self.innerChannels, 4096),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, config_parameter.S * config_parameter.S * self.depth)

        )

    def forward(self, x):
        x = self.model.forward(x)
        out = torch.reshape(x, (-1, config_parameter.S, config_parameter.S, self.depth))
        return out

    

