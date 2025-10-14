"""
    YOLOV3模型的构造程序文件
"""

import torch
import torch.nn as nn
import config_paramters

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# YOLOv3模型组件
class DarknetConv(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding):
        super(DarknetConv, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(outChannels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))



# Resnet模块组件
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarknetConv(channels, channels//2, 1, 1, 0)
        self.conv2 = DarknetConv(channels//2, channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        out += residual
        return out





class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()


        # 初始化卷积层
        self.conv1 = DarknetConv(3, 32, 3, 1, 1)
        self.conv2 = DarknetConv(32, 64, 3, 2, 1)

        # 残差块组1
        self.residual1 = self._make_layer(64, 32, 1)
        self.conv3 = DarknetConv(64, 128, 3, 2, 1)

        # 残差块组2
        self.residual2 = self._make_layer(128, 64, 2)
        self.conv4 = DarknetConv(128, 256, 3, 2, 1)

        # 残差块组3
        self.residual3 = self._make_layer(256, 128, 8)
        self.conv5 = DarknetConv(256, 512, 3, 2, 1)

        # 残差块组4
        self.residual4 = self._make_layer(512, 256, 8)
        self.conv6 = DarknetConv(512, 1024, 3, 2, 1)

        # 残差块组5
        self.residual5 = self._make_layer(1024, 512, 4)

    def _make_layer(self, inChannels, outChannels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(inChannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.conv2(x)

        # 残差块组1
        x = self.residual1(x)
        x = self.conv3(x)

        # 残差块组2
        x = self.residual2(x)
        x = self.conv4(x)

        # 残差块组3 - 输出1
        x = self.residual3(x)
        out3 = x
        x = self.conv5(x)

        # 残差块组4 - 输出2
        x = self.residual4(x)
        out2 = x
        x = self.conv6(x)

        # 残差块组5 - 输出3
        x = self.residual5(x)
        out1 = x

        return out1, out2, out3


# 检测头的实现
class YOLOLayer(nn.Module):
    def __init__(self, anchors, numClasses, imagesDim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.numAnchors = len(anchors)
        self.numClasses = numClasses
        self.imageDim = imagesDim
        self.gridSize = 0

        # 预测层
        self.conv = DarknetConv(512, self.numAnchors * (5 + numClasses), 1, 1, 0)

    def forward(self, x):
        batchSize = x.size(0)
        gridSize = x.size(2)

        # 预测输出
        prediction = self.conv(x)
        # 重塑张量形状，将通道维度分解为多个维度
        prediction = prediction.view(batchSize, self.numAnchors, 5 + self.numClasses, gridSize, gridSize)
        # 调整维度顺序并确保内存连续
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

        # 应用sigmoid到特定通道
        prediction[..., 0:2] = torch.sigmoid(prediction[..., 0:2])      # 中心坐标
        prediction[..., 4:5] = torch.sigmoid(prediction[..., 4:5])      # 对象置信度
        prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])        # 类别概率

        return prediction





# YOLOv3模型的构建
class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()

        self.backbone = Darknet53()

        # YOLO检测头的定义
        self.yoloLayers = nn.ModuleList([
            YOLOLayer(config_paramters.ANCHORS[0], config_paramters.NUM_CLASSES, config_paramters.IMAGE_SIZE),  # 大特征图
            YOLOLayer(config_paramters.ANCHORS[1], config_paramters.NUM_CLASSES, config_paramters.IMAGE_SIZE),  # 中特征图
            YOLOLayer(config_paramters.ANCHORS[2], config_paramters.NUM_CLASSES, config_paramters.IMAGE_SIZE),  # 小特征图
        ])

        # 上采样和特征融合
        self.upsample1 = nn.Sequential(
            DarknetConv(1024, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv1 = DarknetConv(512, 256, 1, 1, 0)

        self.upsample2 = nn.Sequential(
            DarknetConv(512, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv2 = DarknetConv(256, 128, 1, 1, 0)


    def forward(self, x):
        # 骨干网络
        out1, out2, out3 = self.backbone(x)  # out1: 1024, out2: 512, out3: 256

        # 大特征图检测
        yoloOut1 = self.yoloLayers[0](out1)

        # 上采样并融合
        x = self.upsample1(out1)
        x = torch.cat([x, out2], 1)
        x = self.conv1(x)
        yoloOut2 = self.yoloLayers[1](x)

        # 再次上采样并融合
        x = self.upsample2(x)
        x = torch.cat([x, out3], 1)
        x = self.conv2(x)
        yoloOut3 = self.yoloLayers[2](x)

        return [yoloOut1, yoloOut2, yoloOut3]





def main():

    model = YOLOv3().to(device)

    print(model)


if __name__ == '__main__':
    main()
