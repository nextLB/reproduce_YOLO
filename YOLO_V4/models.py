"""
    YOLOV4模型的构建
"""

# 2025.10.18 (V1.1)            --- by next, 初步构建了YOLOV4的模型架构
# 2025.10.18 (V1.1)            --- 修复PANet中的卷积通道不匹配问题



# YOLOV4的主要架构组件为: CSPDarnet53骨干网络、SPP模块、PANet颈部网络和YOLO检测头



import torch
import torch.nn as nn
import torch.nn.functional as F




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
        self.downSample = ConvBnMish(inChannels, outChannels, 3, stride=2, padding=1)

        midChannels = outChannels // 2
        self.splitConv0 = ConvBnMish(outChannels, midChannels, 1)
        self.splitConv1 = ConvBnMish(outChannels, midChannels, 1)

        self.blocks = nn.Sequential(*[ResidualBlock(midChannels) for _ in range(numBlocks)])

        self.concatConv = ConvBnMish(midChannels * 2, outChannels, 1)

    def forward(self, x):
        x = self.downSample(x)

        x0 = self.splitConv0(x)
        x1 = self.splitConv1(x)
        x1 = self.blocks(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concatConv(x)
        return x


# CSPDarknet53骨干网络
class CspDarknet53(nn.Module):
    def __init__(self):
        super(CspDarknet53, self).__init__()
        self.conv1 = ConvBnMish(3, 32, 3, padding=1)
        self.layer1 = CspBlock(32, 64, 1)   # P1
        self.layer2 = CspBlock(64, 128, 2)  # P2
        self.layer3 = CspBlock(128, 256, 8)  # P3
        self.layer4 = CspBlock(256, 512, 8)     # P4
        self.layer5 = CspBlock(512, 1024, 4)    # P5

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)  # P1
        x = self.layer2(x)  # P2
        p3 = self.layer3(x) # P3
        p4 = self.layer4(p3) # P4
        p5 = self.layer5(p4) # P5
        return p3, p4, p5



# SPP模块
class SpatialPyramidPooling(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SpatialPyramidPooling, self).__init__()
        midChannels = inChannels // 2
        self.conv1 = ConvBnMish(inChannels, midChannels, 1)
        self.conv2 = ConvBnMish(midChannels * 4, outChannels, 1)

        self.pool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(13, stride=1, padding=6)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        x = torch.cat([x, p1, p2, p3], dim=1)
        x = self.conv2(x)
        return x


# 上采样模块
class UpsampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UpsampleBlock, self).__init__()
        # 先调整通道数，再上采样
        self.conv = ConvBnMish(inChannels, outChannels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


# 下采样模块
class DownsampleBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DownsampleBlock, self).__init__()
        self.conv = ConvBnMish(inChannels, outChannels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# PANet路径聚合网络
class PathAggregationNetwork(nn.Module):
    def __init__(self):
        super(PathAggregationNetwork, self).__init__()

        # 上采样路径（自顶向下）
        # P5 -> P4
        self.upsampleP5 = UpsampleBlock(512, 256)  # 输入512，输出256

        # P4 -> P3
        self.upsampleP4 = UpsampleBlock(512, 256)  # 输入512，输出256

        # 下采样路径（自底向上）
        # P3 -> P4
        self.downsampleP3 = DownsampleBlock(256, 256)  # 输入256，输出256

        # P4 -> P5
        self.downsampleP4 = DownsampleBlock(512, 512)  # 输入512，输出512

        # 特征融合卷积 - 修正输入通道数
        # 上采样路径的融合卷积
        self.convP4_1 = ConvBnMish(768, 512, 3, padding=1)  # 融合P4(512)和上采样的P5(256) -> 768输入
        self.convP3_1 = ConvBnMish(512, 256, 3, padding=1)  # 融合P3(256)和上采样的P4(256) -> 512输入

        # 下采样路径的融合卷积
        self.convP4_2 = ConvBnMish(768, 512, 3, padding=1)  # 融合P4_1(512)和下采样的P3(256) -> 768输入
        self.convP5_2 = ConvBnMish(1024, 512, 3, padding=1)  # 融合P5(512)和下采样的P4(512) -> 1024输入

    def forward(self, p3, p4, p5):
        # 上采样路径（自顶向下）
        # P5 -> P4
        p5_up = self.upsampleP5(p5)  # 512->256
        p4_cat1 = torch.cat([p4, p5_up], dim=1)  # 512 + 256 = 768
        p4_1 = self.convP4_1(p4_cat1)  # 768->512

        # P4 -> P3
        p4_up = self.upsampleP4(p4_1)  # 512->256
        p3_cat1 = torch.cat([p3, p4_up], dim=1)  # 256 + 256 = 512
        p3_out = self.convP3_1(p3_cat1)  # 512->256

        # 下采样路径（自底向上）
        # P3 -> P4
        p3_down = self.downsampleP3(p3_out)  # 256->256
        p4_cat2 = torch.cat([p4_1, p3_down], dim=1)  # 512 + 256 = 768
        p4_out = self.convP4_2(p4_cat2)  # 768->512

        # P4 -> P5
        p4_down = self.downsampleP4(p4_out)  # 512->512
        p5_cat2 = torch.cat([p5, p4_down], dim=1)  # 512 + 512 = 1024
        p5_out = self.convP5_2(p5_cat2)  # 1024->512

        return p3_out, p4_out, p5_out


# YOLO检测头
class YoloHead(nn.Module):
    def __init__(self, inChannels, numAnchors, numClasses):
        super(YoloHead, self).__init__()
        self.numAnchors = numAnchors
        self.numClasses = numClasses

        self.conv1 = ConvBnMish(inChannels, inChannels * 2, 3, padding=1)
        self.conv2 = nn.Conv2d(inChannels * 2, numAnchors * (5 + numClasses), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # 调整输出形状: [batch, anchors*(5+numClasses), height, width] ->
        # [batch, anchors, height, width, 5+numClasses]
        batch, _, height, width = x.size()
        x = x.view(batch, self.numAnchors, 5 + self.numClasses, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        return x



# 完整的YOLOV4模型
class YOLOv4(nn.Module):
    def __init__(self, numClasses, numAnchors):
        super(YOLOv4, self).__init__()
        self.numClasses = numClasses
        self.numAnchors = numAnchors

        # 骨干网络
        self.backbone = CspDarknet53()

        # SPP模块
        self.spp = SpatialPyramidPooling(1024, 512)

        # 颈部网络
        self.neck = PathAggregationNetwork()

        # 检测头
        self.headP3 = YoloHead(256, numAnchors, numClasses)  # P3输出256通道
        self.headP4 = YoloHead(512, numAnchors, numClasses)  # P4输出512通道
        self.headP5 = YoloHead(512, numAnchors, numClasses)  # P5输出512通道

        # 初始化权重
        self._initializeWeights()

    def forward(self, x):
        # 骨干网络
        p3, p4, p5 = self.backbone(x)

        # SPP模块
        p5 = self.spp(p5)

        # 颈部网络
        p3, p4, p5 = self.neck(p3, p4, p5)

        # 检测头
        outP3 = self.headP3(p3)  # 小目标检测
        outP4 = self.headP4(p4)  # 中目标检测
        outP5 = self.headP5(p5)  # 大目标检测

        return outP3, outP4, outP5

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = YOLOv4(numClasses=20, numAnchors=3).to(device)

    # 测试前向传播
    with torch.no_grad():
        x = torch.randn(2, 3, 416, 416).to(device)
        outP3, outP4, outP5 = model(x)
        print(f"P3 output shape: {outP3.shape}")  # [2, 3, 52, 52, 25]
        print(f"P4 output shape: {outP4.shape}")  # [2, 3, 26, 26, 25]
        print(f"P5 output shape: {outP5.shape}")  # [2, 3, 13, 13, 25]

    return model

if __name__ == '__main__':
    main()