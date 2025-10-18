"""
    YOLOv1 损失函数的实现
    YOLOv1使用平方和误差(Sum Squared Error)作为损失函数
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLOv1的损失的计算
# 2025.10.17 (V1.1)            --- by next, 进一步增强了YOLOv1  loss损失的可读性与鲁棒性等

import torch
import torch.nn as nn
from torch.nn import functional as F


S = 7
B = 2
C = 20
EPSILON = 0.0000001
IMAGE_SIZE = (448, 448)





class YOLOv1Loss(nn.Module):
    def __init__(self):
        super(YOLOv1Loss, self).__init__()
        # 损失函数中的权重参数
        self.lCoord = 5     # 坐标损失的权重，增强边界框定位的重要性
        self.lNoobj = 0.5   # 无目标置信度损失的权重，降低背景预测的重要性



    def forward(self, predictions, targets):
        """
        前向传播计算损失
        Args:
            p: 预测张量 (prediction)，形状为 (batch, S, S, B*5 + C)
            a: 真实标签张量 (anchor/ground truth)，形状与p相同
        Returns:
            total_loss: 总损失值

            损失值的组成为: 坐标损失+边界框尺寸损失+置信度损失+类别损失
        """
        BATCH_SIZE = targets.shape[0]


        # 首先生成适合yolov1计算的目标值
        groundTruth = self._build_targets(targets)
        groundTruth = groundTruth.to(device=predictions.device)


        # Calculate IOU of each predicted bbox against the ground truth
        # 计算每个预测边界框与真实边界框的IOU
        iou = get_iou(predictions, groundTruth)     # 返回现形状: (batch, S, S, B, B)
        # 获取每个网格单元中两个预测框与真实框的最大IOU
        maxIou = torch.max(iou, dim=-1)[0]      # 形状：(batch, S, S, B)

        # Get masks
        # 获取各种掩码
        bboxMask = bbox_attr(groundTruth, 4) > 0.0    # 真实边界框置信度掩码，标识哪些位置有目标
        pTemplate = bbox_attr(predictions, 4) > 0.0   # 预测边界框置信度掩码
        obj_i = bboxMask[..., 0:1]      # 1 if grid I has any object at all     如果网格i包含如何目标，值为1


        # 确定“负责”预测目标的边界框
        responsible = torch.zeros_like(pTemplate).scatter_(     # 形状: (batch, S, S, B)
            -1,     # 在最后一个维度进行散射操作
            torch.argmax(maxIou, dim=-1, keepdim=True),         # (batch, S, S, B)  找到每个网格中IOU最大的边界框索引
            value=1         # 1 if bounding box is "responsible" for predicting the object  如果边界框“负责”预测目标，值为1
        )

        # 组合条件：有目标且边界负责预测
        # 形状: (batch, S, S, B)
        obj_ij = obj_i * responsible        # 1 if bounding exists AND bbox is responsible
        # 无目标区域：既没有目标或者边界框不负责预测
        # 形状：(batch, S, S, B)
        noobj_ij = ~obj_ij      # Otherwise, confidence should be 0

        # XY position losses
        # 计算坐标损失 (x, y)
        xLosses = mse_loss(
            obj_ij * bbox_attr(predictions, 0),   # 只计算有目标位置的x预测
            obj_ij * bbox_attr(groundTruth, 0)    # 只计算有目标位置的x真实值
        )
        yLosses = mse_loss(
            obj_ij * bbox_attr(predictions, 1),   # 只计算有目标位置的y预测
            obj_ij * bbox_attr(groundTruth, 1)    # 只计算有目标位置的y真实值
        )
        posLosses = xLosses + yLosses   # 位置损失总和

        # TODO: 需要继续实现一下部分：
        # 1、边界框尺寸损失(w, h)
        # 2、置信度损失（有目标和无目标）
        # 3、类别概率损失


        # Bbox dimension losses
        # 边界框尺寸损失
        pWidth = bbox_attr(predictions, 2)    # 提取预测的宽度
        aWidth = bbox_attr(groundTruth, 2)    # 提取真实的宽度
        widthLosses = mse_loss(
            obj_ij * torch.sign(pWidth) * torch.sqrt(torch.abs(pWidth) + EPSILON),
            obj_ij * torch.sqrt(aWidth)
        )
        pHeight = bbox_attr(predictions, 3)   # 提取预测的高度
        aHeight = bbox_attr(groundTruth, 3)   # 提取真实的高度
        heightLosses = mse_loss(
            obj_ij * torch.sign(pHeight) * torch.sqrt(torch.abs(pHeight) + EPSILON),
            obj_ij * torch.sqrt(aHeight)
        )
        dimLosses = widthLosses + heightLosses  # 尺寸损失总和

        # 置信度损失 (目标置信度是IOU)
        objConfidenceLosses = mse_loss(
            obj_ij * bbox_attr(predictions, 4),   # 有目标位置的预测置信度
            obj_ij * torch.ones_like(maxIou)    # 有目标位置的真实置信度应为1
        )

        noobjConfidenceLosses = mse_loss(
            noobj_ij * bbox_attr(predictions, 4),     # 无目标位置的预测置信度
            torch.zeros_like(maxIou)            # 无目标位置的真实置信度应为0
        )

        # 分类损失
        classLosses = mse_loss(
            obj_i * predictions[..., :C],    # 有目标网格的预测类别概率
            obj_i * groundTruth[..., :C]     # 有目标网格的真实类别概率
        )

        # 计算总损失，按YOLOv1论文的权重组合各项损失
        # 坐标和尺寸损失，权重为5
        # 有目标置信度损失，权重为1
        # 无目标置信度损失，权重为0.5
        # 分类损失，权重为1
        total = self.lCoord * (posLosses + dimLosses) + objConfidenceLosses + self.lNoobj * noobjConfidenceLosses + classLosses
        total /= BATCH_SIZE

        lossComponents = {
            'coordLoss': 0,
            'objLoss': 0,
            'noObjLoss': 0,
            'classLoss': 0
        }


        lossComponents['coordLoss'] += dimLosses.item()
        lossComponents['objLoss'] += objConfidenceLosses.item()
        lossComponents['noObjLoss'] += noobjConfidenceLosses.item()
        lossComponents['classLoss'] += classLosses.item()

        return total, lossComponents


    def _build_targets(self, targets):


        # 逐个的边界框目标进行处理
        boundingBoxes = {}      # 跟踪每个网格单元格已分配的边界框数量
        classNames = {}     # 跟踪每个网格单元格分配的类别
        depth = 5 * B + C
        groundTruth = torch.zeros((targets.shape[0], S, S, depth))

        # 计算网格尺寸
        gridSizeX = IMAGE_SIZE[0] / S  # 每个网格的宽度
        gridSizeY = IMAGE_SIZE[1] / S  # 每个网格的高度

        for batchIndex in range(targets.shape[0]):
            for i in range(targets[batchIndex].shape[0]):
                if targets[batchIndex][i][4] != 1:
                    break

                midX = targets[batchIndex][i][0]
                midY = targets[batchIndex][i][1]
                width = targets[batchIndex][i][2]
                height = targets[batchIndex][i][3]
                classID = targets[batchIndex][i][5:]


                is_one = classID == 1.0
                indices = torch.nonzero(is_one)
                index = indices.squeeze().tolist()



                # 确定中心点所在的网格单元格
                col = int(midX // gridSizeX)
                row = int(midY // gridSizeY)

                # 确保网格缩影在有效范围内
                if 0 <= col < S and 0 <= row < S:
                    cell = (row, col)

                    # 如果该网格单元格未被分配类别，或者当前类别与已分配类别相同
                    if cell not in classNames or index == classNames[cell]:
                        oneHot = classID

                        # 将类别信息写入ground truth张量的前C个通道
                        groundTruth[batchIndex, row, col, :C] = oneHot
                        classNames[cell] = index

                        # 获取当前网格单元格已分配的边界框数量
                        bboxIndex = boundingBoxes.get(cell, 0)

                        # 如果还有可用的边界框槽位
                        if bboxIndex < B:
                            # 计算边界框相对于网格单元格的归一化坐标
                            bboxTruth = (
                                (midX - col * gridSizeX) / gridSizeX,  # X坐标相对于网格的偏移
                                (midY - row * gridSizeY) / gridSizeY,  # Y坐标相对于网格的偏移
                                (width) / IMAGE_SIZE[0],  # 宽度相对于图像的比率
                                (height) / IMAGE_SIZE[1],  # 高度相对于图像的比率
                                1.0  # 置信度（有目标）
                            )

                            # 计算当前边界框在张量中的起始位置
                            bbox_start = C + 5 * bboxIndex

                            # 将当前边界框信息写入ground truth张量
                            groundTruth[batchIndex, row, col, bbox_start:bbox_start + 5] = torch.tensor(bboxTruth)

                            # 更新该网格单元格的边界框计数
                            boundingBoxes[cell] = bboxIndex + 1

        return groundTruth


def mse_loss(a, b):
    """
    计算均方误差损失，支持多维度输入

    Args:
        a: 输入张量a
        b: 输入张量b

    Returns:
        计算得到的MSE损失值
    """
    # 展平除最后两个维度外的所有维度
    flattenedA = torch.flatten(a, end_dim=-2)
    # 扩展b张量以匹配a的形状
    flattenedB = torch.flatten(b, end_dim=-2).expand_as(flattenedA)

    # 使用PyTorch的MSE损失函数，采用求和 reduction
    return F.mse_loss(
        flattenedA,
        flattenedB,
        reduction='sum'  # 使用求和而不是平均，这是YOLO论文中的做法
    )



def bbox_to_coords(t):
    """
    将边界框格式从 [x, y, width, height] 转换为角点坐标 ([x1, y1], [x2, y2])

    Args:
        t: 输入边界框张量，格式为 [x, y, width, height]

    Returns:
        infoOne: 左上角坐标 [x1, y1]
        infoTwo: 右下角坐标 [x2, y2]
    """
    # 提取宽度和x中心坐标
    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    # 计算左右边界
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    # 提取高度和y中心坐标
    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    # 计算上下边界
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    # 堆叠坐标形成角点表示
    infoOne = torch.stack((x1, y1), dim=4)  # 左上角坐标
    infoTwo = torch.stack((x2, y2), dim=4)  # 右下角坐标

    return infoOne, infoTwo


def bbox_attr(data, i):
    """
    从数据张量中提取所有边界框的第i个属性

    Args:
        data: 输入数据张量，形状为 (..., B*5 + C)
        i: 要提取的属性索引 (0:x, 1:y, 2:width, 3:height, 4:confidence)

    Returns:
        指定属性的张量
    """
    # 计算属性在张量中的起始位置: C + i
    attrStart = C + i
    # 使用切片操作提取所有边界框的指定属性
    return data[..., attrStart::5]

def get_iou(p, a):
    """
    计算预测边界框和真实边界框之间的交并比(IOU)

    Args:
        p: 预测边界框张量
        a: 真实边界框张量

    Returns:
        iou: IOU值，形状为 (batch, S, S, B, B)
    """
    # 将边界框格式从 [x, y, width, height] 转换为角点坐标 [x1, y1, x2, y2]
    p_tl, p_br = bbox_to_coords(p)  # 预测框的左上角和右下角坐标
    a_tl, a_br = bbox_to_coords(a)  # 真实框的左上角和右下角坐标

    # 计算交集的左上角和右下角坐标
    # 交集左上角 = 两个框左上角坐标的最大值
    # 交集右下角 = 两个框右下角坐标的最小值
    coordsJoinSize = (-1, -1, -1, B, B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coordsJoinSize),  # 扩展维度以进行广播计算
        a_tl.unsqueeze(3).expand(coordsJoinSize)  # 扩展维度以进行广播计算
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coordsJoinSize),
        a_br.unsqueeze(3).expand(coordsJoinSize)
    )

    # 计算交集区域的宽和高，并确保非负
    intersectionSides = torch.clamp(br - tl, min=0.0)
    # 计算交集面积
    intersection = intersectionSides[..., 0] * intersectionSides[..., 1]

    # 计算预测框的面积
    pArea = bbox_attr(p, 2) * bbox_attr(p, 3)  # width * height
    pArea = pArea.unsqueeze(4).expand_as(intersection)  # 扩展维度以匹配IOU计算

    # 计算真实框的面积
    aArea = bbox_attr(a, 2) * bbox_attr(a, 3)  # width * height
    aArea = aArea.unsqueeze(4).expand_as(intersection)  # 扩展维度以匹配IOU计算

    # 计算并集面积
    union = pArea + aArea - intersection

    # 处理除零情况：当并集面积为0时，避免除以0
    zeroUnions = (union == 0.0)
    union[zeroUnions] = EPSILON  # 使用一个很小的值替代0
    intersection[zeroUnions] = 0.0  # 对应的交集设为0

    # 返回IOU = 交集面积 / 并集面积
    return intersection / union