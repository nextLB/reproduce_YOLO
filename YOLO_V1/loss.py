
"""
    YOLOv1 损失函数的实现
    YOLOv1使用平方和误差(Sum Squared Error)作为损失函数
"""

# 2025.10.17 (V1.0)            --- by next, 初步实现了YOLOv1的损失配置文件



import torch
from torch import nn as nn
from torch.nn import functional as F
from utils import get_iou, bbox_attr
import config_parameter


# 详细的YOLO损失函数计算过程的例子可见我的READ.md文件中的内容
class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 损失函数中的权重参数
        self.lCoord = 5     # 坐标损失的权重，增强边界框定位的重要性
        self.lNoobj = 0.5   # 无目标置信度损失的权重，降低背景预测的重要性


    def forward(self, p, a):
        """
        前向传播计算损失
        Args:
            p: 预测张量 (prediction)，形状为 (batch, S, S, B*5 + C)
            a: 真实标签张量 (anchor/ground truth)，形状与p相同
        Returns:
            total_loss: 总损失值

            损失值的组成为: 坐标损失+边界框尺寸损失+置信度损失+类别损失
        """

        # Calculate IOU of each predicted bbox against the ground truth
        # 计算每个预测边界框与真实边界框的IOU
        iou = get_iou(p, a)     # 返回现形状: (batch, S, S, B, B)
        # 获取每个网格单元中两个预测框与真实框的最大IOU
        maxIou = torch.max(iou, dim=-1)[0]      # 形状：(batch, S, S, B)

        # Get masks
        # 获取各种掩码
        bboxMask = bbox_attr(a, 4) > 0.0    # 真实边界框置信度掩码，标识哪些位置有目标
        pTemplate = bbox_attr(p, 4) > 0.0   # 预测边界框置信度掩码
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
            obj_ij * bbox_attr(p, 0),   # 只计算有目标位置的x预测
            obj_ij * bbox_attr(a, 0)    # 只计算有目标位置的x真实值
        )
        yLosses = mse_loss(
            obj_ij * bbox_attr(p, 1),   # 只计算有目标位置的y预测
            obj_ij * bbox_attr(a, 1)    # 只计算有目标位置的y真实值
        )
        posLosses = xLosses + yLosses   # 位置损失总和

        # TODO: 需要继续实现一下部分：
        # 1、边界框尺寸损失(w, h)
        # 2、置信度损失（有目标和无目标）
        # 3、类别概率损失


        # Bbox dimension losses
        # 边界框尺寸损失
        pWidth = bbox_attr(p, 2)    # 提取预测的宽度
        aWidth = bbox_attr(a, 2)    # 提取真实的宽度
        widthLosses = mse_loss(
            obj_ij * torch.sign(pWidth) * torch.sqrt(torch.abs(pWidth) + config_parameter.EPSILON),
            obj_ij * torch.sqrt(aWidth)
        )
        pHeight = bbox_attr(p, 3)   # 提取预测的高度
        aHeight = bbox_attr(a, 3)   # 提取真实的高度
        heightLosses = mse_loss(
            obj_ij * torch.sign(pHeight) * torch.sqrt(torch.abs(pHeight) + config_parameter.EPSILON),
            obj_ij * torch.sqrt(aHeight)
        )
        dimLosses = widthLosses + heightLosses  # 尺寸损失总和

        # 置信度损失 (目标置信度是IOU)
        objConfidenceLosses = mse_loss(
            obj_ij * bbox_attr(p, 4),   # 有目标位置的预测置信度
            obj_ij * torch.ones_like(maxIou)    # 有目标位置的真实置信度应为1
        )

        noobjConfidenceLosses = mse_loss(
            noobj_ij * bbox_attr(p, 4),     # 无目标位置的预测置信度
            torch.zeros_like(maxIou)            # 无目标位置的真实置信度应为0
        )

        # 分类损失
        classLosses = mse_loss(
            obj_i * p[..., :config_parameter.C],    # 有目标网格的预测类别概率
            obj_i * a[..., :config_parameter.C]     # 有目标网格的真实类别概率
        )

        # 计算总损失，按YOLOv1论文的权重组合各项损失
        # 坐标和尺寸损失，权重为5
        # 有目标置信度损失，权重为1
        # 无目标置信度损失，权重为0.5
        # 分类损失，权重为1
        total = self.lCoord * (posLosses + dimLosses) + objConfidenceLosses + self.lNoobj * noobjConfidenceLosses + classLosses

        return total / config_parameter.BATCH_SIZE  # 返回批次平均损失


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
