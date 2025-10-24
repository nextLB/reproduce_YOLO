"""
    YOLOV4损失函数的实现
"""



# 2025.10.21 (V1.1)            --- by next, 实现了YOLOV4模型与真实标签值之间的计算

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



NUM_CLASSES = 20
ANCHORS = [
        [12, 16], [19, 36], [40, 28],    # P3/8 小目标
        [36, 75], [76, 55], [72, 146],   # P4/16 中目标
        [142, 110], [192, 243], [459, 401] # P5/32 大目标
    ]

IMAGE_SIZE = 448
LAMBDA_COORD = 5.0
LAMBDA_NO_OBJ = 0.5
LAMBDA_CLS = 1.0

class YOLOv4Loss(nn.Module):
    def __init__(self, numClasses, anchors, imageSize, lambdaCoord, lambdaNoobj, lambdaCls):
        """
        YOLOv4损失函数

        Args:
            num_classes: 类别数量
            anchors: 锚框列表，格式为[(w1, h1), (w2, h2), ...]
            image_size: 图像尺寸
            lambda_coord: 坐标损失权重
            lambda_noobj: 无目标置信度损失权重
        """
        super(YOLOv4Loss, self).__init__()
        self.numClasses = numClasses
        self.numAnchors = len(anchors)
        self.inputSize = imageSize
        self.lambdaCoord = lambdaCoord
        self.lambdaNoObj = lambdaNoobj
        self.lambdaCls = lambdaCls

        # 转换为tensor
        self.anchors = torch.tensor(anchors, dtype=torch.float32)

        # 用于CIoU计算
        self.eps = 1e-7

    def forward(self, predictions, targets):
        """
        前向传播计算损失

        Args:
            predictions: 模型预测结果，包含三个尺度的输出
                        [p3_pred, p4_pred, p5_pred]
            targets: 真实标签，形状为[batch_size, max_objects, 5 + num_classes]

        Returns:
            totalLoss: 总损失
            lossDict: 各损失分量字典
        """
        device = predictions[0].device
        self.anchors = self.anchors.to(device)


        # 初始化损失分量
        totalLoss = 0
        lossCoord = 0
        lossConf = 0
        lossCls = 0
        lossComponents = {
            'coordLoss': 0,
            'confLoss': 0,
            'classLoss': 0
        }

        # 处理每个尺度的预测
        for scaleIdx, pred in enumerate(predictions):
            batchSize, numAnchors, gridH, gridW, _ = pred.size()


            # 为当前尺度选择对应的锚框
            scaleAnchors = self.anchors[scaleIdx * numAnchors:(scaleIdx + 1) * numAnchors]


            # 构建网格坐标
            gridX = torch.arange(gridW, device=device).repeat(batchSize, numAnchors, gridH, 1)
            gridY = torch.arange(gridH, device=device).repeat(batchSize, numAnchors, gridW, 1).transpose(2, 3)

            # 组合网格坐标 [batch, anchors, gridH, gridW, 2]
            grid = torch.stack([gridX, gridY], dim=-1).float()

            # 重塑预测张量
            pred = pred.view(batchSize, numAnchors, gridH, gridW, 5 + self.numClasses)

            # 解码预测框
            predBoxes = self.decodePredictions(pred[..., :4], grid, scaleAnchors, gridH, gridW)
            predConf = torch.sigmoid(pred[..., 4:5])  # 置信度
            predCls = torch.sigmoid(pred[..., 5:])    # 类别概率

            # 为当前尺度构建目标张量
            targetMask, targetBoxes, targetConf, targetCls = self.buildTargets(
                targets, gridH, gridW, scaleAnchors, batchSize, device)

            # 计算各项损失
            scaleLossCoord = self.calculateCoordinateLoss(predBoxes, targetBoxes, targetMask)
            scaleLossConf = self.calculateConfidenceLoss(predConf, targetConf, targetMask)
            scaleLossCls = self.calculateClassLoss(predCls, targetCls, targetMask)

            # 累加损失
            lossCoord += scaleLossCoord
            lossConf += scaleLossConf
            lossCls += scaleLossCls


        # 计算总损失
        totalLoss = (self.lambdaCoord * lossCoord +
                    lossConf +
                    self.lambdaCls * lossCls)


        # 检查是否为NaN
        if torch.isnan(totalLoss):
            print("Warning: Total loss is NaN, returning zero loss")
            totalLoss = torch.tensor(0.0, device=device)

        if torch.isinf(lossCoord):
            lossComponents['coordLoss'] = torch.tensor(0.0, device=device)
        else:
            lossComponents['coordLoss'] = lossCoord

        if torch.isinf(lossConf):
            lossComponents['confLoss'] = torch.tensor(0.0, device=device)
        else:
            lossComponents['confLoss'] = lossConf

        if torch.isinf(lossCls):
            lossComponents['confLoss'] = torch.tensor(0.0, device=device)
        else:
            lossComponents['classLoss'] = lossCls

        return totalLoss, lossComponents


    def decodePredictions(self, predBoxes, grid, anchors, gridH, gridW):
        """
        解码预测框坐标

        Args:
            predBoxes: 预测的边界框 [batch, anchors, gridH, gridW, 4]
            grid: 网格坐标 [batch, anchors, gridH, gridW, 2]
            anchors: 锚框
            gridH: 网格高度
            gridW: 网格宽度

        Returns:
            decodedBoxes: 解码后的边界框 [x, y, w, h]
        """
        # 预测的偏移量
        predXY = torch.sigmoid(predBoxes[..., :2])  # tx, ty
        predWH = predBoxes[..., 2:4]  # tw, th

        # 修复：确保网格坐标与预测张量形状匹配
        # 转换为绝对坐标
        gridSize = torch.tensor([gridW, gridH], device=predBoxes.device).view(1, 1, 1, 1, 2)
        boxXY = (predXY + grid) / gridSize

        # 修复：正确应用锚框
        anchors = anchors.view(1, -1, 1, 1, 2)
        boxWH = (torch.exp(predWH) * anchors) / self.inputSize

        # 组合为[x, y, w, h]格式
        decodedBoxes = torch.cat([boxXY, boxWH], dim=-1)

        return decodedBoxes


    def buildTargets(self, targets, gridH, gridW, anchors, batchSize, device):
        """
        为目标分配构建匹配的目标张量
        """
        # 初始化目标张量
        targetMask = torch.zeros(batchSize, len(anchors), gridH, gridW, 1, device=device)
        targetBoxes = torch.zeros(batchSize, len(anchors), gridH, gridW, 4, device=device)
        targetConf = torch.zeros(batchSize, len(anchors), gridH, gridW, 1, device=device)
        targetCls = torch.zeros(batchSize, len(anchors), gridH, gridW, self.numClasses, device=device)

        # 处理每个批次
        for batchIdx in range(batchSize):
            batchTargets = targets[batchIdx]

            # 过滤有效目标 (confidence > 0)
            validTargets = batchTargets[batchTargets[..., 4] > 0]

            for target in validTargets:
                # 提取目标信息
                gtBox = target[:4]  # [midX, midY, width, height]
                gtCls = target[5:5 + self.numClasses]  # 类别one-hot

                # 计算目标所在的网格位置
                gridX = int(gtBox[0] * gridW)
                gridY = int(gtBox[1] * gridH)

                # 确保网格坐标在有效范围内
                gridX = max(0, min(gridX, gridW - 1))
                gridY = max(0, min(gridY, gridH - 1))

                # 计算与每个锚框的IoU
                ious = []
                for anchorIdx, anchor in enumerate(anchors):
                    # 构建锚框 [x_center, y_center, width, height]
                    anchorBox = torch.tensor([0.5, 0.5, anchor[0] / self.inputSize, anchor[1] / self.inputSize],
                                             device=device)
                    gtBoxNormalized = gtBox.clone()
                    gtBoxNormalized[2:] = gtBoxNormalized[2:] / self.inputSize  # 归一化宽高

                    iou = self.calculateIoU(anchorBox, gtBoxNormalized)
                    ious.append(iou)

                # 选择最佳锚框
                bestAnchorIdx = torch.argmax(torch.tensor(ious))

                # 设置目标
                targetMask[batchIdx, bestAnchorIdx, gridY, gridX] = 1
                targetConf[batchIdx, bestAnchorIdx, gridY, gridX] = 1
                targetBoxes[batchIdx, bestAnchorIdx, gridY, gridX] = gtBox
                targetCls[batchIdx, bestAnchorIdx, gridY, gridX] = gtCls

        return targetMask, targetBoxes, targetConf, targetCls

    def calculateIoU(self, box1, box2):
        """
        计算两个框的IoU
        """
        # 转换为 [x1, y1, x2, y2] 格式
        box1X1 = box1[0] - box1[2] / 2
        box1Y1 = box1[1] - box1[3] / 2
        box1X2 = box1[0] + box1[2] / 2
        box1Y2 = box1[1] + box1[3] / 2

        box2X1 = box2[0] - box2[2] / 2
        box2Y1 = box2[1] - box2[3] / 2
        box2X2 = box2[0] + box2[2] / 2
        box2Y2 = box2[1] + box2[3] / 2

        # 计算交集
        interX1 = torch.max(box1X1, box2X1)
        interY1 = torch.max(box1Y1, box2Y1)
        interX2 = torch.min(box1X2, box2X2)
        interY2 = torch.min(box1Y2, box2Y2)

        interArea = torch.clamp(interX2 - interX1, min=0) * torch.clamp(interY2 - interY1, min=0)

        # 计算并集
        box1Area = (box1X2 - box1X1) * (box1Y2 - box1Y1)
        box2Area = (box2X2 - box2X1) * (box2Y2 - box2Y1)
        unionArea = box1Area + box2Area - interArea + self.eps

        return interArea / unionArea

    def calculateCoordinateLoss(self, predBoxes, targetBoxes, targetMask):
        """
        计算坐标损失 (使用CIoU损失)
        """
        # 只计算有目标的位置
        posMask = targetMask.squeeze(-1) > 0

        if not posMask.any():
            return torch.tensor(0.0, device=predBoxes.device)

        predPos = predBoxes[posMask]
        targetPos = targetBoxes[posMask]

        # 计算CIoU损失
        ciouLoss = self.calculateCIoULoss(predPos, targetPos)

        # 检查是否为NaN
        if torch.isnan(ciouLoss).any():
            print("Warning: CIoU loss contains NaN, returning zero")
            return torch.tensor(0.0, device=predBoxes.device)

        return ciouLoss.mean()


    def calculateCIoULoss(self, predBoxes, targetBoxes):
        """
        计算Complete IoU损失
        """
        # 转换为 [x1, y1, x2, y2] 格式
        predX1 = predBoxes[:, 0] - predBoxes[:, 2] / 2
        predY1 = predBoxes[:, 1] - predBoxes[:, 3] / 2
        predX2 = predBoxes[:, 0] + predBoxes[:, 2] / 2
        predY2 = predBoxes[:, 1] + predBoxes[:, 3] / 2

        targetX1 = targetBoxes[:, 0] - targetBoxes[:, 2] / 2
        targetY1 = targetBoxes[:, 1] - targetBoxes[:, 3] / 2
        targetX2 = targetBoxes[:, 0] + targetBoxes[:, 2] / 2
        targetY2 = targetBoxes[:, 1] + targetBoxes[:, 3] / 2

        # 计算IoU
        interX1 = torch.max(predX1, targetX1)
        interY1 = torch.max(predY1, targetY1)
        interX2 = torch.min(predX2, targetX2)
        interY2 = torch.min(predY2, targetY2)

        interArea = torch.clamp(interX2 - interX1, min=0) * torch.clamp(interY2 - interY1, min=0)

        predArea = (predX2 - predX1) * (predY2 - predY1)
        targetArea = (targetX2 - targetX1) * (targetY2 - targetY1)
        unionArea = predArea + targetArea - interArea + self.eps

        iou = interArea / unionArea

        # 计算中心点距离
        predCenterX = predBoxes[:, 0]
        predCenterY = predBoxes[:, 1]
        targetCenterX = targetBoxes[:, 0]
        targetCenterY = targetBoxes[:, 1]

        centerDistance = torch.pow(predCenterX - targetCenterX, 2) + torch.pow(predCenterY - targetCenterY, 2)

        # 计算最小包围框对角线距离
        encloseX1 = torch.min(predX1, targetX1)
        encloseY1 = torch.min(predY1, targetY1)
        encloseX2 = torch.max(predX2, targetX2)
        encloseY2 = torch.max(predY2, targetY2)

        encloseDiagonal = torch.pow(encloseX2 - encloseX1, 2) + torch.pow(encloseY2 - encloseY1, 2) + self.eps

        # 计算宽高比
        predW, predH = predBoxes[:, 2], predBoxes[:, 3]
        targetW, targetH = targetBoxes[:, 2], targetBoxes[:, 3]

        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(targetW / targetH) - torch.atan(predW / predH), 2)

        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # 计算CIoU
        ciou = iou - (centerDistance / encloseDiagonal) - alpha * v

        return 1 - ciou

    def calculateConfidenceLoss(self, predConf, targetConf, targetMask):
        """
        计算置信度损失
        """
        # 有目标的位置
        posMask = targetMask.squeeze(-1) > 0
        # 无目标的位置
        negMask = ~posMask

        # 有目标的置信度损失
        posLoss = F.binary_cross_entropy(predConf[posMask], targetConf[posMask], reduction='sum')

        # 无目标的置信度损失 (使用较小的权重)
        negLoss = F.binary_cross_entropy(predConf[negMask], targetConf[negMask], reduction='sum')

        # 计算平均损失
        numPos = max(posMask.sum().item(), 1)
        numNeg = max(negMask.sum().item(), 1)

        totalLoss = (posLoss / numPos) + (self.lambdaNoObj * negLoss / numNeg)

        # 检查是否为NaN
        if torch.isnan(totalLoss):
            print("Warning: Confidence loss is NaN, returning zero")
            totalLoss = torch.tensor(0.0, device=predConf.device)

        return totalLoss

    def calculateClassLoss(self, predCls, targetCls, targetMask):
        """
        计算类别损失
        """
        # 只计算有目标的位置
        posMask = targetMask.squeeze(-1) > 0

        if not posMask.any():
            return torch.tensor(0.0, device=predCls.device)

        predPos = predCls[posMask]
        targetPos = targetCls[posMask]

        # 使用二元交叉熵损失
        clsLoss = F.binary_cross_entropy(predPos, targetPos, reduction='mean')

        # 检查是否为NaN
        if torch.isnan(clsLoss):
            print("Warning: Class loss is NaN, returning zero")
            return torch.tensor(0.0, device=predCls.device)

        return clsLoss





def main():
    loss = YOLOv4Loss(NUM_CLASSES,
                      ANCHORS,
                      IMAGE_SIZE,
                      LAMBDA_COORD,
                      LAMBDA_NO_OBJ,
                      LAMBDA_CLS
                      )

    return loss

