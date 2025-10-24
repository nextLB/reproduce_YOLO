"""
    YOLOV4损失函数的实现
"""



# 2025.10.21 (V1.1)            --- by next, 实现了YOLOV4模型与真实标签值之间的计算


import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super(YOLOv4Loss, self).__init__()
        self.numClasses = numClasses
        self.numAnchors = len(anchors) // 3  # 3个尺度
        self.inputSize = imageSize
        self.lambdaCoord = lambdaCoord
        self.lambdaNoObj = lambdaNoobj
        self.lambdaCls = lambdaCls

        # 转换为tensor并分组
        self.anchors = torch.tensor(anchors, dtype=torch.float32).view(3, 3, 2)
        self.eps = 1e-7

    def forward(self, predictions, targets):
        device = predictions[0].device
        self.anchors = self.anchors.to(device)

        totalLoss = torch.tensor(0.0, device=device)
        lossCoord = torch.tensor(0.0, device=device)
        lossConf = torch.tensor(0.0, device=device)
        lossCls = torch.tensor(0.0, device=device)

        for scaleIdx, pred in enumerate(predictions):
            batchSize, numAnchors, gridH, gridW, _ = pred.shape

            # 当前尺度的锚框
            scaleAnchors = self.anchors[scaleIdx]

            # 重塑预测
            pred = pred.view(batchSize, numAnchors, gridH, gridW, 5 + self.numClasses)

            # 使用更稳定的解码方式
            predBoxes, predConf, predCls = self.decodePredictionsStable(
                pred, gridH, gridW, scaleAnchors, device)

            # 构建目标
            targetMask, targetBoxes, targetConf, targetCls = self.buildTargetsStable(
                targets, gridH, gridW, scaleAnchors, batchSize, device)

            # 计算损失（带数值检查）
            scaleLossCoord = self.calculateCoordinateLossStable(predBoxes, targetBoxes, targetMask)
            scaleLossConf = self.calculateConfidenceLossStable(predConf, targetConf, targetMask)
            scaleLossCls = self.calculateClassLossStable(predCls, targetCls, targetMask)

            # 累加
            lossCoord += scaleLossCoord
            lossConf += scaleLossConf
            lossCls += scaleLossCls

        # 总损失
        totalLoss = (self.lambdaCoord * lossCoord + lossConf + self.lambdaCls * lossCls)

        lossComponents = {
            'coordLoss': lossCoord.item(),
            'confLoss': lossConf.item(),
            'classLoss': lossCls.item()
        }

        return totalLoss, lossComponents

    def decodePredictionsStable(self, pred, gridH, gridW, anchors, device):
        """更稳定的预测解码"""
        batchSize, numAnchors, _, _, _ = pred.shape

        # 创建网格
        gridY, gridX = torch.meshgrid(
            torch.arange(gridH, device=device),
            torch.arange(gridW, device=device),
            indexing='ij'
        )
        gridX = gridX.view(1, 1, gridH, gridW).repeat(batchSize, numAnchors, 1, 1)
        gridY = gridY.view(1, 1, gridH, gridW).repeat(batchSize, numAnchors, 1, 1)
        grid = torch.stack([gridX, gridY], dim=-1).float()

        # 提取预测值
        predXY = torch.sigmoid(pred[..., :2])  # 使用sigmoid确保在[0,1]
        predWH = pred[..., 2:4]

        # 限制wh的范围，防止指数爆炸
        predWH = torch.clamp(predWH, -10, 10)

        # 解码坐标
        boxXY = (predXY + grid) / torch.tensor([gridW, gridH], device=device).view(1, 1, 1, 1, 2)
        anchors = anchors.view(1, numAnchors, 1, 1, 2)
        boxWH = (torch.exp(predWH) * anchors) / self.inputSize

        # 确保坐标在有效范围内
        boxXY = torch.clamp(boxXY, 0.0, 1.0)
        boxWH = torch.clamp(boxWH, 0.0, 1.0)

        predBoxes = torch.cat([boxXY, boxWH], dim=-1)

        # 置信度和类别
        predConf = torch.sigmoid(pred[..., 4:5])
        predCls = torch.sigmoid(pred[..., 5:])

        return predBoxes, predConf, predCls

    def buildTargetsStable(self, targets, gridH, gridW, anchors, batchSize, device):
        """更稳定的目标构建"""
        targetMask = torch.zeros(batchSize, len(anchors), gridH, gridW, 1, device=device)
        targetBoxes = torch.zeros(batchSize, len(anchors), gridH, gridW, 4, device=device)
        targetConf = torch.zeros(batchSize, len(anchors), gridH, gridW, 1, device=device)
        targetCls = torch.zeros(batchSize, len(anchors), gridH, gridW, self.numClasses, device=device)

        for batchIdx in range(batchSize):
            batchTargets = targets[batchIdx]
            validTargets = batchTargets[batchTargets[..., 4] > 0]

            for target in validTargets:
                gtBox = target[:4].clone()
                gtCls = target[5:5 + self.numClasses].clone()

                # 确保gtBox在有效范围内
                gtBox = torch.clamp(gtBox, 0.0, 1.0)

                # 计算网格位置
                gridX = int(gtBox[0] * gridW)
                gridY = int(gtBox[1] * gridH)
                gridX = max(0, min(gridX, gridW - 1))
                gridY = max(0, min(gridY, gridH - 1))

                # 选择最佳锚框
                bestAnchorIdx = 0
                bestIou = 0

                for anchorIdx, anchor in enumerate(anchors):
                    anchorBox = torch.tensor([0.5, 0.5, anchor[0] / self.inputSize, anchor[1] / self.inputSize],
                                             device=device)
                    gtBoxNormalized = gtBox.clone()
                    gtBoxNormalized[2:] = gtBoxNormalized[2:]  # 已经是归一化的

                    iou = self.calculateIoUStable(anchorBox, gtBoxNormalized)
                    if iou > bestIou:
                        bestIou = iou
                        bestAnchorIdx = anchorIdx

                # 分配目标
                targetMask[batchIdx, bestAnchorIdx, gridY, gridX] = 1
                targetConf[batchIdx, bestAnchorIdx, gridY, gridX] = 1
                targetBoxes[batchIdx, bestAnchorIdx, gridY, gridX] = gtBox
                targetCls[batchIdx, bestAnchorIdx, gridY, gridX] = gtCls

        return targetMask, targetBoxes, targetConf, targetCls

    def calculateIoUStable(self, box1, box2):
        """稳定的IoU计算"""
        # 转换为xyxy格式
        b1_x1, b1_y1, b1_x2, b1_y2 = self.boxToXyxy(box1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self.boxToXyxy(box2)

        # 交集
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 并集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + self.eps

        return inter_area / union_area

    def boxToXyxy(self, box):
        """将[x_center, y_center, width, height]转换为[x1, y1, x2, y2]"""
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        return x1, y1, x2, y2

    def calculateCoordinateLossStable(self, predBoxes, targetBoxes, targetMask):
        """稳定的坐标损失计算"""
        posMask = targetMask.squeeze(-1) > 0

        if not posMask.any():
            return torch.tensor(0.0, device=predBoxes.device)

        predPos = predBoxes[posMask]
        targetPos = targetBoxes[posMask]

        # 数值检查
        if torch.isnan(predPos).any() or torch.isnan(targetPos).any():
            print("Warning: NaN in coordinate loss input, skipping")
            return torch.tensor(0.0, device=predBoxes.device)

        # 使用更简单的IoU损失而不是CIoU
        iouLoss = self.calculateIoULoss(predPos, targetPos)

        return iouLoss.mean()

    def calculateIoULoss(self, predBoxes, targetBoxes):
        """简单的IoU损失"""
        ious = []
        for i in range(len(predBoxes)):
            iou = self.calculateIoUStable(predBoxes[i], targetBoxes[i])
            ious.append(iou)

        ious = torch.stack(ious)
        return 1 - ious

    def calculateConfidenceLossStable(self, predConf, targetConf, targetMask):
        """稳定的置信度损失计算"""
        posMask = targetMask.squeeze(-1) > 0
        negMask = ~posMask

        # 严格限制数值范围
        predConf = torch.clamp(predConf, 1e-10, 1.0 - 1e-10)
        targetConf = torch.clamp(targetConf, 0.0, 1.0)

        # 检查是否有有效位置
        if posMask.sum() == 0 and negMask.sum() == 0:
            return torch.tensor(0.0, device=predConf.device)

        totalLoss = torch.tensor(0.0, device=predConf.device)

        # 正样本损失
        if posMask.sum() > 0:
            posPred = predConf[posMask]
            posTarget = targetConf[posMask]

            # 最终检查
            if not (torch.all(posPred >= 0) and torch.all(posPred <= 1) and
                    torch.all(posTarget >= 0) and torch.all(posTarget <= 1)):
                print("Warning: Invalid values in positive confidence loss")
            else:
                posLoss = F.binary_cross_entropy(posPred, posTarget, reduction='sum')
                totalLoss += posLoss / max(posMask.sum().item(), 1)

        # 负样本损失
        if negMask.sum() > 0:
            negPred = predConf[negMask]
            negTarget = targetConf[negMask]

            # 最终检查
            if not (torch.all(negPred >= 0) and torch.all(negPred <= 1) and
                    torch.all(negTarget >= 0) and torch.all(negTarget <= 1)):
                print("Warning: Invalid values in negative confidence loss")
            else:
                negLoss = F.binary_cross_entropy(negPred, negTarget, reduction='sum')
                totalLoss += self.lambdaNoObj * negLoss / max(negMask.sum().item(), 1)

        return totalLoss

    def calculateClassLossStable(self, predCls, targetCls, targetMask):
        """稳定的类别损失计算"""
        posMask = targetMask.squeeze(-1) > 0

        if not posMask.any():
            return torch.tensor(0.0, device=predCls.device)

        predPos = predCls[posMask]
        targetPos = targetCls[posMask]

        # 严格限制数值范围
        predPos = torch.clamp(predPos, 1e-10, 1.0 - 1e-10)
        targetPos = torch.clamp(targetPos, 0.0, 1.0)

        # 最终检查
        if not (torch.all(predPos >= 0) and torch.all(predPos <= 1) and
                torch.all(targetPos >= 0) and torch.all(targetPos <= 1)):
            print("Warning: Invalid values in class loss, returning zero")
            return torch.tensor(0.0, device=predCls.device)

        return F.binary_cross_entropy(predPos, targetPos, reduction='mean')





def main():
    loss = YOLOv4Loss(NUM_CLASSES,
                      ANCHORS,
                      IMAGE_SIZE,
                      LAMBDA_COORD,
                      LAMBDA_NO_OBJ,
                      LAMBDA_CLS
                      )

    return loss

