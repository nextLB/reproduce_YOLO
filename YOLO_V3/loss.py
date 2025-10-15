"""
    YOLOV3  loss函数的具体实现
"""
import torch
import torch.nn as nn
import config_paramters


# YOLOv3损失函数的定义
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mseLoss = nn.MSELoss(reduction='sum')
        self.bceLoss = nn.BCEWithLogitsLoss(reduction='sum')
        self.objScale = 1
        self.noobjScale = 0.5
        self.classScale = 1
        self.coordScale = 5


    def forward(self, predictions, targets):
        loss = 0
        totalObjects = 0
        for i, prediction in enumerate(predictions):
            # 获取当前尺度的锚框
            anchors = config_paramters.ANCHORS[i]
            gridSize = prediction.size(2)

            # 获取预测的各个部分
            x = prediction[..., 0]      # 中心点x
            y = prediction[..., 1]      # 中心点y
            w = prediction[..., 2]      # 宽度
            h = prediction[..., 3]      # 高度
            predConf = prediction[..., 4]       # 置信度
            predCls = prediction[..., 5:]       # 分类


            # 构建目标张量
            targetTensor, objMask, noobjMask = self.build_targets(prediction, targets, anchors, gridSize)

            # 计算有目标的网格数量
            nObjects = objMask.sum()
            totalObjects += nObjects

            if nObjects > 0:
                # 边界框坐标损失
                lossX = self.mseLoss(x[objMask], targetTensor[objMask][:, 0])
                lossY = self.mseLoss(y[objMask], targetTensor[objMask][:, 1])

                # 边界框尺寸损失
                lossW = self.mseLoss(w[objMask], targetTensor[objMask][:, 2])
                lossH = self.mseLoss(h[objMask], targetTensor[objMask][:, 3])

                loss += self.coordScale * (lossX + lossY + lossW + lossH)

                # 分类损失
                lossCls = self.bceLoss(predCls[objMask], targetTensor[objMask][:, 5:])
                loss += self.classScale * lossCls

            # 置信度损失
            lossConfObj = self.bceLoss(predConf[objMask], targetTensor[objMask][:, 4])
            lossConfNoobj = self.bceLoss(predConf[noobjMask], targetTensor[noobjMask][:, 4])
            loss += self.objScale * lossConfObj + self.noobjScale * lossConfNoobj

        # 如果没有检测到如何目标，返回一个基础损失避免除零
        if totalObjects == 0:
            return torch.tensor(1.0, device=predictions[0].device, requires_grad=True)

        return loss / totalObjects



    def build_targets(self, prediction, targets, anchors, gridSize):
        batchSize = prediction.size(0)
        numAnchors = len(anchors)

        # 初始化目标张量
        targetTensor = torch.zeros(batchSize, numAnchors, gridSize, gridSize, 5 + config_paramters.NUM_CLASSES, device=prediction.device)
        # 初始化对象掩码
        objMask = torch.zeros(batchSize, numAnchors, gridSize, gridSize,
                              device=prediction.device, dtype=torch.bool)
        noobjMask = torch.ones(batchSize, numAnchors, gridSize, gridSize,
                               device=prediction.device, dtype=torch.bool)

        # 将anchor缩放到当前特征图尺度
        scaled_anchors = []
        for anchor in anchors:
            anchor_w = anchor[0] / (config_paramters.IMAGE_SIZE / gridSize)
            anchor_h = anchor[1] / (config_paramters.IMAGE_SIZE / gridSize)
            scaled_anchors.append((anchor_w, anchor_h))

        # 为每个目标分配锚框
        for batchIdx in range(batchSize):

            # 获取当前batch的所有目标，形状为[num_targets, 6]
            batchTargets = targets[batchIdx]

            # 过滤掉填充的目标
            validMask = batchTargets[:, 0] >= 0
            batchTargets = batchTargets[validMask]

            if len(batchTargets) == 0:
                continue


            # 单个目标的来计算
            for target in batchTargets:
                # target的length 应该是5 然后第一个是class_id
                classId = int(target[0])
                box = target[1:5]       # 后面四个是边界框坐标

                # 转换边界框格式
                xCenter, yCenter, width, height = box
                xCenter *= gridSize
                yCenter *= gridSize
                width *= gridSize
                height *= gridSize

                # 找到最佳匹配框
                bestIou = 0
                bestAnchor = 0

                for anchorIdx, anchor in enumerate(scaled_anchors):
                    anchorW, anchorH = anchor

                    # 计算IoU

                    # 计算交集
                    inter_width = min(width, anchorW)
                    inter_height = min(height, anchorH)
                    inter = inter_width * inter_height

                    # 计算并集
                    union = width * height + anchorW * anchorH - inter
                    iou = inter / (union + 1e-16)

                    if iou > bestIou:
                        bestIou = iou
                        bestAnchor = anchorIdx



                # 计算网格位置
                gridX = int(xCenter)
                gridY = int(yCenter)

                # 确保网格坐标在有效范围内
                if 0 <= gridX < gridSize and 0 <= gridY < gridSize:

                    # 设置目标值
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 0] = xCenter - gridX
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 1] = yCenter - gridY
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 2] = torch.log(width / scaled_anchors[bestAnchor][0] + 1e-16)
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 3] = torch.log(height / scaled_anchors[bestAnchor][1] + 1e-16)
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 4] = 1     # 对象置信度
                    targetTensor[batchIdx, bestAnchor, gridY, gridX, 5 + classId] = 1       # 类别概率

                    objMask[batchIdx, bestAnchor, gridY, gridX] = 1
                    noobjMask[batchIdx, bestAnchor, gridY, gridX] = 0


        return targetTensor, objMask, noobjMask

