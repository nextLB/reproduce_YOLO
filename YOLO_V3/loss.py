"""
    YOLOV3的损失函数具体实现的程序
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLOV3的损失函数
# 2025.10.17 (V1.1)            --- by next, 使用了更通用的接口形式更新了YOLOv3的损失函数



import torch
import torch.nn as nn



ANCHORS = [
            [(10, 13), (16, 30), (33, 23)],   # P3/8
            [(30, 61), (62, 45), (59, 119)],  # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]
NUM_CLASSES = 20
IMAGE_SIZE = 480




class YOLOv3Loss(nn.Module):
    def __init__(self):
        super(YOLOv3Loss, self).__init__()
        self.anchors = ANCHORS
        self.numClasses = NUM_CLASSES
        self.imageSize = IMAGE_SIZE
        # 参数一：'none' - 不进行降维
        # 含义：返回与输入相同形状的损失张量，每个位置都保留独立的损失值
        # 使用场景：当你需要对不同位置的损失进行加权或选择性处理时
        # 参数二：'mean' - 求平均值（默认）
        # 含义：返回所有元素损失的平均值
        # 使用场景：大多数标准训练场景
        # 参数三：'sum' - 求和
        # 含义：返回所有元素损失的总和
        # 使用场景：当你想要总损失而不是平均损失时
        self.mseLoss = nn.MSELoss(reduction='none')
        self.bceLoss = nn.BCELoss(reduction='none')


        # 损失权重的设定
        self.lambdaCoord = 5.0
        self.lambdaObj = 1.0
        self.lambdaNoObj = 0.5
        self.lambdaClass = 1.0

        # 将锚框转换为适合三个尺度的格式
        self.anchorBoxes = self._process_anchors(ANCHORS)

    def _process_anchors(self, anchors):
        """处理锚框，为每个尺度创建锚框张量"""
        processedAnchors = []
        for anchorGroup in anchors:
            anchorTensor = torch.tensor(anchorGroup, dtype=torch.float32)
            processedAnchors.append(anchorTensor)
        return processedAnchors

    def _build_targets(self, predictions, groundTruths, anchorIndices):
        """
        构建目标张量，将ground truth映射到对应的预测尺度
        """
        batchSize, numAnchors, gridSize, _, _ = predictions.shape
        device = predictions.device

        # 初始化目标张量
        objMask = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        noObjMask = torch.ones(batchSize, numAnchors, gridSize, gridSize, device=device)
        tx = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        ty = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        tw = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        th = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        tConf = torch.zeros(batchSize, numAnchors, gridSize, gridSize, device=device)
        tClass = torch.zeros(batchSize, numAnchors, gridSize, gridSize, self.numClasses, device=device)

        # 当前尺度的锚框
        anchors = self.anchorBoxes[anchorIndices].to(device)

        for batchIdx in range(batchSize):
            groundTruth = groundTruths[batchIdx]

            # 过滤掉无效的ground truth（置信度为0的）
            validIndices = groundTruth[:, 4] > 0.5
            if not validIndices.any():
                continue

            validGt = groundTruth[validIndices]

            for objIdx in range(validGt.shape[0]):
                gt = validGt[objIdx]

                # [midX, midY, width, height, conf, one_hot_classes]
                midX = gt[0]
                midY = gt[1]
                width = gt[2]
                height = gt[3]
                conf = gt[4]
                classVec = gt[5:5 + self.numClasses]  # 取类别部分

                # 计算网格位置
                gridX = int(midX * gridSize)
                gridY = int(midY * gridSize)

                # 确保网格坐标在有效范围内
                gridX = max(0, min(gridX, gridSize - 1))
                gridY = max(0, min(gridY, gridSize - 1))

                # 计算与锚框的IoU
                gtBox = torch.tensor([width * self.imageSize, height * self.imageSize], device=device)
                ious = []
                for anchor in anchors:
                    anchorBox = anchor.clone()
                    inter = torch.min(gtBox[0], anchorBox[0]) * torch.min(gtBox[1], anchorBox[1])
                    union = gtBox[0] * gtBox[1] + anchorBox[0] * anchorBox[1] - inter
                    iou = inter / (union + 1e-16)
                    ious.append(iou)

                # 选择最佳锚框
                bestAnchor = torch.argmax(torch.tensor(ious))

                # 设置目标值
                objMask[batchIdx, bestAnchor, gridY, gridX] = 1
                noObjMask[batchIdx, bestAnchor, gridY, gridX] = 0

                tx[batchIdx, bestAnchor, gridY, gridX] = midX * gridSize - gridX
                ty[batchIdx, bestAnchor, gridY, gridX] = midY * gridSize - gridY
                tw[batchIdx, bestAnchor, gridY, gridX] = torch.log(
                    width * self.imageSize / anchors[bestAnchor][0] + 1e-16)
                th[batchIdx, bestAnchor, gridY, gridX] = torch.log(
                    height * self.imageSize / anchors[bestAnchor][1] + 1e-16)
                tConf[batchIdx, bestAnchor, gridY, gridX] = 1
                tClass[batchIdx, bestAnchor, gridY, gridX] = classVec[:self.numClasses]

        return objMask, noObjMask, tx, ty, tw, th, tConf, tClass

    def forward(self, predictions, groundTruths):
        """
        计算YOLOv3损失

        Args:
            predictions: 三个尺度的预测列表 [pred1, pred2, pred3]
            groundTruths: 真实标签 [batch_size, 300, 5 + num_classes]

        Returns:
            totalLoss: 总损失
            lossComponents: 各损失分量字典
        """
        totalLoss = 0
        lossComponents = {
            'coordLoss': 0,
            'objLoss': 0,
            'noObjLoss': 0,
            'classLoss': 0
        }

        # 对三个尺度分别计算损失
        for scaleIdx, pred in enumerate(predictions):
            batchSize, numAnchors, gridSize, _, predDim = pred.shape

            # 重塑预测张量
            pred = pred.view(batchSize, numAnchors, gridSize, gridSize, predDim)

            # 获取预测值
            predX = pred[..., 0]
            predY = pred[..., 1]
            predW = pred[..., 2]
            predH = pred[..., 3]
            predConf = pred[..., 4]
            predClass = pred[..., 5:]

            # 构建目标      生成适合yolov3计算的目标值
            objMask, noObjMask, tx, ty, tw, th, tConf, tClass = self._build_targets(
                pred, groundTruths, scaleIdx
            )

            # 坐标损失
            coordLossX = self.mseLoss(predX * objMask, tx * objMask).sum()
            coordLossY = self.mseLoss(predY * objMask, ty * objMask).sum()
            coordLossW = self.mseLoss(predW * objMask, tw * objMask).sum()
            coordLossH = self.mseLoss(predH * objMask, th * objMask).sum()
            coordLoss = (coordLossX + coordLossY + coordLossW + coordLossH) * self.lambdaCoord

            # 置信度损失
            objLoss = self.bceLoss(predConf * objMask, tConf * objMask).sum() * self.lambdaObj
            noObjLoss = self.bceLoss(predConf * noObjMask, tConf * noObjMask).sum() * self.lambdaNoObj

            # 类别损失
            classLoss = self.bceLoss(
                predClass[objMask.bool()],
                tClass[objMask.bool()]
            ).sum() * self.lambdaClass

            # 尺度损失
            scaleLoss = coordLoss + objLoss + noObjLoss + classLoss

            totalLoss += scaleLoss
            lossComponents['coordLoss'] += coordLoss.item()
            lossComponents['objLoss'] += objLoss.item()
            lossComponents['noObjLoss'] += noObjLoss.item()
            lossComponents['classLoss'] += classLoss.item()

        return totalLoss, lossComponents

