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
        self.mseLoss = nn.MSELoss()
        self.bceLoss = nn.BCELoss()
        self.objScale = 1
        self.noobjScale = 100
        self.classScale = 1
        self.coordScale = 5


    def forward(self, predictions, targets):
        loss = 0
        for i, prediction in enumerate(predictions):
            # 获取当前尺度的锚框
            anchors = config_paramters.ANCHORS[i]
            gridSize = prediction.size(2)

            # 转换预测格式
            predBoxes = prediction[..., :4]
            predConf = prediction[..., 4:5]
            predCls = prediction[..., 5:]

            # 构建目标张量
            self.build_targets(prediction, targets, anchors, gridSize)



    def build_targets(self, prediction, targets, anchors, gridSize):
        batchSize = prediction.size(0)
        numAnchors = len(anchors)

        # 初始化目标张量
        targetTensor = torch.zeros(batchSize, numAnchors, gridSize, gridSize, 5 + config_paramters.NUM_CLASSES, device=prediction.device)

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



