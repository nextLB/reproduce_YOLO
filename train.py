"""
    YOLO的训练程序
"""


# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLO的训练主程序文件
# 2025.10.17 (V1.1)            --- by next, 目标为实现YOLO训练程序的多版本复用


import datasets
import YOLO_V3.models
import YOLO_V3.loss
import config_parameter
import torch.optim as optim
import torch
from tqdm import tqdm



# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def YOLOV3_VOC2007_TRAIN_MAIN():


    # 创建与获取数据集
    trainDataLoader, valDataLoader = datasets.VOC2007_MAIN()

    # 创建与获取YOLOV3模型
    model = YOLO_V3.models.main()

    # 定义优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=config_parameter.LEARNING_RATE, momentum=config_parameter.MOMENTUM, weight_decay=config_parameter.WEIGHT_DECAY)

    # 学习率调度器  多步学习率调度器   milestones=[50, 80]: 触发学习率调整的epoch位置   gamma=0.1: 学习率衰减的乘数因子
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    # 创建损失函数
    criterion = YOLO_V3.loss.YOLOv3Loss()


    # 训练循环
    for epoch in range(config_parameter.MAX_EPOCHS):
        model.train()
        trainTotalLoss = 0
        trainLoop = tqdm(trainDataLoader, desc="training")
        for batchIdx, (images, targets) in enumerate(trainLoop):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)







if __name__ == '__main__':
    YOLOV3_VOC2007_TRAIN_MAIN()
