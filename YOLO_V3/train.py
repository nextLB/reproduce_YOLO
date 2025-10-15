"""
    YOLOV3的训练程序
"""
import datasets
from feature_maps import MyFeatureMapHook
import loss
import models
import torch
import utils
import torch.optim as optim
import config_paramters
from tqdm import tqdm


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():
    saveModelPath, saveLogFilePath, saveFeatureMapsPath = utils.create_all_path()

    # 创建与获取数据集
    trainDataLoader, valDataLoader = datasets.main()

    # 创建与获取YOLOV3模型
    model = models.main()

    # 定义优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=config_paramters.LEARNING_RATE, momentum=config_paramters.MOMENTUM, weight_decay=config_paramters.WEIGHT_DECAY)
    criterion = loss.YOLOLoss().to(device)

    # 学习率调度器  多步学习率调度器   milestones=[50, 80]: 触发学习率调整的epoch位置   gamma=0.1: 学习率衰减的乘数因子
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


    # 训练循环
    for epoch in range(config_paramters.MAX_EPOCHS):
        model.train()
        trainTotalLoss = 0
        trainLoop = tqdm(trainDataLoader, desc="training")

        for batchIdx, (images, targets) in enumerate(trainLoop):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)


            # 计算损失
            v3Loss = criterion(predictions, targets)

            # 反向传播
            optimizer.zero_grad()
            v3Loss.backward()
            optimizer.step()

            trainTotalLoss += v3Loss.item()

            # 更新进度条
            trainLoop.set_postfix(loss=v3Loss.item())

        trainAvgLoss = trainTotalLoss / len(trainDataLoader)
        print(f'本轮训练集的平均损失为: {trainAvgLoss}')

        # 更新学习率
        scheduler.step()


        # model.eval()
        # valTotalLoss = 0
        # valLoop = tqdm(valDataLoader, desc="valing")
        # for batchIdx, (images, targets) in enumerate(valLoop):
        #     images = images.to(device)
        #     targets = targets.to(device)
        #
        #     # 前向传播
        #     predictions = model(images)
        #
        #     # 计算损失
        #     v3Loss = criterion(predictions, targets)
        #
        #     valTotalLoss += v3Loss.item()
        #
        #     # 更新进度条
        #     valLoop.set_postfix(loss=v3Loss.item())
        #
        # valAvgLoss = valTotalLoss / len(valDataLoader)
        # print(f'本轮验证集的平均损失为: {valAvgLoss}')




if __name__ == '__main__':
    main()

