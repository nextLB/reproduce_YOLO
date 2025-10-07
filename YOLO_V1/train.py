"""
    YOLOV1的训练程序
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import models
import config_parameter
from loss import SumSquaredErrorLoss
import datasets
from feature_maps import MyFeatureMapHook

saveFeatureMapsPath = './feature_maps'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 在这里定义一下要可视化的特征层
targetLayers = [
    'backbone_before_identity',
    'reshape_after_identity',
    'detectionNet_after_identity'
]


def main():
    # SummaryWriter是使用pytorch的TensorBoard集成功能
    # 在终端运行 tensorboard --logdir=runs 然后浏览器打开 http://localhost:6006
    writer = SummaryWriter()    # 默认创建 runs/当前时间 目录
    now = datetime.now()


    model = models.YOLOv1ResNet().to(device)
    lossFunction = SumSquaredErrorLoss()


    # Adam works better
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_parameter.LEARNING_RATE
    )

    # Learning rate scheduler (NOT NEEDED)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=utils.scheduler_lambda
    # )

    # Load the dataset
    trainDataLoader, valDataLoader = datasets.main()

    # Create folders
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weightDir = os.path.join(root, 'weights')
    if not os.path.isdir(weightDir):
        os.makedirs(weightDir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))

    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)


    #####################
    #       Train       #
    #####################
    for epoch in tqdm(range(config_parameter.MAX_EPOCHS), desc='Epoch'):
        model.train()
        trainLoss = 0
        batchIdx = 0
        for originalData, augmentationData, groundTruth in tqdm(trainDataLoader, desc='Train', leave=False):
            data = augmentationData.to(device)
            labels = groundTruth.to(device)

            # TODO: 注册特征层
            if epoch % 10 == 0 and batchIdx % 100 == 0:
                # initial feature hook
                hookHandler = MyFeatureMapHook(model,
                                               outputDir=f"{saveFeatureMapsPath}/epoch_{epoch}_batchIndex_{batchIdx}",
                                               imgIndex=0)
                hookHandler.register_hooks(targetLayers)

            optimizer.zero_grad()
            predictions = model.forward(data)

            # TODO: 保存特征层
            if epoch % 10 == 0 and batchIdx % 100 == 0:
                # save feature maps
                hookHandler.save_feature_maps()
                hookHandler.remove_hooks()

            loss = lossFunction(predictions, labels)
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() / len(trainDataLoader)
            batchIdx += 1
            del data, labels

        # Step and graph scheduler once an epoch
        # writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        # scheduler.step()

        train_losses = np.append(train_losses, [[epoch], [trainLoss]], axis=1)
        writer.add_scalar('Loss/train', trainLoss, epoch)


        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for originalData, augmentationData, groundTruth in tqdm(valDataLoader, desc='Test', leave=False):
                    data = augmentationData.to(device)
                    labels = groundTruth.to(device)

                    predictions = model.forward(data)
                    loss = lossFunction(predictions, labels)

                    test_loss += loss.item() / len(valDataLoader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()


    save_metrics()
    torch.save(model.state_dict(), os.path.join(weightDir, 'final'))


if __name__ == '__main__':
    main()






