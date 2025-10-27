"""
    YOLO的训练程序
"""


# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLO的训练主程序文件
# 2025.10.17 (V1.1)            --- by next, 目标为实现YOLO训练程序的多版本复用


import datasets
import YOLO_V1.models
import YOLO_V1.loss
import YOLO_V3.models
import YOLO_V3.loss
import YOLO_V4.models
import YOLO_V4.loss
import config_parameter
import torch.optim as optim
import torch
from tqdm import tqdm
from datetime import datetime
import os
import gc
from torch.utils.tensorboard import SummaryWriter


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def YOLOV1_VOC2007_TRAIN_MAIN():
    # 创建模型存储的路径
    nowTime = datetime.now()
    saveModelPath = os.path.join('../YOLO_results/YOLO_V1', f'{nowTime.year}_{nowTime.day}_{nowTime.hour}_{nowTime.minute}_{nowTime.second}')
    os.makedirs(saveModelPath, exist_ok=True)

    # 创建TensorBoard日治目录
    # 本地查看运行：tensorboard --logdir=YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    # 然后打开浏览器，访问 http://localhost:6006（默认端口）
    # 网站服务器查看时，请运行 tensorboard dev upload --logdir YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    logDir = os.path.join(saveModelPath, 'YOLOV1_tensorboard_logs')
    os.makedirs(logDir, exist_ok=True)
    writer = SummaryWriter(log_dir=logDir)

    # 创建与获取数据集
    trainDataLoader, valDataLoader = datasets.VOC2007_MAIN()

    # 创建与获取YOLOV3模型
    model = YOLO_V1.models.YOLOv1ResNet().to(device)


    # 定义优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=config_parameter.LEARNING_RATE, momentum=config_parameter.MOMENTUM, weight_decay=config_parameter.WEIGHT_DECAY)

    # 学习率调度器  多步学习率调度器   milestones=[50, 80]: 触发学习率调整的epoch位置   gamma=0.1: 学习率衰减的乘数因子
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.1)

    # 创建损失函数
    criterion = YOLO_V1.loss.YOLOv1Loss()


    # 记录模型图结构
    sampleImages, _ = next(iter(trainDataLoader))
    sampleImages = sampleImages.to(device)
    writer.add_graph(model, sampleImages)


    bestValLoss = float('inf')
    globalStep = 0
    # 训练循环
    for epoch in range(config_parameter.MAX_EPOCHS):
        model.train()
        trainTotalLoss = 0
        trainLoop = tqdm(trainDataLoader, desc="training")

        # 记录每个epoch的训练损失分量累计值
        epochCoordLoss = 0
        epochObjLoss = 0
        epochNoobjLoss = 0
        epochClassLoss = 0
        for batchIdx, (images, targets) in enumerate(trainLoop):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)


            # 计算损失
            totalLoss, lossComponents = criterion(predictions, targets)

            # 反向传播
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            trainTotalLoss += totalLoss.item()

            # 累计损失分量
            epochCoordLoss += lossComponents["coordLoss"]
            epochObjLoss += lossComponents["objLoss"]
            epochNoobjLoss += lossComponents["noObjLoss"]
            epochClassLoss += lossComponents["classLoss"]

            # 记录每个batch的训练损失到TensorBoard
            writer.add_scalar('Train/Batch_Total_Loss', totalLoss.item(), globalStep)
            writer.add_scalar('Train/Batch_Coord_Loss', lossComponents["coordLoss"], globalStep)
            writer.add_scalar('Train/Batch_Obj_Loss', lossComponents["objLoss"], globalStep)
            writer.add_scalar('Train/Batch_NoObj_Loss', lossComponents["noObjLoss"], globalStep)
            writer.add_scalar('Train/Batch_Class_Loss', lossComponents["classLoss"], globalStep)

            globalStep += 1  # 更新全局步数

            # 更新进度条
            trainLoop.set_postfix({
                'Total Loss': f'{totalLoss.item():.4f}',
                'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                'Obj Loss': f'{lossComponents["objLoss"]:.4f}',
                'NoObj Loss': f'{lossComponents["noObjLoss"]:.4f}',
                'Class Loss': f'{lossComponents["classLoss"]:.4f}'
            })

        # 记录每个epoch的平均训练损失
        avgTrainLoss = trainTotalLoss / len(trainDataLoader)
        writer.add_scalar('Train/Epoch_Total_Loss', avgTrainLoss, epoch)
        writer.add_scalar('Train/Epoch_Coord_Loss', epochCoordLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_Obj_Loss', epochObjLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_NoObj_Loss', epochNoobjLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_Class_Loss', epochClassLoss / len(trainDataLoader), epoch)

        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # 更新学习率
        scheduler.step()

        # 打印epoch统计信息
        print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgTrainLoss:.4f}')

        # 如果符合轮次要求就进行验证
        if epoch % 5 == 0:
            model.eval()
            valTotalLoss = 0
            valLoop = tqdm(valDataLoader, desc="valing")

            valCoordLoss = 0
            valObjLoss = 0
            valNoobjLoss = 0
            valClassLoss = 0
            for batchIdx, (images, targets) in enumerate(valLoop):
                images = images.to(device)
                targets = targets.to(device)

                # 前向传播
                predictions = model(images)

                # 计算损失
                totalLoss, lossComponents = criterion(predictions, targets)

                valTotalLoss += totalLoss.item()

                valCoordLoss += lossComponents["coordLoss"]
                valObjLoss += lossComponents["objLoss"]
                valNoobjLoss += lossComponents["noObjLoss"]
                valClassLoss += lossComponents["classLoss"]

                # 更新进度条
                valLoop.set_postfix({
                    'Total Loss': f'{totalLoss.item():.4f}',
                    'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                    'Obj Loss': f'{lossComponents["objLoss"]:.4f}',
                    'NoObj Loss': f'{lossComponents["noObjLoss"]:.4f}',
                    'Class Loss': f'{lossComponents["classLoss"]:.4f}'
                })

            # 打印epoch统计信息
            # 记录验证损失到TensorBoard - 新增
            avgValLoss = valTotalLoss / len(valDataLoader)
            writer.add_scalar('Val/Epoch_Total_Loss', avgValLoss, epoch)
            writer.add_scalar('Val/Epoch_Coord_Loss', valCoordLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Obj_Loss', valObjLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_NoObj_Loss', valNoobjLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Class_Loss', valClassLoss / len(valDataLoader), epoch)
            print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgValLoss:.4f}')

            if avgValLoss <= bestValLoss:
                bestValLoss = avgValLoss
                torch.save(model.state_dict(), os.path.join(saveModelPath, "YOLO_V1_low_loss.pth"))
                print(f"模型已保存至 {saveModelPath}")


    # 关闭TensorBoard writer
    writer.close()

    # 清除缓存
    del model
    gc.collect()
    torch.cuda.empty_cache()




def YOLOV3_VOC2007_TRAIN_MAIN():

    # 创建模型存储的路径
    nowTime = datetime.now()
    saveModelPath = os.path.join('../YOLO_results/YOLO_V3', f'{nowTime.year}_{nowTime.day}_{nowTime.hour}_{nowTime.minute}_{nowTime.second}')
    os.makedirs(saveModelPath, exist_ok=True)

    # 创建TensorBoard日治目录
    # 本地查看运行：tensorboard --logdir=YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    # 然后打开浏览器，访问 http://localhost:6006（默认端口）
    # 网站服务器查看时，请运行 tensorboard dev upload --logdir YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    logDir = os.path.join(saveModelPath, 'YOLOV3_tensorboard_logs')
    os.makedirs(logDir, exist_ok=True)
    writer = SummaryWriter(log_dir=logDir)

    # 创建与获取数据集
    trainDataLoader, valDataLoader = datasets.VOC2007_MAIN()

    # 创建与获取YOLOV3模型
    model = YOLO_V3.models.main()

    # 定义优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=config_parameter.LEARNING_RATE, momentum=config_parameter.MOMENTUM, weight_decay=config_parameter.WEIGHT_DECAY)

    # 学习率调度器  多步学习率调度器   milestones=[50, 80]: 触发学习率调整的epoch位置   gamma=0.1: 学习率衰减的乘数因子
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.1)

    # 创建损失函数
    criterion = YOLO_V3.loss.YOLOv3Loss()

    # 记录模型图结构
    sampleImages, _ = next(iter(trainDataLoader))
    sampleImages = sampleImages.to(device)
    writer.add_graph(model, sampleImages)


    bestValLoss = float('inf')
    globalStep = 0
    # 训练循环
    for epoch in range(config_parameter.MAX_EPOCHS):
        model.train()
        trainTotalLoss = 0
        trainLoop = tqdm(trainDataLoader, desc="training")

        # 记录每个epoch的训练损失分量累计值
        epochCoordLoss = 0
        epochObjLoss = 0
        epochNoobjLoss = 0
        epochClassLoss = 0
        for batchIdx, (images, targets) in enumerate(trainLoop):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            totalLoss, lossComponents = criterion(predictions, targets)

            # 反向传播
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            trainTotalLoss += totalLoss.item()

            # 累计损失分量
            epochCoordLoss += lossComponents["coordLoss"]
            epochObjLoss += lossComponents["objLoss"]
            epochNoobjLoss += lossComponents["noObjLoss"]
            epochClassLoss += lossComponents["classLoss"]

            # 记录每个batch的训练损失到TensorBoard
            writer.add_scalar('Train/Batch_Total_Loss', totalLoss.item(), globalStep)
            writer.add_scalar('Train/Batch_Coord_Loss', lossComponents["coordLoss"], globalStep)
            writer.add_scalar('Train/Batch_Obj_Loss', lossComponents["objLoss"], globalStep)
            writer.add_scalar('Train/Batch_NoObj_Loss', lossComponents["noObjLoss"], globalStep)
            writer.add_scalar('Train/Batch_Class_Loss', lossComponents["classLoss"], globalStep)

            globalStep += 1  # 更新全局步数

            # 更新进度条
            trainLoop.set_postfix({
                'Total Loss': f'{totalLoss.item():.4f}',
                'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                'Obj Loss': f'{lossComponents["objLoss"]:.4f}',
                'NoObj Loss': f'{lossComponents["noObjLoss"]:.4f}',
                'Class Loss': f'{lossComponents["classLoss"]:.4f}'
            })

        # 记录每个epoch的平均训练损失
        avgTrainLoss = trainTotalLoss / len(trainDataLoader)
        writer.add_scalar('Train/Epoch_Total_Loss', avgTrainLoss, epoch)
        writer.add_scalar('Train/Epoch_Coord_Loss', epochCoordLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_Obj_Loss', epochObjLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_NoObj_Loss', epochNoobjLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_Class_Loss', epochClassLoss / len(trainDataLoader), epoch)

        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # 更新学习率
        scheduler.step()

        # 打印epoch统计信息
        print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgTrainLoss:.4f}')

        # 如果符合轮次要求就进行验证
        if epoch % 5 == 0:
            model.eval()
            valTotalLoss = 0
            valLoop = tqdm(valDataLoader, desc="valing")

            valCoordLoss = 0
            valObjLoss = 0
            valNoobjLoss = 0
            valClassLoss = 0
            for batchIdx, (images, targets) in enumerate(valLoop):
                images = images.to(device)
                targets = targets.to(device)

                # 前向传播
                predictions = model(images)

                # 计算损失
                totalLoss, lossComponents = criterion(predictions, targets)

                valTotalLoss += totalLoss.item()

                valCoordLoss += lossComponents["coordLoss"]
                valObjLoss += lossComponents["objLoss"]
                valNoobjLoss += lossComponents["noObjLoss"]
                valClassLoss += lossComponents["classLoss"]

                # 更新进度条
                valLoop.set_postfix({
                    'Total Loss': f'{totalLoss.item():.4f}',
                    'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                    'Obj Loss': f'{lossComponents["objLoss"]:.4f}',
                    'NoObj Loss': f'{lossComponents["noObjLoss"]:.4f}',
                    'Class Loss': f'{lossComponents["classLoss"]:.4f}'
                })

            # 打印epoch统计信息
            # 记录验证损失到TensorBoard - 新增
            avgValLoss = valTotalLoss / len(valDataLoader)
            writer.add_scalar('Val/Epoch_Total_Loss', avgValLoss, epoch)
            writer.add_scalar('Val/Epoch_Coord_Loss', valCoordLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Obj_Loss', valObjLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_NoObj_Loss', valNoobjLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Class_Loss', valClassLoss / len(valDataLoader), epoch)
            print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgValLoss:.4f}')

            if avgValLoss <= bestValLoss:
                bestValLoss = avgValLoss
                torch.save(model.state_dict(), os.path.join(saveModelPath, "YOLO_V3_low_loss.pth"))
                print(f"模型已保存至 {saveModelPath}")


    # 关闭TensorBoard writer
    writer.close()

    # 清除缓存
    del model
    gc.collect()
    torch.cuda.empty_cache()





def YOLOV4_VOC2007_TRAIN_MAIN():
    # 创建模型存储的路径
    nowTime = datetime.now()
    saveModelPath = os.path.join('../YOLO_results/YOLO_V4', f'{nowTime.year}_{nowTime.day}_{nowTime.hour}_{nowTime.minute}_{nowTime.second}')
    os.makedirs(saveModelPath, exist_ok=True)

    # 创建TensorBoard日治目录
    # 本地查看运行：tensorboard --logdir=YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    # 然后打开浏览器，访问 http://localhost:6006（默认端口）
    # 网站服务器查看时，请运行 tensorboard dev upload --logdir YOLO_results/YOLO_V3/你的训练文件夹/tensorboard_logs
    logDir = os.path.join(saveModelPath, 'YOLOV4_tensorboard_logs')
    os.makedirs(logDir, exist_ok=True)
    writer = SummaryWriter(log_dir=logDir)

    # 创建与获取数据集
    trainDataLoader, valDataLoader = datasets.VOC2007_MAIN()

    # 创建与获取YOLOV3模型
    model = YOLO_V4.models.main()

    # 定义优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=config_parameter.LEARNING_RATE, momentum=config_parameter.MOMENTUM, weight_decay=config_parameter.WEIGHT_DECAY)

    # 学习率调度器  多步学习率调度器   milestones=[50, 80]: 触发学习率调整的epoch位置   gamma=0.1: 学习率衰减的乘数因子
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.1)

    # 创建损失函数
    criterion = YOLO_V4.loss.main()

    # 记录模型图结构
    sampleImages, _ = next(iter(trainDataLoader))
    sampleImages = sampleImages.to(device)
    writer.add_graph(model, sampleImages)


    bestValLoss = float('inf')
    globalStep = 0
    # 训练循环
    for epoch in range(config_parameter.MAX_EPOCHS):
        model.train()
        trainTotalLoss = 0
        trainLoop = tqdm(trainDataLoader, desc="training")

        # 记录每个epoch的训练损失分量累计值
        epochCoordLoss = 0
        epochConfLoss = 0
        epochClassLoss = 0
        for batchIdx, (images, targets) in enumerate(trainLoop):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            totalLoss, lossComponents = criterion(predictions, targets)

            # 反向传播
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            trainTotalLoss += totalLoss.item()

            # 累计损失分量
            epochCoordLoss += lossComponents["coordLoss"]
            epochConfLoss += lossComponents['confLoss']
            epochClassLoss += lossComponents["classLoss"]

            # 记录每个batch的训练损失到TensorBoard
            writer.add_scalar('Train/Batch_Total_Loss', totalLoss.item(), globalStep)
            writer.add_scalar('Train/Batch_Coord_Loss', lossComponents["coordLoss"], globalStep)
            writer.add_scalar('Train/Batch_Conf_Loss', lossComponents['confLoss'], globalStep)
            writer.add_scalar('Train/Batch_Class_Loss', lossComponents["classLoss"], globalStep)

            globalStep += 1  # 更新全局步数

            # 更新进度条
            trainLoop.set_postfix({
                'Total Loss': f'{totalLoss.item():.4f}',
                'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                'Confidence Loss': f'{lossComponents["confLoss"]:.4f}',
                'Class Loss': f'{lossComponents["classLoss"]:.4f}'
            })

        # 记录每个epoch的平均训练损失
        avgTrainLoss = trainTotalLoss / len(trainDataLoader)
        writer.add_scalar('Train/Epoch_Total_Loss', avgTrainLoss, epoch)
        writer.add_scalar('Train/Epoch_Coord_Loss', epochCoordLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Batch_Conf_Loss', epochConfLoss / len(trainDataLoader), epoch)
        writer.add_scalar('Train/Epoch_Class_Loss', epochClassLoss / len(trainDataLoader), epoch)

        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # 更新学习率
        scheduler.step()

        # 打印epoch统计信息
        print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgTrainLoss:.4f}')

        # 如果符合轮次要求就进行验证
        if epoch % 5 == 0:
            model.eval()
            valTotalLoss = 0
            valLoop = tqdm(valDataLoader, desc="valing")

            valCoordLoss = 0
            valConfLoss = 0
            valClassLoss = 0
            for batchIdx, (images, targets) in enumerate(valLoop):
                images = images.to(device)
                targets = targets.to(device)

                # 前向传播
                predictions = model(images)

                # 计算损失
                totalLoss, lossComponents = criterion(predictions, targets)

                valTotalLoss += totalLoss.item()

                valCoordLoss += lossComponents["coordLoss"]
                valConfLoss += lossComponents["confLoss"]
                valClassLoss += lossComponents["classLoss"]

                # 更新进度条
                valLoop.set_postfix({
                    'Total Loss': f'{totalLoss.item():.4f}',
                    'Coord Loss': f'{lossComponents["coordLoss"]:.4f}',
                    'Confidence Loss': f'{lossComponents["confLoss"]:.4f}',
                    'Class Loss': f'{lossComponents["classLoss"]:.4f}'
                })

            # 打印epoch统计信息
            # 记录验证损失到TensorBoard - 新增
            avgValLoss = valTotalLoss / len(valDataLoader)
            writer.add_scalar('Val/Epoch_Total_Loss', avgValLoss, epoch)
            writer.add_scalar('Val/Epoch_Coord_Loss', valCoordLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Confidence_Loss', valConfLoss / len(valDataLoader), epoch)
            writer.add_scalar('Val/Epoch_Class_Loss', valClassLoss / len(valDataLoader), epoch)
            print(f'Epoch {epoch + 1}/{config_parameter.MAX_EPOCHS}, Average Loss: {avgValLoss:.4f}')

            if avgValLoss <= bestValLoss:
                bestValLoss = avgValLoss
                torch.save(model.state_dict(), os.path.join(saveModelPath, "YOLO_V4_low_loss.pth"))
                print(f"模型已保存至 {saveModelPath}")



    # 关闭TensorBoard writer
    writer.close()

    # 清除缓存
    del model
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == '__main__':
    # YOLOV3_VOC2007_TRAIN_MAIN()
    YOLOV1_VOC2007_TRAIN_MAIN()
    # YOLOV4_VOC2007_TRAIN_MAIN()





