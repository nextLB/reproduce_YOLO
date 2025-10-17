
"""
    YOLO数据集的处理与加载程序文件
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLO的数据集配置和加载文件
# 2025.10.17 (V1.1)            --- by next, 实现了数据增强的功能，并进一步提高了这个数据集加载文件的通用性



import torch
import config_path
import config_parameter
import next_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import xml.etree.ElementTree as ET



# VOC 2007相关数据获取可访问仓库 git@github.com:nextLB/VOC2007.git 进行拉取
class VOC2007Dataset(Dataset):
    def __init__(self):
        super(VOC2007Dataset, self).__init__()

        # 加载所有类别标签
        self.classes = next_utils.load_train_classes()
        # 加载与处理数据集路径
        self.imagesName = next_utils.load_train_images_name()
        self.labelsName = next_utils.load_train_labels_name()


    def __len__(self):
        return len(self.imagesName)

    def __getitem__(self, item):
        imagePath = os.path.join(config_path.TRAIN_DATA_PATH, f'JPEGImages/{self.imagesName[item]}')
        labelPath = os.path.join(config_path.TRAIN_DATA_PATH, f'Annotations/{self.labelsName[item]}')

        # 加载图像数据
        imageData = Image.open(imagePath).convert("RGB")
        originalData = imageData.copy()
        originalWidth, originalHeight = originalData.size

        # 加载真实目标框数据
        boundingBoxes = self.load_real_bboxes(labelPath)

        # # 可视化数据
        # next_utils.visualize_image_with_bboxes(imageData, boundingBoxes)





        return originalWidth, originalHeight

    # 加载真实目标框的坐标函数
    def load_real_bboxes(self, labelPath):
        tree = ET.parse(labelPath)
        root = tree.getroot()

        # 提取所有物体的边界框
        boundingBoxes = []
        for obj in root.iter('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boundingBoxes.append((name, (xmin, ymin, xmax, ymax)))

        return boundingBoxes


















def main():
    fullDatasets = VOC2007Dataset()

    # 划分索引，整理出训练集与验证集
    indices = list(range(len(fullDatasets)))
    trainIndices, valIndices = train_test_split(
        indices,
        test_size=config_parameter.RATIO,   # 作为验证集的比例
        random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True        # 设置是否打乱
    )

    trainDatasets = Subset(VOC2007Dataset(), trainIndices)
    valDatasets = Subset(VOC2007Dataset(), valIndices)

    # 创建数据集加载器
    trainDataLoader = DataLoader(
        trainDatasets,
        batch_size=config_parameter.BATCH_SIZE,
        shuffle=True,
        num_workers=config_parameter.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )

    valDataLoader = DataLoader(
        valDatasets,
        batch_size=config_parameter.BATCH_SIZE,
        shuffle=False,
        num_workers=config_parameter.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )


    os.makedirs('visualization', exist_ok=True)
    # 可视化一下加载的数据集
    with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:
        for batchIndex in enumerate(trainDataLoader):
            pbarDataloader.update(1)



    #     # 训练集
    #     for batchIndex, (originalData, augmentationData, targets) in enumerate(trainDataLoader):
    #         for i in range(config_parameter.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)
    #
    #     # 验证集
    #     for batchIndex, (originalData, augmentationData, targets) in enumerate(valDataLoader):
    #         for i in range(config_parameter.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)


    return trainDataLoader, valDataLoader


if __name__ == '__main__':
    main()






