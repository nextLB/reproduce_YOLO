
"""
    YOLOV1 数据集的相关配置
"""
import os.path

import torch

import config_path
import config_parameter
import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET



baseTransformer = T.Compose([
    T.Resize(config_parameter.IMAGE_SIZE),
    T.ToTensor(),
])



trainTransformer = T.Compose([
    T.Resize(config_parameter.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# VOC 2007相关数据获取可访问仓库 git@github.com:nextLB/VOC2007.git 进行拉取
class YOLOPascalVocDataset(Dataset):
    def __init__(self, setType):
        assert setType in {'train', 'test'}

        # 加载所有类别标签
        self.classes = utils.load_train_classes()
        # 加载与处理数据集路径
        self.imagesName = utils.load_train_images_name()
        self.labelsName = utils.load_train_labels_name()



    def __len__(self):
        return len(self.imagesName)

    def __getitem__(self, item):
        imagePath = os.path.join(config_path.TRAIN_DATA_PATH, f'JPEGImages/{self.imagesName[item]}')
        labelPath = os.path.join(config_path.TRAIN_DATA_PATH, f'Annotations/{self.labelsName[item]}')

        # TODO: 加载图像数据，并进行数据增强
        imageData = Image.open(imagePath).convert("RGB")
        originalData = imageData.copy()
        originalWidth, originalHeight = originalData.size

        augmentationData = trainTransformer(imageData)
        originalData = baseTransformer(originalData)



        # TODO: 加载xml文件，并处理和生成一系列标签数据
        tree = ET.parse(labelPath)
        root = tree.getroot()

        # 获取其图片尺寸
        size = root.find('size')
        xmlWidth = int(size.find('width').text)
        xmlHeight = int(size.find('height').text)

        # 验证图像尺寸一致性
        if originalWidth != xmlWidth or originalHeight != xmlHeight:
            print(f"Warning: Image size mismatch for {self.imagesName[item]}")

        # 计算缩放比例
        scaleX = config_parameter.IMAGE_SIZE[0] / originalWidth
        scaleY = config_parameter.IMAGE_SIZE[1] / originalHeight

        # 提取所有物体的边界框
        boundingBoxes = []
        resizedBoundingBoxes = []
        for obj in root.iter('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boundingBoxes.append((name, (xmin, ymin, xmax, ymax)))

            resizedXMin = int(xmin * scaleX)
            resizedXMax = int(xmax * scaleX)
            resizedYMin = int(ymin * scaleY)
            resizedYMax = int(ymax * scaleY)
            resizedBoundingBoxes.append((name, (resizedXMin, resizedYMin, resizedXMax, resizedYMax)))


        utils.visualize_image_and_label(originalData, augmentationData, resizedBoundingBoxes)

        return originalData, augmentationData











def main():
    fullDatasets = YOLOPascalVocDataset('train')
    classes = fullDatasets.classes  # 获取类别列表

    # 划分索引，整理出训练集与验证集
    indices = list(range(len(fullDatasets)))
    trainIndices, valIndices = train_test_split(
        indices,
        test_size=config_parameter.RATIO,   # 作为验证集的比例
        random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True        # 设置是否打乱
    )

    trainDatasets = Subset(
        YOLOPascalVocDataset(
            'train',
        ),
        trainIndices
    )

    valDatasets = Subset(
        YOLOPascalVocDataset(
            'train',
        ),
        valIndices
    )

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
        for batchIndex, (originalData, augmentationData) in enumerate(trainDataLoader):
            for i in range(config_parameter.BATCH_SIZE):
                print('i')
            pbarDataloader.update(1)


if __name__ == '__main__':
    main()






