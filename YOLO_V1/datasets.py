
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


        # utils.visualize_image_and_label(originalData, augmentationData, resizedBoundingBoxes)

        # 初始化跟踪字典和ground truth张量
        boundingBoxes = {}      # 跟踪每个网格单元格已分配的边界框数量
        classNames = {}     # 跟踪每个网格单元格分配的类别
        depth = 5 * config_parameter.B + config_parameter.C     # 张量深度：B个边界框×5个参数 + C个类别
        groundTruth = torch.zeros((config_parameter.S, config_parameter.S, depth))

        # 计算网格尺寸
        gridSizeX = config_parameter.IMAGE_SIZE[0] / config_parameter.S  # 每个网格的宽度
        gridSizeY = config_parameter.IMAGE_SIZE[1] / config_parameter.S  # 每个网格的高度



        # 处理每个边界框，构建ground truth张量
        for name, coords in resizedBoundingBoxes:
            # 获取类别索引 - 添加错误处理
            if name not in self.classes:
                print(f"Warning: Unrecognized class '{name}' in image {self.imagesName[item]}. Skipping this object.")
                continue

            classIndex = self.classes[name]
            xMin, yMin, xMax, yMax = coords

            # 计算边界框中心点坐标
            midX = (xMax + xMin) / 2
            midY = (yMax + yMin) / 2

            # 确定中心点所在的网格单元格
            col = int(midX // gridSizeX)
            row = int(midY // gridSizeY)

            # 确保网格缩影在有效范围内
            if 0 <= col < config_parameter.S and 0 <= row < config_parameter.S:
                cell = (row, col)


                # 如果该网格单元格未被分配类别，或者当前类别与已分配类别相同
                if cell not in classNames or name == classNames[cell]:
                    # 创建类别one-hot编码向量
                    oneHot = torch.zeros(config_parameter.C)
                    oneHot[classIndex] = 1.0

                    # 将类别信息写入ground truth张量的前C个通道
                    groundTruth[row, col, :config_parameter.C] = oneHot
                    classNames[cell] = name

                    # 获取当前网格单元格已分配的边界框数量
                    bboxIndex = boundingBoxes.get(cell, 0)

                    # 如果还有可用的边界框槽位
                    if bboxIndex < config_parameter.B:
                        # 计算边界框相对于网格单元格的归一化坐标
                        bboxTruth = (
                            (midX - col * gridSizeX) / gridSizeX,  # X坐标相对于网格的偏移
                            (midY - row * gridSizeY) / gridSizeY,  # Y坐标相对于网格的偏移
                            (xMax - xMin) / config_parameter.IMAGE_SIZE[0],  # 宽度相对于图像的比率
                            (yMax - yMin) / config_parameter.IMAGE_SIZE[1],  # 高度相对于图像的比率
                            1.0  # 置信度（有目标）
                        )

                        # 计算当前边界框在张量中的起始位置
                        bbox_start = config_parameter.C + 5 * bboxIndex

                        # 将当前边界框信息写入ground truth张量
                        groundTruth[row, col, bbox_start:bbox_start + 5] = torch.tensor(bboxTruth)

                        # 更新该网格单元格的边界框计数
                        boundingBoxes[cell] = bboxIndex + 1


        return originalData, augmentationData, groundTruth











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
        # 训练集
        for batchIndex, (originalData, augmentationData, targets) in enumerate(trainDataLoader):
            for i in range(config_parameter.BATCH_SIZE):
                print(targets.shape)
            pbarDataloader.update(1)

        # 验证集
        for batchIndex, (originalData, augmentationData, targets) in enumerate(valDataLoader):
            for i in range(config_parameter.BATCH_SIZE):
                print(targets.shape)
            pbarDataloader.update(1)


if __name__ == '__main__':
    main()






