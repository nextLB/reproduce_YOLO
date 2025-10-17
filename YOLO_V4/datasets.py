
"""
    YOLOV4 数据集的相关配置
"""

# 2025.10.17 (V1.0)            --- by next, 初步实现了YOLOv4的数据集配置和加载文件


# import os.path
# import torch
# import config_path
# import config_parameters
# import utils
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset
# from tqdm import tqdm
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torchvision.transforms as T
# from PIL import Image
# import xml.etree.ElementTree as ET
#
#
#
# baseTransformer = T.Compose([
#     T.Resize(config_parameters.IMAGE_SIZE),
#     T.ToTensor(),
# ])
#
#
#
# trainTransformer = T.Compose([
#     T.Resize(config_parameters.IMAGE_SIZE),
#     T.ToTensor(),
#     T.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])
#
#
# # VOC 2007相关数据获取可访问仓库 git@github.com:nextLB/VOC2007.git 进行拉取
# class YOLOPascalVocDataset(Dataset):
#     def __init__(self, setType, transform):
#         assert setType in {'train', 'test'}
#
#         # 加载所有类别标签
#         self.classes = utils.load_train_classes()
#         # 加载与处理数据集路径
#         self.imagesName = utils.load_train_images_name()
#         self.labelsName = utils.load_train_labels_name()
#
#         self.transform = transform
#
#
#
#     def __len__(self):
#         return len(self.imagesName)
#
#     def __getitem__(self, item):
#         imagePath = os.path.join(config_path.TRAIN_DATA_PATH, f'JPEGImages/{self.imagesName[item]}')
#         labelPath = os.path.join(config_path.TRAIN_DATA_PATH, f'Annotations/{self.labelsName[item]}')
#
#         # TODO: 加载图像数据，并进行数据增强
#         imageData = Image.open(imagePath).convert("RGB")
#         originalData = imageData.copy()
#         originalWidth, originalHeight = originalData.size
#
#         augmentationData = self.transform(imageData)
#         originalData = baseTransformer(originalData)
#
#
#
#         # TODO: 加载xml文件，并处理和生成一系列标签数据
#         tree = ET.parse(labelPath)
#         root = tree.getroot()
#
#         # 获取其图片尺寸
#         size = root.find('size')
#         xmlWidth = int(size.find('width').text)
#         xmlHeight = int(size.find('height').text)
#
#         # 验证图像尺寸一致性
#         if originalWidth != xmlWidth or originalHeight != xmlHeight:
#             print(f"Warning: Image size mismatch for {self.imagesName[item]}")
#
#         # 计算缩放比例
#         scaleX = config_parameters.IMAGE_SIZE[0] / originalWidth
#         scaleY = config_parameters.IMAGE_SIZE[1] / originalHeight
#
#         # 提取所有物体的边界框
#         boundingBoxes = []
#         resizedBoundingBoxes = []
#         for obj in root.iter('object'):
#             name = obj.find('name').text
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)
#             boundingBoxes.append((name, (xmin, ymin, xmax, ymax)))
#
#             resizedXMin = int(xmin * scaleX)
#             resizedXMax = int(xmax * scaleX)
#             resizedYMin = int(ymin * scaleY)
#             resizedYMax = int(ymax * scaleY)
#             resizedBoundingBoxes.append((name, (resizedXMin, resizedYMin, resizedXMax, resizedYMax)))
#
#
#         # utils.visualize_image_and_label(originalData, augmentationData, resizedBoundingBoxes)
#
#         # 初始化跟踪字典和ground truth张量
#         boundingBoxes = {}      # 跟踪每个网格单元格已分配的边界框数量
#         classNames = {}     # 跟踪每个网格单元格分配的类别
#         depth = 5 * config_parameters.B + config_parameters.C     # 张量深度：B个边界框×5个参数 + C个类别
#         groundTruth = torch.zeros((config_parameters.S, config_parameters.S, depth))
#
#         # 计算网格尺寸
#         gridSizeX = config_parameters.IMAGE_SIZE[0] / config_parameters.S  # 每个网格的宽度
#         gridSizeY = config_parameters.IMAGE_SIZE[1] / config_parameters.S  # 每个网格的高度
#
#
#
#         # 处理每个边界框，构建ground truth张量
#         for name, coords in resizedBoundingBoxes:
#             # 获取类别索引 - 添加错误处理
#             if name not in self.classes:
#                 print(f"Warning: Unrecognized class '{name}' in image {self.imagesName[item]}. Skipping this object.")
#                 continue
#
#             classIndex = self.classes[name]
#             xMin, yMin, xMax, yMax = coords
#
#             # 计算边界框中心点坐标
#             midX = (xMax + xMin) / 2
#             midY = (yMax + yMin) / 2
#
#             # 确定中心点所在的网格单元格
#             col = int(midX // gridSizeX)
#             row = int(midY // gridSizeY)
#
#             # 确保网格缩影在有效范围内
#             if 0 <= col < config_parameters.S and 0 <= row < config_parameters.S:
#                 cell = (row, col)
#
#
#                 # 如果该网格单元格未被分配类别，或者当前类别与已分配类别相同
#                 if cell not in classNames or name == classNames[cell]:
#                     # 创建类别one-hot编码向量
#                     oneHot = torch.zeros(config_parameters.C)
#                     oneHot[classIndex] = 1.0
#
#                     # 将类别信息写入ground truth张量的前C个通道
#                     groundTruth[row, col, :config_parameters.C] = oneHot
#                     classNames[cell] = name
#
#                     # 获取当前网格单元格已分配的边界框数量
#                     bboxIndex = boundingBoxes.get(cell, 0)
#
#                     # 如果还有可用的边界框槽位
#                     if bboxIndex < config_parameters.B:
#                         # 计算边界框相对于网格单元格的归一化坐标
#                         bboxTruth = (
#                             (midX - col * gridSizeX) / gridSizeX,  # X坐标相对于网格的偏移
#                             (midY - row * gridSizeY) / gridSizeY,  # Y坐标相对于网格的偏移
#                             (xMax - xMin) / config_parameters.IMAGE_SIZE[0],  # 宽度相对于图像的比率
#                             (yMax - yMin) / config_parameters.IMAGE_SIZE[1],  # 高度相对于图像的比率
#                             1.0  # 置信度（有目标）
#                         )
#
#                         # 计算当前边界框在张量中的起始位置
#                         bbox_start = config_parameters.C + 5 * bboxIndex
#
#                         # 将当前边界框信息写入ground truth张量
#                         groundTruth[row, col, bbox_start:bbox_start + 5] = torch.tensor(bboxTruth)
#
#                         # 更新该网格单元格的边界框计数
#                         boundingBoxes[cell] = bboxIndex + 1
#
#
#         return originalData, augmentationData, groundTruth
#
#
#
#
#
#
#
# def main():
#     fullDatasets = YOLOPascalVocDataset('train', baseTransformer)
#
#     # 划分索引，整理出训练集与验证集
#     indices = list(range(len(fullDatasets)))
#     trainIndices, valIndices = train_test_split(
#         indices,
#         test_size=config_parameters.RATIO,   # 作为验证集的比例
#         random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
#         shuffle=True        # 设置是否打乱
#     )
#
#     trainDatasets = Subset(
#         YOLOPascalVocDataset(
#             'train',
#             trainTransformer
#         ),
#         trainIndices
#     )
#
#     valDatasets = Subset(
#         YOLOPascalVocDataset(
#             'train',
#             baseTransformer
#         ),
#         valIndices
#     )
#
#     # 创建数据集加载器
#     trainDataLoader = DataLoader(
#         trainDatasets,
#         batch_size=config_parameters.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config_parameters.NUM_WORKERS,
#         drop_last=True  # 不保留最后一个不完整批次
#     )
#
#     valDataLoader = DataLoader(
#         valDatasets,
#         batch_size=config_parameters.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config_parameters.NUM_WORKERS,
#         drop_last=True  # 不保留最后一个不完整批次
#     )
#
#
#     # os.makedirs('visualization', exist_ok=True)
#     # # 可视化一下加载的数据集
#     # with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:
#     #     # 训练集
#     #     for batchIndex, (originalData, augmentationData, targets) in enumerate(trainDataLoader):
#     #         for i in range(config_parameters.BATCH_SIZE):
#     #             print(targets.shape)
#     #         pbarDataloader.update(1)
#
#     #     # 验证集
#     #     for batchIndex, (originalData, augmentationData, targets) in enumerate(valDataLoader):
#     #         for i in range(config_parameter.BATCH_SIZE):
#     #             print(targets.shape)
#     #         pbarDataloader.update(1)
#
#
#     return trainDataLoader, valDataLoader
#
#
# if __name__ == '__main__':
#     main()


"""
    YOLOV4 数据集的相关配置
"""

# 2025.10.17 (V1.0)            --- by next, 初步实现了YOLOv4的数据集配置和加载文件

import os.path
import torch
import config_path
import config_parameters
import utils
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import xml.etree.ElementTree as ET


class YOLOAugmentation:
    """YOLO数据增强类，同时处理图像和边界框"""

    def __init__(self, image_size, augmentation_prob=0.5):
        self.image_size = image_size
        self.augmentation_prob = augmentation_prob

    def __call__(self, image, bboxes):
        """
        应用数据增强
        Args:
            image: PIL Image
            bboxes: list of (class_name, (xmin, ymin, xmax, ymax))
        Returns:
            augmented_image: 增强后的图像
            augmented_bboxes: 增强后的边界框
        """
        # 以一定概率应用增强
        if random.random() > self.augmentation_prob:
            # 不应用增强，只调整大小
            image_resized = image.resize(self.image_size, Image.BILINEAR)
            return image_resized, bboxes

        # 随机选择增强方法
        aug_method = random.choice([
            'random_crop',
            'affine_transform',
            'color_adjust',
            'combination'
        ])

        if aug_method == 'random_crop':
            image_aug, bboxes_aug = self.random_crop(image, bboxes)
        elif aug_method == 'affine_transform':
            image_aug, bboxes_aug = self.affine_transform(image, bboxes)
        elif aug_method == 'color_adjust':
            image_aug, bboxes_aug = self.color_adjust(image, bboxes)
        else:  # combination
            # 组合多种增强
            image_temp, bboxes_temp = self.random_crop(image, bboxes)
            image_temp, bboxes_temp = self.affine_transform(image_temp, bboxes_temp)
            image_aug, bboxes_aug = self.color_adjust(image_temp, bboxes_temp)

        # 确保图像调整到目标尺寸
        if image_aug.size != self.image_size:
            image_aug = image_aug.resize(self.image_size, Image.BILINEAR)

            # 调整边界框到新尺寸
            orig_width, orig_height = image_aug.size
            scale_x = self.image_size[0] / orig_width
            scale_y = self.image_size[1] / orig_height

            scaled_bboxes = []
            for class_name, bbox in bboxes_aug:
                xmin, ymin, xmax, ymax = bbox
                scaled_bboxes.append((
                    class_name,
                    (int(xmin * scale_x), int(ymin * scale_y),
                     int(xmax * scale_x), int(ymax * scale_y))
                ))
            bboxes_aug = scaled_bboxes

        return image_aug, bboxes_aug

    def random_crop(self, image, bboxes):
        """随机裁剪"""
        width, height = image.size

        # 随机裁剪比例 (0.6 ~ 0.9)
        crop_ratio = random.uniform(0.6, 0.9)
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        # 确保裁剪尺寸合理
        crop_width = max(crop_width, 50)
        crop_height = max(crop_height, 50)

        # 随机裁剪位置
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))

        # 调整边界框
        new_bboxes = []
        for class_name, bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox

            # 计算裁剪后的新坐标
            new_xmin = max(xmin - left, 0)
            new_ymin = max(ymin - top, 0)
            new_xmax = min(xmax - left, crop_width)
            new_ymax = min(ymax - top, crop_height)

            # 检查边界框是否有效（面积大于0且在图像内）
            bbox_width = new_xmax - new_xmin
            bbox_height = new_ymax - new_ymin
            if bbox_width > 5 and bbox_height > 5:  # 确保边界框足够大
                new_bboxes.append((class_name, (new_xmin, new_ymin, new_xmax, new_ymax)))

        # 如果裁剪后没有有效边界框，返回原始图像和边界框
        if not new_bboxes:
            return image, bboxes

        return cropped_image, new_bboxes

    def affine_transform(self, image, bboxes):
        """仿射变换（旋转、平移、缩放）"""
        width, height = image.size

        # 随机旋转角度 (-15° ~ 15°)
        angle = random.uniform(-15, 15)

        # 随机缩放 (0.8 ~ 1.2)
        scale = random.uniform(0.8, 1.2)

        # 随机平移 (-0.1 ~ 0.1)
        translate_x = random.uniform(-0.1, 0.1) * width
        translate_y = random.uniform(-0.1, 0.1) * height

        # 应用仿射变换到图像 - 修复fillcolor问题
        try:
            # 尝试新版本API
            transformed_image = TF.affine(
                image,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=0,
                fill=(128, 128, 128)  # 使用fill参数
            )
        except TypeError:
            # 回退到旧版本API（不支持fill参数）
            transformed_image = TF.affine(
                image,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=0
            )

        # 计算变换矩阵
        rad = math.radians(angle)
        cos_angle = math.cos(rad)
        sin_angle = math.sin(rad)

        # 图像中心点
        center_x = width / 2
        center_y = height / 2

        new_bboxes = []
        for class_name, bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox

            # 计算边界框的四个角点
            corners = [
                (xmin, ymin),  # 左上
                (xmax, ymin),  # 右上
                (xmax, ymax),  # 右下
                (xmin, ymax)  # 左下
            ]

            # 变换每个角点
            transformed_corners = []
            for x, y in corners:
                # 相对于中心点
                rel_x = x - center_x
                rel_y = y - center_y

                # 旋转
                rotated_x = rel_x * cos_angle - rel_y * sin_angle
                rotated_y = rel_x * sin_angle + rel_y * cos_angle

                # 缩放
                scaled_x = rotated_x * scale
                scaled_y = rotated_y * scale

                # 平移
                translated_x = scaled_x + center_x + translate_x
                translated_y = scaled_y + center_y + translate_y

                transformed_corners.append((translated_x, translated_y))

            # 计算变换后的边界框
            xs = [point[0] for point in transformed_corners]
            ys = [point[1] for point in transformed_corners]

            new_xmin = max(0, min(xs))
            new_ymin = max(0, min(ys))
            new_xmax = min(width, max(xs))
            new_ymax = min(height, max(ys))

            # 检查边界框是否有效
            bbox_width = new_xmax - new_xmin
            bbox_height = new_ymax - new_ymin
            if bbox_width > 5 and bbox_height > 5:  # 确保边界框足够大
                new_bboxes.append((class_name, (new_xmin, new_ymin, new_xmax, new_ymax)))

        # 如果没有有效边界框，返回原始数据
        if not new_bboxes:
            return image, bboxes

        return transformed_image, new_bboxes

    def color_adjust(self, image, bboxes):
        """颜色调整（亮度、对比度、饱和度、色调）"""
        # 随机亮度调整
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)

        # 随机对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_contrast(image, contrast_factor)

        # 随机饱和度调整
        saturation_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_saturation(image, saturation_factor)

        # 随机色调调整（限制在较小范围）
        hue_factor = random.uniform(-0.05, 0.05)  # 减小范围避免颜色失真
        image = TF.adjust_hue(image, hue_factor)

        # 颜色调整不影响边界框坐标
        return image, bboxes


# 基础变换（用于验证集）
baseTransformer = T.Compose([
    T.Resize(config_parameters.IMAGE_SIZE),
    T.ToTensor(),
])


# 训练变换（包含数据增强）
class TrainTransformer:
    def __init__(self, image_size, augmentation_prob=0.7):
        self.image_size = image_size
        self.augmentation = YOLOAugmentation(image_size, augmentation_prob)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image, bboxes):
        # 应用数据增强
        augmented_image, augmented_bboxes = self.augmentation(image, bboxes)

        # 转换为Tensor
        tensor_image = TF.to_tensor(augmented_image)

        # 归一化
        tensor_image = self.normalize(tensor_image)

        return tensor_image, augmented_bboxes


# 创建训练变换实例
trainTransformer = TrainTransformer(config_parameters.IMAGE_SIZE, augmentation_prob=0.7)


# VOC 2007相关数据获取可访问仓库 git@github.com:nextLB/VOC2007.git 进行拉取
class YOLOPascalVocDataset(Dataset):
    def __init__(self, setType, transform, visualize=False):
        assert setType in {'train', 'test'}
        self.setType = setType
        self.visualize = visualize

        # 加载所有类别标签
        self.classes = utils.load_train_classes()
        # 加载与处理数据集路径
        self.imagesName = utils.load_train_images_name()
        self.labelsName = utils.load_train_labels_name()

        self.transform = transform

    def __len__(self):
        return len(self.imagesName)

    def __getitem__(self, item):
        imagePath = os.path.join(config_path.TRAIN_DATA_PATH, f'JPEGImages/{self.imagesName[item]}')
        labelPath = os.path.join(config_path.TRAIN_DATA_PATH, f'Annotations/{self.labelsName[item]}')

        # 加载图像数据
        imageData = Image.open(imagePath).convert("RGB")
        originalWidth, originalHeight = imageData.size

        # 加载XML文件并解析边界框
        tree = ET.parse(labelPath)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        xmlWidth = int(size.find('width').text)
        xmlHeight = int(size.find('height').text)

        # 验证图像尺寸一致性
        if originalWidth != xmlWidth or originalHeight != xmlHeight:
            print(f"Warning: Image size mismatch for {self.imagesName[item]}")

        # 提取所有物体的边界框（原始尺寸）
        originalBoundingBoxes = []
        for obj in root.iter('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            originalBoundingBoxes.append((name, (xmin, ymin, xmax, ymax)))

        # 创建原始数据的副本（用于可视化）
        originalData = imageData.copy()
        originalData = baseTransformer(originalData)

        # 应用变换（对于训练集，这会应用数据增强）
        if self.setType == 'train' and isinstance(self.transform, TrainTransformer):
            # 训练集：应用数据增强
            augmentationData, augmentedBoundingBoxes = self.transform(imageData, originalBoundingBoxes)
        else:
            # 验证集/测试集：只调整大小和转换
            augmentationData = imageData.resize(config_parameters.IMAGE_SIZE, Image.BILINEAR)
            augmentationData = TF.to_tensor(augmentationData)
            if hasattr(self.transform, 'normalize'):  # 如果是训练变换，还需要归一化
                augmentationData = self.transform.normalize(augmentationData)

            # 对于验证集，需要将边界框缩放到目标尺寸
            scaleX = config_parameters.IMAGE_SIZE[0] / originalWidth
            scaleY = config_parameters.IMAGE_SIZE[1] / originalHeight
            augmentedBoundingBoxes = []
            for name, bbox in originalBoundingBoxes:
                xmin, ymin, xmax, ymax = bbox
                resizedXMin = int(xmin * scaleX)
                resizedXMax = int(xmax * scaleX)
                resizedYMin = int(ymin * scaleY)
                resizedYMax = int(ymax * scaleY)
                augmentedBoundingBoxes.append((name, (resizedXMin, resizedYMin, resizedXMax, resizedYMax)))

        # 生成ground truth张量
        groundTruth = self._generate_ground_truth(augmentedBoundingBoxes)


        print(f"可视化样本: {self.imagesName[item]}")
        print(f"增强后边界框数量: {len(augmentedBoundingBoxes)}")
        utils.visualize_image_and_label(originalData, augmentationData, augmentedBoundingBoxes)

        return originalData, augmentationData, groundTruth

    def _generate_ground_truth(self, bounding_boxes):
        """根据边界框生成ground truth张量"""
        # 初始化跟踪字典和ground truth张量
        boundingBoxesCount = {}  # 跟踪每个网格单元格已分配的边界框数量
        classNames = {}  # 跟踪每个网格单元格分配的类别
        depth = 5 * config_parameters.B + config_parameters.C  # 张量深度：B个边界框×5个参数 + C个类别
        groundTruth = torch.zeros((config_parameters.S, config_parameters.S, depth))

        # 计算网格尺寸
        gridSizeX = config_parameters.IMAGE_SIZE[0] / config_parameters.S  # 每个网格的宽度
        gridSizeY = config_parameters.IMAGE_SIZE[1] / config_parameters.S  # 每个网格的高度

        # 处理每个边界框，构建ground truth张量
        for name, coords in bounding_boxes:
            # 获取类别索引 - 添加错误处理
            if name not in self.classes:
                print(f"Warning: Unrecognized class '{name}'. Skipping this object.")
                continue

            classIndex = self.classes[name]
            xMin, yMin, xMax, yMax = coords

            # 计算边界框中心点坐标
            midX = (xMax + xMin) / 2
            midY = (yMax + yMin) / 2

            # 确定中心点所在的网格单元格
            col = int(midX // gridSizeX)
            row = int(midY // gridSizeY)

            # 确保网格索引在有效范围内
            if 0 <= col < config_parameters.S and 0 <= row < config_parameters.S:
                cell = (row, col)

                # 如果该网格单元格未被分配类别，或者当前类别与已分配类别相同
                if cell not in classNames or name == classNames[cell]:
                    # 创建类别one-hot编码向量
                    oneHot = torch.zeros(config_parameters.C)
                    oneHot[classIndex] = 1.0

                    # 将类别信息写入ground truth张量的前C个通道
                    groundTruth[row, col, :config_parameters.C] = oneHot
                    classNames[cell] = name

                    # 获取当前网格单元格已分配的边界框数量
                    bboxIndex = boundingBoxesCount.get(cell, 0)

                    # 如果还有可用的边界框槽位
                    if bboxIndex < config_parameters.B:
                        # 计算边界框相对于网格单元格的归一化坐标
                        bboxTruth = (
                            (midX - col * gridSizeX) / gridSizeX,  # X坐标相对于网格的偏移
                            (midY - row * gridSizeY) / gridSizeY,  # Y坐标相对于网格的偏移
                            (xMax - xMin) / config_parameters.IMAGE_SIZE[0],  # 宽度相对于图像的比率
                            (yMax - yMin) / config_parameters.IMAGE_SIZE[1],  # 高度相对于图像的比率
                            1.0  # 置信度（有目标）
                        )

                        # 计算当前边界框在张量中的起始位置
                        bbox_start = config_parameters.C + 5 * bboxIndex

                        # 将当前边界框信息写入ground truth张量
                        groundTruth[row, col, bbox_start:bbox_start + 5] = torch.tensor(bboxTruth)

                        # 更新该网格单元格的边界框计数
                        boundingBoxesCount[cell] = bboxIndex + 1

        return groundTruth


def main():
    # 创建数据集实例 - 只在第一个样本上可视化
    train_datasets = YOLOPascalVocDataset('train', trainTransformer, visualize=True)
    val_datasets = YOLOPascalVocDataset('test', baseTransformer, visualize=False)  # 验证集不使用数据增强

    # 划分索引，整理出训练集与验证集
    indices = list(range(len(train_datasets)))
    trainIndices, valIndices = train_test_split(
        indices,
        test_size=config_parameters.RATIO,  # 作为验证集的比例
        random_state=42,  # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True  # 设置是否打乱
    )

    trainDatasets = Subset(train_datasets, trainIndices)
    valDatasets = Subset(val_datasets, valIndices)

    # 创建数据集加载器
    trainDataLoader = DataLoader(
        trainDatasets,
        batch_size=config_parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=config_parameters.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )

    valDataLoader = DataLoader(
        valDatasets,
        batch_size=config_parameters.BATCH_SIZE,
        shuffle=False,
        num_workers=config_parameters.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )

    # 可视化检查
    print(f"训练集大小: {len(trainDatasets)}")
    print(f"验证集大小: {len(valDatasets)}")

    # 测试一个批次
    for batch_idx, (originalData, augmentationData, targets) in enumerate(trainDataLoader):
        print(f"Batch {batch_idx}:")
        print(f"  Original data shape: {originalData.shape}")
        print(f"  Augmented data shape: {augmentationData.shape}")
        print(f"  Targets shape: {targets.shape}")

        if batch_idx >= 2:  # 只检查前几个批次
            break

    return trainDataLoader, valDataLoader


if __name__ == '__main__':
    main()

