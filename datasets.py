
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
import random
import math
import cv2
import numpy as np



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

        # 数据增强：随机裁剪
        croppedImage, croppedBoundingBoxes = self.random_crop_with_bboxes(imageData,
                                                                          boundingBoxes,
                                                                          config_parameter.RANDOM_CROP_RATIO,
                                                                          config_parameter.RANDOM_CROP_SCOPE[0],
                                                                          config_parameter.RANDOM_CROP_SCOPE[1])


        # 数据增强：随机仿射变换
        affineImage, affineBoundingBoxes = self.random_affine_transform(croppedImage,
                                                                        croppedBoundingBoxes,
                                                                        config_parameter.AFFINE_ROTATION,
                                                                        config_parameter.AFFINE_TRANSLATION,
                                                                        config_parameter.AFFINE_SCALE,
                                                                        config_parameter.AFFINE_SHEAR,
                                                                        config_parameter.AFFINE_FLIP,
                                                                        config_parameter.RANDOM_AFFINE_RATIO,
                                                                        config_parameter.AFFINE_BORDER)


        print(affineImage.size)
        # 可视化数据
        next_utils.visualize_image_with_bboxes(affineImage, affineBoundingBoxes)


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



    # 数据增强：随机裁剪
    def random_crop_with_bboxes(self, image, boundingBoxes, cropProb, minScale, maxScale):
        """
        对图像和对应的边界框进行随机裁剪

        参数:
        image: PIL Image对象
        bounding_boxes: 边界框列表，格式为 [(name, (xmin, ymin, xmax, ymax)), ...]
        crop_prob: 执行裁剪的概率 (0-1)
        min_scale: 最小裁剪比例 (0-1)
        max_scale: 最大裁剪比例 (0-1)

        返回:
        cropped_image: 裁剪后的PIL Image
        cropped_bboxes: 裁剪后对应的边界框列表
        """
        # 以一定概率决定是否执行裁剪
        if random.random() > cropProb:
            return image, boundingBoxes

        # 获取图像尺寸
        width, height = image.size

        # 随机确定裁剪比例
        scale = random.uniform(minScale, maxScale)

        # 计算裁剪区域的尺寸
        crop_width = int(width * scale)
        crop_height = int(height * scale)

        # 随机确定裁剪区域的起始位置
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))

        # 调整边界框坐标
        cropped_bboxes = []
        for class_name, bbox in boundingBoxes:
            xmin, ymin, xmax, ymax = bbox

            # 计算边界框与裁剪区域的交集
            inter_xmin = max(xmin, left)
            inter_ymin = max(ymin, top)
            inter_xmax = min(xmax, right)
            inter_ymax = min(ymax, bottom)

            # 检查边界框是否在裁剪区域内
            if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
                # 计算新的边界框坐标（相对于裁剪后的图像）
                new_xmin = inter_xmin - left
                new_ymin = inter_ymin - top
                new_xmax = inter_xmax - left
                new_ymax = inter_ymax - top

                # 确保新的边界框坐标在有效范围内
                new_xmin = max(0, min(new_xmin, crop_width))
                new_ymin = max(0, min(new_ymin, crop_height))
                new_xmax = max(0, min(new_xmax, crop_width))
                new_ymax = max(0, min(new_ymax, crop_height))

                cropped_bboxes.append((class_name, (new_xmin, new_ymin, new_xmax, new_ymax)))

        return cropped_image, cropped_bboxes


    # 数据增强：随机仿射变换
    def random_affine_transform(self, image, boundingBoxes, rotationRange, translationRange, scaleRange, shearRange,flipProb, transformProb, borderValue):

        """
        使用OpenCV进行更精确的仿射变换

        参数:
        image: PIL Image对象
        bounding_boxes: 边界框列表
        rotation_range: 旋转角度范围 (度)
        translation_range: 平移范围 (相对于图像尺寸的比例)
        scale_range: 缩放范围
        shear_range: 剪切角度范围 (度)
        flip_prob: 水平翻转的概率
        transform_prob: 执行变换的概率
        border_value: 边界填充值

        返回:
        transformed_image: 变换后的PIL Image
        transformed_bboxes: 变换后对应的边界框列表
        """
        # 以一定概率决定是否执行变换
        if random.random() > transformProb:
            return image, boundingBoxes

        # 获取图像尺寸
        width, height = image.size

        # 随机生成变换参数
        rotation = random.uniform(rotationRange[0], rotationRange[1]) if rotationRange else 0
        translationX = random.uniform(translationRange[0], translationRange[1]) * width if translationRange else 0
        translationY = random.uniform(translationRange[0], translationRange[1]) * height if translationRange else 0
        scale = random.uniform(scaleRange[0], scaleRange[1]) if scaleRange else 1.0
        shearX = math.radians(random.uniform(shearRange[0], shearRange[1])) if shearRange else 0
        shearY = math.radians(random.uniform(shearRange[0], shearRange[1])) if shearRange else 0
        flip = random.random() < flipProb if flipProb else False

        # 将PIL图像转换为OpenCV格式
        cvImage = np.array(image)
        cvImage = cv2.cvtColor(cvImage, cv2.COLOR_RGB2BGR)

        # 水平翻转
        if flip:
            cv_image = cv2.flip(cvImage, 1)

        # 计算变换矩阵
        center = (width / 2, height / 2)

        # 构建旋转矩阵
        rotationMatrix = cv2.getRotationMatrix2D(center, rotation, scale)

        # 添加剪切变换
        shearMatrix = np.array([
            [1, math.tan(shearX), 0],
            [math.tan(shearY), 1, 0]
        ], dtype=np.float32)

        # 组合变换矩阵
        affineMatrix = np.dot(rotationMatrix, shearMatrix)

        # 添加平移
        affineMatrix[0, 2] += translationX
        affineMatrix[1, 2] += translationY

        # 计算变换后的图像尺寸
        cosTheta = abs(affineMatrix[0, 0])
        sinTheta = abs(affineMatrix[0, 1])

        newWidth = int(height * sinTheta + width * cosTheta)
        newHeight = int(height * cosTheta + width * sinTheta)

        # 调整变换矩阵，使变换后的图像居中
        affineMatrix[0, 2] += (newWidth - width) / 2
        affineMatrix[1, 2] += (newHeight - height) / 2

        # 应用仿射变换
        transformedCvImage = cv2.warpAffine(
            cvImage, affineMatrix, (newWidth, newHeight),
            flags=cv2.INTER_LINEAR, borderValue=borderValue
        )

        # 将OpenCV图像转换回PIL格式
        transformedImage = Image.fromarray(
            cv2.cvtColor(transformedCvImage, cv2.COLOR_BGR2RGB)
        )

        # 应用变换到边界框
        transformedBboxes = []
        for className, bbox in boundingBoxes:
            xmin, ymin, xmax, ymax = bbox

            # 水平翻转
            if flip:
                xmin, xmax = width - xmax, width - xmin

            # 边界框的四个角点
            corners = np.array([
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmax, ymax, 1],
                [xmin, ymax, 1]
            ], dtype=np.float32)

            # 应用变换到角点
            transformedCorners = np.dot(affineMatrix, corners.T).T

            # 计算变换后的边界框
            newXmin = np.min(transformedCorners[:, 0])
            newYmin = np.min(transformedCorners[:, 1])
            newXmax = np.max(transformedCorners[:, 0])
            newYmax = np.max(transformedCorners[:, 1])

            # 确保边界框坐标在有效范围内
            newXmin = max(0, min(newXmin, newWidth))
            newYmin = max(0, min(newYmin, newHeight))
            newXmax = max(0, min(newXmax, newWidth))
            newYmax = max(0, min(newYmax, newHeight))

            # 确保边界框有合理的尺寸
            if newXmax - newXmin > 1 and newYmax - newYmin > 1:
                transformedBboxes.append(
                    (className, (int(newXmin), int(newYmin), int(newXmax), int(newYmax)))
                )

        return transformedImage, transformedBboxes







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






