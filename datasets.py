
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
import torchvision.transforms as transforms



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

        # 数据增强：随机颜色变换
        coloredImage, coloredBoundingBoxes = self.random_color_augmentation(affineImage, affineBoundingBoxes, config_parameter.RANDOM_COLOR_RATIO)

        # 调整图像尺寸
        augmentedImage, augmentedBoundingBoxes = self.resize_image_and_bboxes(coloredImage, coloredBoundingBoxes, config_parameter.IMAGE_SIZE)


        # # 可视化数据
        # next_utils.visualize_image_with_bboxes(augmentedImage, augmentedBoundingBoxes)

        # 对增强后的图像数据做最终的归一化处理
        augmentedImage = self.normalize_image(augmentedImage, config_parameter.MEAN, config_parameter.STD)

        # 将图像转换为tensor张量
        toTensorTansform = transforms.Compose([transforms.ToTensor()])
        augmentedImage = toTensorTansform(augmentedImage)
        augmentedImage = augmentedImage.float()


        # 初始化最终返回的ground truth张量
        depth = 5 + config_parameter.CLASS_NUMBER       # 张量深度：5个参数 + C个类别
        groudTruth = torch.zeros(config_parameter.TARGETS_SIZE, depth)

        index = 0
        # 处理每个边界框，构建ground truth张量
        for name, coords in augmentedBoundingBoxes:
            # 获取类别索引 - 添加错误处理
            if name not in self.classes:
                print(f"Warning: Unrecognized class '{name}' in image {self.imagesName[item]}. Skipping this object.")
                continue

            classIndex = self.classes[name]
            xMin, yMin, xMax, yMax = coords

            # 计算边界框中心点坐标
            midX = (xMax + xMin) / 2
            midY = (yMax + yMin) / 2

            # 计算边界框的宽和高
            width = xMax - xMin
            height = yMax - yMin

            # 创建类别one-hot编码向量
            oneHot = torch.zeros(config_parameter.CLASS_NUMBER)
            oneHot[classIndex] = 1.0

            # 将上述计算得到的所有信息写入到groundTruth张量中
            groudTruth[index, 0] = midX     # x的中心坐标
            groudTruth[index, 1] = midY     # y的中心坐标
            groudTruth[index, 2] = width    # 宽度
            groudTruth[index, 3] = height   # 高度
            groudTruth[index, 4] = 1.0      # 置信度
            groudTruth[index, 5:] = oneHot  # 类别的onehot向量

            index += 1


        return augmentedImage, groudTruth



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
        cropWidth = int(width * scale)
        cropHeight = int(height * scale)

        # 随机确定裁剪区域的起始位置
        left = random.randint(0, width - cropWidth)
        top = random.randint(0, height - cropHeight)
        right = left + cropWidth
        bottom = top + cropHeight

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
                new_xmin = max(0, min(new_xmin, cropWidth))
                new_ymin = max(0, min(new_ymin, cropHeight))
                new_xmax = max(0, min(new_xmax, cropWidth))
                new_ymax = max(0, min(new_ymax, cropHeight))

                cropped_bboxes.append((class_name, (new_xmin, new_ymin, new_xmax, new_ymax)))

        return cropped_image, cropped_bboxes



    # 数据增强：随机仿射变换
    def random_affine_transform(self, image, boundingBoxes, rotationRange, translationRange, scaleRange, shearRange,
                              flipProb, transformProb, borderValue):
        """
        使用OpenCV进行更精确的仿射变换

        参数:
        image: PIL Image对象
        boundingBoxes: 边界框列表
        rotationRange: 旋转角度范围 (度)
        translationRange: 平移范围 (相对于图像尺寸的比例)
        scaleRange: 缩放范围
        shearRange: 剪切角度范围 (度)
        flipProb: 水平翻转的概率
        transformProb: 执行变换的概率
        borderValue: 边界填充值

        返回:
        transformedImage: 变换后的PIL Image
        transformedBboxes: 变换后对应的边界框列表
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
            cvImage = cv2.flip(cvImage, 1)

        # 计算变换矩阵
        center = (width / 2, height / 2)

        # 构建旋转矩阵
        rotationMatrix = cv2.getRotationMatrix2D(center, rotation, scale)

        # 修正矩阵乘法问题：使用齐次坐标
        # 将旋转矩阵转换为3x3齐次坐标矩阵
        rotationMatrixHomo = np.vstack([rotationMatrix, [0, 0, 1]])

        # 构建剪切矩阵（3x3齐次坐标）
        shearMatrixHomo = np.array([
            [1, math.tan(shearX), 0],
            [math.tan(shearY), 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # 组合变换矩阵（先旋转缩放，再剪切）
        affineMatrixHomo = np.dot(shearMatrixHomo, rotationMatrixHomo)

        # 添加平移
        affineMatrixHomo[0, 2] += translationX
        affineMatrixHomo[1, 2] += translationY

        # 取前两行作为仿射变换矩阵
        affineMatrix = affineMatrixHomo[:2, :]

        # 计算变换后的图像尺寸
        # 计算原始图像的四个角点
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ], dtype=np.float32).T

        # 计算变换后的角点
        transformedCorners = np.dot(affineMatrix, corners)

        # 计算新图像的尺寸
        newWidth = int(np.max(transformedCorners[0]) - np.min(transformedCorners[0]))
        newHeight = int(np.max(transformedCorners[1]) - np.min(transformedCorners[1]))

        # 调整变换矩阵，使变换后的图像在正坐标区域
        affineMatrix[0, 2] -= np.min(transformedCorners[0])
        affineMatrix[1, 2] -= np.min(transformedCorners[1])

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
            bboxCorners = np.array([
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmax, ymax, 1],
                [xmin, ymax, 1]
            ], dtype=np.float32).T

            # 应用变换到角点
            transformedBboxCorners = np.dot(affineMatrix, bboxCorners)

            # 计算变换后的边界框
            newXmin = np.min(transformedBboxCorners[0])
            newYmin = np.min(transformedBboxCorners[1])
            newXmax = np.max(transformedBboxCorners[0])
            newYmax = np.max(transformedBboxCorners[1])

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


    # 数据增强：随机颜色变换
    def random_color_augmentation(self, image, boundingBoxes, augmentationProb):
        """
        对图像进行随机颜色增强，不影响边界框标签

        参数:
        image: PIL Image对象
        boundingBoxes: 边界框列表，格式为 [(name, (xmin, ymin, xmax, ymax)), ...]
        augmentationProb: 执行颜色增强的概率 (0-1)

        返回:
        augmentedImage: 颜色增强后的PIL Image
        boundingBoxes: 不变的边界框列表
        """
        # 以一定概率决定是否执行颜色增强
        if random.random() > augmentationProb:
            return image, boundingBoxes

        # 将PIL图像转换为numpy数组以便使用OpenCV
        cvImage = np.array(image)

        # 随机选择一种或多种颜色增强方式
        augmentationType = random.choice(['brightness', 'contrast', 'saturation', 'hue', 'multiple'])

        if augmentationType == 'brightness':
            # 随机亮度调整
            brightnessFactor = random.uniform(0.7, 1.3)
            hsv = cv2.cvtColor(cvImage, cv2.COLOR_RGB2HSV)
            # 转换为float32避免溢出
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] * brightnessFactor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            augmentedCvImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        elif augmentationType == 'contrast':
            # 随机对比度调整
            contrastFactor = random.uniform(0.7, 1.3)
            # 转换为float32进行计算
            floatImage = cvImage.astype(np.float32)
            floatImage = floatImage * contrastFactor
            floatImage = np.clip(floatImage, 0, 255)
            augmentedCvImage = floatImage.astype(np.uint8)

        elif augmentationType == 'saturation':
            # 随机饱和度调整
            saturationFactor = random.uniform(0.7, 1.3)
            hsv = cv2.cvtColor(cvImage, cv2.COLOR_RGB2HSV)
            # 转换为float32避免溢出
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturationFactor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            augmentedCvImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        elif augmentationType == 'hue':
            # 随机色调调整
            hueShift = random.randint(-10, 10)
            hsv = cv2.cvtColor(cvImage, cv2.COLOR_RGB2HSV)
            # 转换为float32避免溢出问题
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hueShift) % 180
            hsv = hsv.astype(np.uint8)
            augmentedCvImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        else:  # 'multiple'
            # 组合多种增强方式
            augmentedCvImage = cvImage.copy().astype(np.float32)

            # 亮度调整
            if random.random() > 0.5:
                brightnessFactor = random.uniform(0.8, 1.2)
                hsv = cv2.cvtColor(augmentedCvImage.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 2] = hsv[:, :, 2] * brightnessFactor
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                augmentedCvImage = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            # 对比度调整
            if random.random() > 0.5:
                contrastFactor = random.uniform(0.8, 1.2)
                augmentedCvImage = augmentedCvImage * contrastFactor

            # 饱和度调整
            if random.random() > 0.5:
                saturationFactor = random.uniform(0.8, 1.2)
                hsv = cv2.cvtColor(augmentedCvImage.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * saturationFactor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                augmentedCvImage = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            # 确保像素值在有效范围内
            augmentedCvImage = np.clip(augmentedCvImage, 0, 255).astype(np.uint8)

        # 确保像素值在有效范围内
        if augmentedCvImage.dtype != np.uint8:
            augmentedCvImage = np.clip(augmentedCvImage, 0, 255).astype(np.uint8)

        # 转换回PIL图像格式
        augmentedImage = Image.fromarray(augmentedCvImage)

        # 边界框保持不变
        return augmentedImage, boundingBoxes

    def resize_image_and_bboxes(self, image, boundingBoxes, targetSize):
        """
        将图像和边界框调整到指定尺寸

        参数:
        image: PIL Image对象
        boundingBoxes: 边界框列表，格式为 [(name, (xmin, ymin, xmax, ymax)), ...]
        targetSize: 目标尺寸，格式为 (width, height)

        返回:
        resizedImage: 调整尺寸后的PIL Image
        resizedBboxes: 调整后的边界框列表，格式与输入相同
        """
        # 获取原始图像尺寸
        originalWidth, originalHeight = image.size
        targetWidth, targetHeight = targetSize

        # 计算宽高缩放比例
        scaleX = targetWidth / originalWidth
        scaleY = targetHeight / originalHeight

        # 调整图像尺寸
        resizedImage = image.resize((targetWidth, targetHeight), Image.BILINEAR)

        # 调整边界框坐标
        resizedBboxes = []
        for className, bbox in boundingBoxes:
            xmin, ymin, xmax, ymax = bbox

            # 根据缩放比例调整边界框坐标
            newXmin = int(xmin * scaleX)
            newYmin = int(ymin * scaleY)
            newXmax = int(xmax * scaleX)
            newYmax = int(ymax * scaleY)

            # 确保边界框坐标在有效范围内
            newXmin = max(0, min(newXmin, targetWidth))
            newYmin = max(0, min(newYmin, targetHeight))
            newXmax = max(0, min(newXmax, targetWidth))
            newYmax = max(0, min(newYmax, targetHeight))

            # 确保边界框有合理的尺寸
            if newXmax - newXmin > 1 and newYmax - newYmin > 1:
                resizedBboxes.append((className, (newXmin, newYmin, newXmax, newYmax)))

        return resizedImage, resizedBboxes


    # 对我的图像执行归一化方法
    def normalize_image(self, image, mean, std):
        """
        使用均值和标准差对图像进行归一化（与PyTorch T.Normalize一致）

        参数:
        image: PIL Image对象或numpy数组 (H, W, C) 或 (C, H, W)
        mean: 各通道的均值
        std: 各通道的标准差

        返回:
        normalizedImage: 归一化后的图像，格式与输入一致
        """
        # 如果输入是PIL图像，转换为numpy数组
        if isinstance(image, Image.Image):
            imageArray = np.array(image).astype(np.float32)
            isPil = True
            # PIL图像通常是 (H, W, C)，需要转换为 (C, H, W) 进行归一化
            if len(imageArray.shape) == 3:
                imageArray = imageArray.transpose(2, 0, 1)
        else:
            imageArray = image.astype(np.float32)
            isPil = False

        # 确保imageArray是 (C, H, W) 格式
        if len(imageArray.shape) == 3 and imageArray.shape[0] != 3:
            # 如果是 (H, W, C) 格式，转换为 (C, H, W)
            imageArray = imageArray.transpose(2, 0, 1)

        # 将均值和标准差转换为numpy数组
        meanArray = np.array(mean).reshape(-1, 1, 1)
        stdArray = np.array(std).reshape(-1, 1, 1)

        # 执行归一化：(image - mean) / std
        normalizedArray = (imageArray - meanArray * 255) / (stdArray * 255)

        # 根据输入类型返回相应格式
        if isPil:
            # 对于PIL图像，我们需要转换回 (H, W, C) 格式
            normalizedArray = normalizedArray.transpose(1, 2, 0)
            # 归一化后的值可能不在[0,255]范围内，所以不能直接转PIL
            # 返回numpy数组，或者进行反归一化用于显示
            return normalizedArray
        else:
            return normalizedArray




def VOC2007_MAIN():
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


    # os.makedirs('visualization', exist_ok=True)
    # # 可视化一下加载的数据集
    # with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:
    #     for batchIndex, (augmentedImage, targets) in enumerate(trainDataLoader):
    #             pbarDataloader.update(1)
    #     # 训练集
    #     for batchIndex, (augmentedImage, targets) in enumerate(trainDataLoader):
    #         for i in range(config_parameter.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)
    #
    #     # 验证集
    #     for batchIndex, (augmentedImage, targets) in enumerate(valDataLoader):
    #         for i in range(config_parameter.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)


    return trainDataLoader, valDataLoader


if __name__ == '__main__':
    VOC2007_MAIN()






