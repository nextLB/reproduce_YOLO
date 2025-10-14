"""
    Yolov3 的数据集构建程序文件
"""
import os.path

import cv2
import torch
import math
import config_paramters
import config_path
import utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



# 数据加载和预处理
class COCODataset(Dataset):
    def __init__(self, datasetPath, imageSize):

        self.datasetPath = datasetPath
        self.imageSize = imageSize
        self.classNames = utils.load_class_names_from_yaml(config_path.DATASETS_YAML)

        # 用于存储加载图像和标签的路径
        self.images = []
        self.labels = []

        imagesDir = os.path.join(datasetPath, 'images/train2017')
        labelsDir = os.path.join(datasetPath, 'labels/train2017')

        for imageFile in os.listdir(imagesDir):
            if imageFile.endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(imagesDir, imageFile)
                labelPath = os.path.join(labelsDir, os.path.splitext(imageFile)[0] + '.txt')
                if os.path.exists(labelPath):
                    self.images.append(imagePath)
                    self.labels.append(labelPath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):


        if random.random() < config_paramters.RANDOM_LOAD_IMAGE_RATIO:
            # 以mosaic数据增强的形式加载图像
            image, targets = self.load_image_mosaic(idx)
        else:
            # 以普通形式加载图像
            image, targets = self.load_image_ordinary(idx)


        return image, targets



    # 以普通形式加载图像
    def load_image_ordinary(self, idx):

        # 加载图像
        imgPath = self.images[idx]
        image = Image.open(imgPath).convert('RGB')
        originalSize = image.size  # (width, height)

        # 加载标签
        labelPath = self.labels[idx]
        boxes = []
        with open(labelPath, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([class_id, x_center, y_center, width, height])

        boxes = np.array(boxes) if boxes else np.zeros((0, 5))



        # 数据增强
        image, boxes = self.random_affine(image, boxes, config_paramters.DEGREES, config_paramters.TRANSLATE, config_paramters.SCALE, config_paramters.SHEAR)

        # 调整图像大小
        image, boxes = self.resize(image, boxes, originalSize)

        # 颜色增强
        image = self.random_color(image)


        # # 可视化单个图像与类别
        # utils.visualize_single_image(
        #     image, boxes,
        #     class_names=self.classNames
        # )



        # 将处理过的数据转换为tensor
        imageTensor = transforms.ToTensor()(image)

        # 填充
        paddedImage = torch.zeros(3, self.imageSize, self.imageSize)
        _, h, w = imageTensor.shape
        paddedImage[:, :h, :w] = imageTensor

        targets = torch.zeros((config_paramters.TARGETS_SIZE, 6))
        if len(boxes) > 0:
            targets[:len(boxes), 1:] = torch.from_numpy(boxes)
            targets[:, 0] = idx

        return paddedImage, targets



    # 以mosaic数据增强的形式加载图像
    def load_image_mosaic(self, idx):
        # Mosaic数据增强
        indices = [idx] + [random.randint(0, len(self.images)-1) for _ in range(3)]

        random.shuffle(indices)

        # mosaicImage的尺寸是 (self.imageSize*2, self.imageSize*2, 3)
        mosaicImage = np.full((self.imageSize*2, self.imageSize*2, 3), 114, dtype=np.uint8)
        mosaicBoxes = []

        for i, index in enumerate(indices):
            # 加载图片
            imagePath = self.images[index]
            image = Image.open(imagePath).convert('RGB')

            # 加载标签
            labelPath = self.labels[index]
            boxes = []
            with open(labelPath, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])

            boxes = np.array(boxes) if boxes else np.zeros((0, 5))


            # 先调整图像与标签尺寸为指定大小
            originalSize = image.size  # (width, height)
            newImage, newBoxes = self.resize(image, boxes, originalSize)

            # # 可视化单个图像与类别
            # utils.visualize_single_image(
            #     newImage, newBoxes,
            #     # class_names=['class0', 'class1', 'class2'],  # 替换为你的类别
            #     title=f"Original Image {i + 1}"
            # )

            newImage = np.array(newImage)
            h, w = newImage.shape[:2]

            # 放置位置
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = 0, 0, w, h
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = w, 0, 2*w, h
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = 0, h, w, 2*h
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = w, h, 2*w, 2*h
                x1b, y1b, x2b, y2b = 0, 0, w, h

            mosaicImage[y1a:y2a, x1a:x2a] = newImage[y1b:y2b, x1b:x2b]

            # 调整边界框坐标
            padw = x1a - x1b
            padh = y1a - y1b

            for box in boxes:
                class_id, x_center, y_center, width, height = box
                x_center = x_center * w + padw
                y_center = y_center * h + padh
                width = width * w
                height = height * h

                # 转换为相对于整个mosaic图像的坐标
                x_center /= (2 * w)
                y_center /= (2 * h)
                width /= (2 * w)
                height /= (2 * h)

                mosaicBoxes.append([class_id, x_center, y_center, width, height])

            # # 可视化整个mosaic图像
            # utils.visualize_mosaic(mosaicImage, mosaicBoxes)


        # 随机裁剪
        mosaicImage, mosaicBoxes = self.random_crop(mosaicImage, mosaicBoxes)

        # 调整大小
        mosaicImage = Image.fromarray(mosaicImage)
        mosaicImage, mosaicBoxes = self.resize(mosaicImage, np.array(mosaicBoxes),
                                                   mosaicImage.size)

        # # 可视化整个mosaic图像
        # utils.visualize_mosaic(mosaicImage,  mosaic_boxes=mosaicBoxes, class_names=self.classNames)

        # 将处理过的数据转换为tensor
        imageTensor = transforms.ToTensor()(mosaicImage)

        # 填充
        paddedImage = torch.zeros(3, self.imageSize, self.imageSize)
        _, h, w = imageTensor.shape
        paddedImage[:, :h, :w] = imageTensor

        targets = torch.zeros((config_paramters.TARGETS_SIZE, 6))
        if len(mosaicBoxes) > 0:
            targets[:len(mosaicBoxes), 1:] = torch.from_numpy(mosaicBoxes)
            targets[:, 0] = idx

        return paddedImage, targets



    # 调整图像与标签尺寸为指定大小
    def resize(self, image, boxes, originalSize):
        w, h = originalSize
        new_w, new_h = self.imageSize, self.imageSize

        # 计算缩放比例
        r = min(new_w / w, new_h / h)
        nw, nh = int(w * r), int(h * r)

        # 调整图像大小
        image = image.resize((nw, nh), Image.BILINEAR)

        # 创建新图像
        new_image = Image.new('RGB', (new_w, new_h), (114, 114, 114))
        new_image.paste(image, ((new_w - nw) // 2, (new_h - nh) // 2))

        # 调整边界框坐标
        if len(boxes) > 0:
            dw = (new_w - nw) / 2
            dh = (new_h - nh) / 2

            boxes[:, 1] = (boxes[:, 1] * w * r + dw) / new_w  # x_center
            boxes[:, 2] = (boxes[:, 2] * h * r + dh) / new_h  # y_center
            boxes[:, 3] = boxes[:, 3] * w * r / new_w         # width
            boxes[:, 4] = boxes[:, 4] * h * r / new_h         # height

        return new_image, boxes


    # 对于图像和标签进行随机裁剪的操作
    def random_crop(self, image, boxes):
        if random.random() < config_paramters.RANDOM_CROP_RATIO:
            return image, boxes

        h, w = image.shape[:2]

        # 随机裁剪尺寸大小
        scaleH = random.uniform(config_paramters.RANDOM_CROP_SCOPE[0], config_paramters.RANDOM_CROP_SCOPE[1])
        scaleW = random.uniform(config_paramters.RANDOM_CROP_SCOPE[0], config_paramters.RANDOM_CROP_SCOPE[1])

        newH = int(h * scaleH)
        newW = int(w * scaleW)

        # 随机裁剪位置
        y = random.randint(0, h - newH)
        x = random.randint(0, w - newW)

        # 裁剪图像
        croppedImg = image[y:y+newH, x:x+newW]

        # 调整边界框
        newBoxes = []
        for box in boxes:
            classId, xCenter, yCenter, width, height = box

            # 转换为绝对坐标
            xCenterAbs = xCenter * w
            yCenterAbs = yCenter * h
            widthAbs = width * w
            heightAbs = height * h

            # 计算边界框坐标
            x1 = xCenterAbs - widthAbs / 2
            y1 = yCenterAbs - heightAbs / 2
            x2 = xCenterAbs + widthAbs / 2
            y2 = yCenterAbs + heightAbs / 2

            # 检查边界框是否在裁剪区域内
            if (x1 < x + newW and x2 > x and y1 < y + newH and y2 > y):
                # 调整坐标
                x1 = max(x1, x) - x
                y1 = max(y1, y) - y
                x2 = min(x2, x + newW) - x
                y2 = min(y2, y + newH) - y

                # 转换回相对坐标
                new_x_center = (x1 + x2) / 2 / newW
                new_y_center = (y1 + y2) / 2 / newH
                new_width = (x2 - x1) / newW
                new_height = (y2 - y1) / newH

                # 过滤太小的边界框
                if new_width > 0.01 and new_height > 0.01:
                    newBoxes.append([classId, new_x_center, new_y_center, new_width, new_height])


        return croppedImg, newBoxes


    # 随机仿射变换
    def random_affine(self, image, boxes, degrees, translate, scale, shear):
        if random.random() < 0.5:
            return image, boxes

        image = np.array(image)
        height, width = image.shape[:2]

        # 保存原始图像尺寸用于坐标转换
        orig_h, orig_w = height, width

        # 旋转和缩放
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

        # 平移
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * width
        T[1, 2] = random.uniform(-translate, translate) * height

        # 剪切
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

        M = S @ T @ R
        imw = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))

        # 变换边界框
        n = len(boxes)
        if n:
            # 将中心点坐标转换为角点坐标
            center_boxes = boxes.copy()

            # 转换为像素坐标 [class_id, x_center, y_center, width, height] -> 角点坐标
            x_center = center_boxes[:, 1] * orig_w
            y_center = center_boxes[:, 2] * orig_h
            box_width = center_boxes[:, 3] * orig_w
            box_height = center_boxes[:, 4] * orig_h

            # 计算角点坐标 [x_min, y_min, x_max, y_max]
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            x_max = x_center + box_width / 2
            y_max = y_center + box_height / 2

            # 准备角点坐标(像素坐标)
            xy = np.ones((n * 4, 3))

            # 四个角点: 左上, 右上, 右下, 左下
            corners = np.column_stack([
                x_min, y_min,  # 左上
                x_max, y_min,  # 右上
                x_max, y_max,  # 右下
                x_min, y_max  # 左下
            ]).reshape(n * 4, 2)

            xy[:, :2] = corners

            # 应用变换
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # 透视除法

            # 创建新的边界框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]

            new_x_min = x.min(1)
            new_y_min = y.min(1)
            new_x_max = x.max(1)
            new_y_max = y.max(1)

            # 裁剪到有效范围
            new_x_min = np.clip(new_x_min, 0, width)
            new_y_min = np.clip(new_y_min, 0, height)
            new_x_max = np.clip(new_x_max, 0, width)
            new_y_max = np.clip(new_y_max, 0, height)

            # 转换回中心点坐标和宽高（归一化）
            new_x_center = ((new_x_min + new_x_max) / 2) / width
            new_y_center = ((new_y_min + new_y_max) / 2) / height
            new_width = (new_x_max - new_x_min) / width
            new_height = (new_y_max - new_y_min) / height

            # 合并类别信息
            classes = boxes[:, 0]
            new_boxes = np.column_stack([classes, new_x_center, new_y_center, new_width, new_height])

            # 过滤无效边界框
            w = new_width * width  # 像素宽度
            h = new_height * height  # 像素高度
            area = w * h

            # 原始边界框面积（用于比较）
            orig_area = (box_width * box_height)

            # 使用绝对像素尺寸进行过滤
            i = (w > 2) & (h > 2) & (area / (orig_area + 1e-16) > 0.05) & (
                        np.maximum(w / (h + 1e-16), h / (w + 1e-16)) < 20)
            boxes = new_boxes[i]

        return Image.fromarray(imw), boxes


    # 随机颜色的调整
    def random_color(self, image):
        if random.random() < config_paramters.RANDOM_COLOR_RATIO:
            return image

        image = np.array(image)

        # HSV颜色空间增强
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 调整饱和度
        sFactor = random.uniform(0.5, 1.5)
        s = np.clip(s * sFactor, 0, 255).astype(np.uint8)

        # 调整明度
        vFactor = random.uniform(0.5, 1.5)
        v = np.clip(v * vFactor, 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(image)



def main():

    # TODO: 先检查一下数据集是否已经下载过了，如果下载过了，就不用再执行下载函数了

    # 下载数据集
    # utils.check_dataset(config_path.DATASETS_YAML)

    # 构建训练、验证、测试等数据集路径，便于后续的加载与访问
    dataInfo = utils.create_data_path(config_path.DATASETS_YAML)

    # 创建数据集
    fullDatasets = COCODataset(dataInfo['path'], config_paramters.IMAGE_SIZE)


    # 划分索引，整理出训练集与验证集
    indices = list(range(len(fullDatasets)))
    trainIndices, valIndices = train_test_split(
        indices,
        test_size=config_paramters.RATIO,   # 作为验证集的比例
        random_state=42,    # 设定随机数种子 使得每次分配的整体集合是一致的
        shuffle=True        # 设置是否打乱
    )

    trainDatasets = Subset(
        COCODataset(dataInfo['path'], config_paramters.IMAGE_SIZE),
        trainIndices
    )

    valDatasets = Subset(
        COCODataset(dataInfo['path'], config_paramters.IMAGE_SIZE),
        valIndices
    )

    # 创建数据集加载器
    trainDataLoader = DataLoader(
        trainDatasets,
        batch_size=config_paramters.BATCH_SIZE,
        shuffle=True,
        num_workers=config_paramters.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )

    valDataLoader = DataLoader(
        valDatasets,
        batch_size=config_paramters.BATCH_SIZE,
        shuffle=False,
        num_workers=config_paramters.NUM_WORKERS,
        drop_last=True  # 不保留最后一个不完整批次
    )


    # os.makedirs('visualization', exist_ok=True)
    # # 可视化一下加载的数据集
    # with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:
    #
    #     # 训练集
    #     for batchIndex, (augmentationData, targets) in enumerate(trainDataLoader):
    #         for i in range(config_paramters.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)
    #
    #     # 验证集
    #     for batchIndex, (augmentationData, targets) in enumerate(valDataLoader):
    #         for i in range(config_paramters.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)


    return trainDataLoader, valDataLoader


if __name__ == '__main__':
    main()




