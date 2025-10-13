"""
    Yolov3 的数据集构建程序文件
"""
import os.path
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



# 数据加载和预处理
class COCODataset(Dataset):
    def __init__(self, datasetPath, imageSize):

        self.datasetPath = datasetPath
        self.imageSize = imageSize
        self.classNames = utils.load_class_names_from_yaml(config_path.DATASETS_YAML)
        self.mosaicBorder = [-imageSize//2, -imageSize//2]

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
        # 以mosaic数据增强的形式加载图像
        self.load_image_mosaic(idx)

        return idx

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
        utils.visualize_mosaic(mosaicImage,  mosaic_boxes=mosaicBoxes, class_names=self.classNames)



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


    os.makedirs('visualization', exist_ok=True)
    # 可视化一下加载的数据集
    with tqdm(total=len(trainDataLoader)+len(valDataLoader), desc="数据集可视化中") as pbarDataloader:

        for batchIndex, (idx) in enumerate(trainDataLoader):
            for i in range(config_paramters.BATCH_SIZE):
                print('===============>>>')
            pbarDataloader.update(1)


        # # 训练集
        # for batchIndex, (originalData, augmentationData, targets) in enumerate(trainDataLoader):
        #     for i in range(config_paramters.BATCH_SIZE):
        #         print(targets.shape)
        #     pbarDataloader.update(1)

    #     # 验证集
    #     for batchIndex, (originalData, augmentationData, targets) in enumerate(valDataLoader):
    #         for i in range(config_parameter.BATCH_SIZE):
    #             print(targets.shape)
    #         pbarDataloader.update(1)



if __name__ == '__main__':
    main()




