

"""
    自主实现的本项目需要用到的工具函数
"""
import cv2
import torch
import json
import os
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt
import config_path
import re
import numpy as np
import config_parameter


# 获取训练数据集所有类别的函数
def load_train_classes():
    readClassPath = os.path.join(config_path.TRAIN_DATA_PATH, 'ImageSets/Main')

    classNames = []
    for root, dirs, files in os.walk(readClassPath):
        for nowFile in files:
            splitList = nowFile.split('_')
            if splitList[0] not in classNames and len(splitList) == 2 and splitList[0] != 'val':
                classNames.append(splitList[0])

    # 对类别名称进行排序以确保一致性
    classNames.sort()

    # 转换为字典形式：{class_name: index}
    class_dict = {class_name: idx for idx, class_name in enumerate(classNames)}

    return class_dict



# 获取训练数据集所有图像的名称，便于后续构建数据集加载路径
def load_train_images_name():
    readImagesPath = os.path.join(config_path.TRAIN_DATA_PATH, 'JPEGImages')

    imagesName = []
    for root, dirs, files in os.walk(readImagesPath):
        for nowFile in files:
            imagesName.append(nowFile)

    # 定义正则排序规则
    def extract_number(filename):
        numberStr = re.findall(r'\d+', filename)[0]     # 取第一个数字串
        return int(numberStr)

    sortedImagesName = sorted(imagesName, key=extract_number)

    return sortedImagesName



# 获取训练集的标签文件名称，便于后续构建数据集加载路径
def load_train_labels_name():
    readLabelsPath = os.path.join(config_path.TRAIN_DATA_PATH, 'Annotations')

    labelsName = []
    for root, dirs, files in os.walk(readLabelsPath):
        for nowFile in files:
            labelsName.append(nowFile)


    # 定义正则排序规则
    def extract_number(filename):
        numberStr = re.findall(r'\d+', filename)[0]     # 取第一个数字串
        return int(numberStr)

    sortedLabelsName = sorted(labelsName, key=extract_number)

    return sortedLabelsName




def visualize_image_and_label(originalImage, augmentationImage, boundingBoxes):
    # 将Tensor转换为numpy数组并调整通道顺序
    originalImage = originalImage.numpy().transpose(1, 2, 0)
    augmentationImage = augmentationImage.numpy().transpose(1, 2, 0)

    # 对增强图像进行反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    augmentationImage = augmentationImage * std + mean
    augmentationImage = np.clip(augmentationImage, 0, 1)

    # 对原始图像也进行反归一化（如果需要）
    originalImage = originalImage * std + mean
    originalImage = np.clip(originalImage, 0, 1)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 显示原始图像
    ax1.imshow(originalImage)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 显示增强图像
    ax2.imshow(augmentationImage)
    ax2.set_title('Augmentation Image')
    ax2.axis('off')

    # 定义不同类别的颜色（可选）
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # 使用matplotlib的Rectangle绘制边界框并添加类别名称
    for i, (className, (xMin, yMin, xMax, yMax)) in enumerate(boundingBoxes):
        color = colors[i % len(colors)]  # 循环使用颜色

        # 在原始图像上绘制矩形
        rect1 = patches.Rectangle((xMin, yMin), xMax - xMin, yMax - yMin,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect1)

        # 在原始图像上添加类别名称（带背景）
        ax1.text(xMin, yMin - 5, className,
                 fontsize=10, color='white', weight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

        # 在增强图像上绘制矩形
        rect2 = patches.Rectangle((xMin, yMin), xMax - xMin, yMax - yMin,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect2)

        # 在增强图像上添加类别名称（带背景）
        ax2.text(xMin, yMin - 5, className,
                 fontsize=10, color='white', weight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.show()








