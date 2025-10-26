
"""
    自主实现的本项目需要用到的工具函数
"""
import os
import config_path
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches




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




def visualize_image_with_bboxes(imageData, boundingBoxes, figsize=(12, 8)):
    """
    可视化图像和对应的边界框

    参数:
    imageData: PIL Image对象
    boundingBoxes: 边界框列表，格式为 [(name, (xmin, ymin, xmax, ymax)), ...]
    figsize: 图像显示大小
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, figsize=figsize)

    # 显示图像
    ax.imshow(imageData)

    # 为不同的类别定义颜色（可以根据需要扩展）
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']

    # 绘制每个边界框
    for i, (class_name, bbox) in enumerate(boundingBoxes):
        xmin, ymin, xmax, ymax = bbox

        # 计算边界框的宽度和高度
        width = xmax - xmin
        height = ymax - ymin

        # 选择颜色（循环使用颜色）
        color = colors[i % len(colors)]

        # 创建矩形补丁
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )

        # 添加矩形到坐标轴
        ax.add_patch(rect)

        # 添加类别标签
        ax.text(
            xmin, ymin - 5, class_name,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )

    # 设置标题
    ax.set_title(f'Image with Bounding Boxes ({len(boundingBoxes)} objects)', fontsize=14)

    # 移除坐标轴
    ax.axis('off')

    # 自动调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()





