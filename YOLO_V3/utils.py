"""
    复现YOLOv3过程中需要用到的工具
"""

import yaml
import os
import requests
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import config_path
import shutil
import config_paramters


# 检查数据集
def check_dataset(dataYamlPath):
    # 1. 读取YAML文件
    with open(dataYamlPath, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    print("=== YAML文件内容 ===")
    print(f"数据集路径: {data['path']}")
    print(f"训练集: {data['train']}")
    print(f"验证集: {data['val']}")
    print(f"下载链接: {data['download']}")
    print(f"类别数量: {len(data['names'])}")

    # 2. 显示类别信息
    print("\n=== 类别信息 ===")
    for class_id, class_name in data['names'].items():
        print(f"{class_id}: {class_name}")

    # 3. 下载数据集
    download_url = data['download']
    dataset_path = data['path']

    print(f"\n=== 开始下载数据集 ===")
    print(f"下载链接: {download_url}")
    print(f"保存到: {dataset_path}")

    # 创建目录
    os.makedirs(dataset_path, exist_ok=True)

    # 下载文件
    zip_filename = os.path.join(dataset_path, "coco128.zip")

    try:
        # 下载
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"下载进度: {progress:.1f}%", end='\r')

        print(f"\n下载完成: {zip_filename}")

        # 解压文件
        print("正在解压文件...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dataset_path))

        print("解压完成!")

        # 删除zip文件（可选）
        os.remove(zip_filename)
        print("临时文件已清理")

    except Exception as e:
        print(f"下载或解压过程中出现错误: {e}")

    return data



# 构建数据集路径
def create_data_path(dataYamlPath):
    # 1. 读取YAML文件
    with open(dataYamlPath, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # 2、构建完整路径
    data['train'] = os.path.join(data['path'], data['train'])
    data['val'] = os.path.join(data['path'], data['val'])

    return data



# 加载mosaic数据时的可视化函数     (具体用法可详见我的数据加载器的load_image_mosaic方法中)
def visualize_mosaic(mosaic_image, mosaic_boxes, class_names=None, figsize=(12, 12)):
    """
    可视化mosaic图像和边界框

    参数:
    - mosaic_image: numpy数组，mosaic图像
    - mosaic_boxes: list，边界框列表，格式为[[class_id, x_center, y_center, width, height], ...]
    - class_names: list，类别名称列表，如果为None则显示class_id
    - figsize: tuple，图像显示大小
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 检查是否为PIL Image对象，如果是则转换为numpy数组
    if isinstance(mosaic_image, Image.Image):
        mosaic_image = np.array(mosaic_image)
    else:
        mosaic_image = mosaic_image

    # 显示图像
    ax.imshow(mosaic_image)

    # 获取图像尺寸
    img_height, img_width = mosaic_image.shape[:2]

    # 定义颜色映射（可以根据需要修改）
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    # 绘制每个边界框
    for box in mosaic_boxes:
        class_id, x_center, y_center, width, height = box

        # 将归一化坐标转换为绝对坐标
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # 计算边界框的左上角坐标
        x_min = x_center_abs - width_abs / 2
        y_min = y_center_abs - height_abs / 2

        # 创建矩形补丁
        rect = patches.Rectangle(
            (x_min, y_min), width_abs, height_abs,
            linewidth=2, edgecolor=colors[int(class_id) % len(colors)],
            facecolor='none', alpha=0.8
        )

        # 添加矩形到图像
        ax.add_patch(rect)

        # 添加类别标签
        if class_names is not None and int(class_id) < len(class_names):
            label = class_names[int(class_id)]
        else:
            label = f'Class {int(class_id)}'

        ax.text(
            x_min, y_min - 5, label,
            color=colors[int(class_id) % len(colors)],
            fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )

    # 设置标题和坐标轴
    ax.set_title('Mosaic Image with Bounding Boxes', fontsize=16, fontweight='bold')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # 注意：y轴需要反转以正确显示图像

    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# 可视化单个图像与类别的函数   (具体用法可详见我的数据加载器的load_image_mosaic方法中)
def visualize_single_image(image, boxes, class_names=None, figsize=(10, 10), title="Image with Bounding Boxes"):
    """
    可视化单张图像和边界框

    参数:
    - image: numpy数组或PIL图像
    - boxes: list，边界框列表 [[class_id, x_center, y_center, width, height], ...]
    - class_names: list，类别名称列表
    - figsize: tuple，图像显示大小
    - title: str，图像标题
    """
    # 确保图像是numpy数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    img_height, img_width = image.shape[:2]

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)

    # 定义颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    # 绘制每个边界框
    for box in boxes:
        class_id, x_center, y_center, width, height = box

        # 将归一化坐标转换为绝对坐标
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # 计算边界框的左上角坐标
        x_min = x_center_abs - width_abs / 2
        y_min = y_center_abs - height_abs / 2

        # 创建矩形补丁
        color = colors[int(class_id) % len(colors)]
        rect = patches.Rectangle(
            (x_min, y_min), width_abs, height_abs,
            linewidth=2, edgecolor=color,
            facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

        # 添加类别标签
        if class_names is not None and int(class_id) < len(class_names):
            label = class_names[int(class_id)]
        else:
            label = f'Class {int(class_id)}'

        ax.text(
            x_min, y_min - 5, label,
            color=color, fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )

    # 设置标题和坐标轴
    ax.set_title(f'{title}\n({len(boxes)} boxes)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# 加载指定yaml文件中的类别标签
def load_class_names_from_yaml(yaml_path):
    """
    从YAML配置文件中加载实际的类别名称

    参数:
    - yaml_path: str, YAML文件的完整路径

    返回:
    - list: 格式为 ['person', 'bicycle', 'car', ...] 的实际类别名称列表
    """
    try:
        # 读取YAML文件
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取names字段
        names_dict = data.get('names', {})

        # 按照类别ID排序并提取实际类别名称
        class_names = [names_dict[class_id] for class_id in sorted(names_dict.keys())]

        return class_names

    except FileNotFoundError:
        print(f"错误: 找不到文件 {yaml_path}")
        return []
    except yaml.YAMLError as e:
        print(f"错误: YAML解析失败 - {e}")
        return []
    except Exception as e:
        print(f"错误: 加载类别名称失败 - {e}")
        return []



def clear_folder(folderPath: str):

    # 遍历文件夹内的所有内容
    for item in os.listdir(folderPath):
        itemPath = os.path.join(folderPath, item)
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(itemPath) or os.path.islink(itemPath):
                os.unlink(itemPath)
                print(f'已删除文件夹: {itemPath}')
            elif os.path.isdir(itemPath):
                # 使用 shutil.rmtree 删除文件夹及其内容
                shutil.rmtree(itemPath)
                print(f"已删除文件夹： {itemPath}")

        except Exception as e:
            print(f"删除 {itemPath} 时出错： {e}")

    print(f"文件夹 {folderPath} 内容已清空")




# 创建模型训练过程中各文件的存储路径
def create_all_path():
    saveModelPath = os.path.join(config_path.MODEL_PATH, f'maxEpochs_{config_paramters.MAX_EPOCHS}_learningRate_{config_paramters.LEARNING_RATE}')
    saveLogFilePath = os.path.join(config_path.LOG_FILE_PATH, f'maxEpochs_{config_paramters.MAX_EPOCHS}_learningRate_{config_paramters.LEARNING_RATE}')
    saveFeatureMapsPath = os.path.join(config_path.FEATURE_MAPS_PATH, f'maxEpochs_{config_paramters.MAX_EPOCHS}_learningRate_{config_paramters.LEARNING_RATE}')
    os.makedirs(saveModelPath, exist_ok=True)
    clear_folder(saveModelPath)
    os.makedirs(saveModelPath, exist_ok=True)

    os.makedirs(saveLogFilePath, exist_ok=True)
    clear_folder(saveLogFilePath)
    os.makedirs(saveLogFilePath, exist_ok=True)

    os.makedirs(saveFeatureMapsPath, exist_ok=True)
    clear_folder(saveFeatureMapsPath)
    os.makedirs(saveFeatureMapsPath, exist_ok=True)

    return saveModelPath, saveLogFilePath, saveFeatureMapsPath






