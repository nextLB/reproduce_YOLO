"""
    复现YOLOv3过程中需要用到的工具
"""

import yaml
import os
import requests
import zipfile


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