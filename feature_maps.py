"""
    本文件程序主要是在模型训练和评估等的过程中，提取模型架构的相关特征图的操作，以便于更好的理解模型的架构
"""

# 2025.10.15 (V1.0)            --- by next, 初步实现了YOLOv1的特征图提取程序文件




from typing import Callable, Any, List, Dict
import torch.nn as nn
import os
import torch
from matplotlib import cm
from PIL import Image
import numpy as np


# 特征图处理与保存函数
def save_feature_maps(featureMaps: torch.Tensor, layerName: str, outputDir: str, maxChannels: int, imgIndex: int, cmap: str) -> None:
    """
        保存特征图为彩色图片 (伪彩色映射)
        参数：
            featureMaps (torch.Tensor): 特征图张量
            layerName (str): 层名称
            outputDir (str): 保存目录
            maxChannels (int): 每个层保存的最大通道数
            imgIndex (int): 批次中要保存的图像索引
            cmap (str): 伪彩色映射方案

        关于 cmap 颜色映射类型的一些说明和解释
            默认 / 通用类：
                'viridis'：默认值，人眼对亮度变化敏感，适合大多数场景
                'plasma'：暖色调为主，适合突出高值区域
                'inferno'：深色背景，高值区域呈亮色
                'magma'：类似岩浆的色调，对比度强
            灰度 / 单色系：
                'gray'：纯灰度映射，0 为黑，255 为白
                'bone'：带蓝色调的灰度，常用于医学图像
            发散型（高低值对比）：
                'coolwarm'：低值冷色调（蓝），高值暖色调（红）
                'bwr'：蓝 - 白 - 红渐变，适合展示正负差异
            顺序型（单一色调渐变）：
                'Blues'：从浅蓝到深蓝
                'Greens'：从浅绿到深绿
                'Reds'：从浅红到深红
            特殊用途：
                'jet'：传统 rainbow 色调（但因色彩均匀性差，不推荐用于精确可视化）
                'hsv'：基于色相的循环映射，适合周期性数据
    """
    #检查存储特征图的文件夹
    os.makedirs(outputDir, exist_ok=True)
    colorMap = cm.get_cmap(cmap)

    # 逐批次的取出
    for index in range(len(featureMaps)):
        tempOutputDir = os.path.join(outputDir, f'image_{index}')
        os.makedirs(tempOutputDir, exist_ok=True)
        # detach() 方法的作用是将当前张量从计算图中 “分离” 出来，使其不再跟踪梯度（即后续操作不会影响反向传播）。
        feature = featureMaps[index].detach().cpu().float()

        # 对特征图进行逐个通道的取出
        for channelIdx in range(min(feature.shape[0], maxChannels)):
            channelData = feature[channelIdx]

            # 归一化到[0, 1]
            minVal = channelData.min()
            channelData = channelData - minVal
            maxVal = channelData.max()
            if maxVal > 0:  # 避免除以0
                channelData = channelData / maxVal

            # 应用颜色映射
            # 应用颜色映射后，正常情况下会有四通道维度，RGBA
            coloredData = colorMap(channelData.numpy())

            # 检查 coloredData的维度并进行相应的处理
            # 当此时变换后的数据只有两个维度，说明数据是一维的
            if coloredData.ndim == 2:
                # 如果是二维数据，则将其转换为三维数据 [H, W, C]
                # 这里假设 coloredData 的形状是 [484, 4], 表示484个位置，每个位置有4个值 (RGBA)
                # 我们需要将其重塑为 [484, 1, 4] 或 [1, 484, 4], 然后转换为RGB图像
                coloredData = coloredData.reshape((coloredData.shape[0], 1, coloredData.shape[1]))

                coloredData = (coloredData[:, :, :3] * 255).astype(np.uint8)    # 取前3通道 (RGB) 并转为整数

            else:
                coloredData = (coloredData[:, :, :3] * 255).astype(np.uint8)    # 取前3通道 (RGB) 并转为整数

            # 转换为PIL图像并保存
            img = Image.fromarray(coloredData)
            savePath = os.path.join(tempOutputDir, f"{layerName}_channel_{channelIdx}_color.png")
            img.save(savePath)



class MyFeatureMapHook:
    # 特征图提取类的初始化函数
    def __init__(self, model: nn.Module, outputDir: str, imgIndex: int) -> None:
        self.model = model
        self.outputDir = outputDir
        self.imgIndex = imgIndex
        self.featureMaps = {}
        self.hooks = []

    # 提取保留数据的方法
    def _hook_fn(self, layerName: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, inputData: Any, outputData: Any) -> None:
            self.featureMaps[layerName] = outputData

        return hook

    # 为指定层注册钩子
    def register_hooks(self, layerNames:List[str]) -> None:
        for name, module in self.model.named_modules():
            if name in layerNames:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    # 保存所有捕获的特征图
    def save_feature_maps(self) -> None:
        saveMaxChannels = 10000
        for layerName, feature in self.featureMaps.items():
            saveDir = os.path.join(self.outputDir, layerName)
            save_feature_maps(feature, layerName, saveDir, maxChannels=saveMaxChannels, imgIndex=self.imgIndex, cmap='viridis')

    # 移除所有钩子
    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    # 执行前向传播并捕获特征图
    def __call__(self, x: Any) -> Dict[str, torch.Tensor]:
        self.model(x)
        return self.featureMaps




if __name__ == '__main__':
    exampleHook = MyFeatureMapHook(None, None, None)



