

"""
    自主实现的本项目需要用到的工具函数
"""

import torch
import json
import os
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt
import config_path



# 加载标签类别的函数
def load_class_dict():
    if os.path.exists(config_path.CLASSES_PATH):
        with open(config_path.CLASSES_PATH, 'r') as file:
            return json.load(file)

    newDict = {}
    save_class_dict(newDict)
    return newDict



# 保存标签字典的函数
def save_class_dict(obj):
    folder = os.path.dirname(config_path.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config_path.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


