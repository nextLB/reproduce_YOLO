"""
    YOLOV1的训练程序
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main():
    # SummaryWriter是使用pytorch的TensorBoard集成功能
    # 在终端运行  然后浏览器打开 http://localhost:6006
    writer = SummaryWriter()    # 默认创建 runs/当前时间 目录

    now = datetime.now()


if __name__ == '__main__':
    main()






