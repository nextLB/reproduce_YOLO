"""
    Yolov3 的数据集构建程序文件
"""

import config_paramters
import config_path
import utils
from torch.utils.data import Dataset, DataLoader


# 数据加载和预处理
class COCODataset(Dataset):
    def __init__(self, datasetPath, imageSize):

        self.datasetPath = datasetPath
        self.imageSize = imageSize
        print(datasetPath)


def main():

    # TODO: 先检查一下数据集是否已经下载过了，如果下载过了，就不用再执行下载函数了

    # 下载数据集
    # utils.check_dataset(config_path.DATASETS_YAML)

    # 构建训练、验证、测试等数据集路径，便于后续的加载与访问
    dataInfo = utils.create_data_path(config_path.DATASETS_YAML)

    # 创建数据集
    trainDataset = COCODataset(dataInfo['path'], config_paramters.IMAGE_SIZE)



if __name__ == '__main__':
    main()




