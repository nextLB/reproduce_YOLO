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


# 数据加载和预处理
class COCODataset(Dataset):
    def __init__(self, datasetPath, imageSize):

        self.datasetPath = datasetPath
        self.imageSize = imageSize
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
        print(self.images[idx], self.labels[idx])


        return idx








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

        for batchIndex in enumerate(trainDataLoader):
            for i in range(config_paramters.BATCH_SIZE):
                print(f'batchIndex: {batchIndex}, i: {i}')
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




