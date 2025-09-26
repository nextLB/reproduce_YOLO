
"""
    YOLOV1 数据集的相关配置
"""
import torch
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import config_path
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import config_parameter




class YOLOPascalVocDataset(Dataset):
    def __init__(self, setType, normalize, augment):
        assert setType in {'train', 'test'}

        self.dataset = VOCDetection(
            root=config_path.DATA_PATH,
            year='2007',
            image_set=('train' if setType == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config_parameter.IMAGE_SIZE)
            ])
        )








def main():
    trainDatasets = YOLOPascalVocDataset('train', True, True)




if __name__ == '__main__':
    main()






