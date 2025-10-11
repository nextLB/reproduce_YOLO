"""
    Yolov3 的数据集构建程序文件
"""

import config_paramters
import config_path
import utils




def main():
    # 检查数据集
    utils.check_dataset(config_path.DATASETS_YAML)

    # 构建训练、验证、测试等数据集路径，便于后续的加载与访问





if __name__ == '__main__':
    main()




