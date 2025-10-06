# next 关于YOLO系列论文的复现


## 程序运行环境的配置

---创建基于anconda的python虚拟环境

    conda create -n next_pytorch python=3.11

---启动创建的python虚拟环境

    conda activate next_pytorch

---配置与安装相关依赖

    pytorch环境配置 

        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    ultralytics配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics

    numpy配置
        
        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==2.1.2
    
    pandas配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.3.2

    pillow配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow

    matplotlib配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib

    tqdm配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm==4.67.1

    opencv配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless

    albumentations配置
        
        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple albumentations

    sklearn配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn==1.7.2
    
    tensorboard配置

        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard



