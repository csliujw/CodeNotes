```shell
# 查看安装了那些软件
conda list 
# 查看anaconda信息
conda info
# 查看有哪些环境
conda info -e
# 创建新环境，用清华源创建环境总失败
conda create -n 环境名称 python=指定python版本
# 卸载环境中的某些包
conda uninstall pytorch=1.4.0 卸载指定版本的软件

# 现在torch官网安装pytorch的方式快的可怕，用啥清华源。没必要了。好吧，我错了，也就python3.8的安装快。

# 换源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 然后安装pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly

# 移除源
conda config --remove-key channels

# 清华源安装opencv-python 也可以用豆瓣源啥的。
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

