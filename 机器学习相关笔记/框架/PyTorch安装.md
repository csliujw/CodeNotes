# 安装记录`PyTorch`

- 安装cuda
- 安装cudnn
- 安装pytorch gpu版本

安装cuda的时候要查看自己显卡驱动和那个版本的cuda匹配，要下载匹配的。

<a href="https://developer.nvidia.com/cuda-toolkit-archive">旧版cuda下载地址</a>

安装cudnn的过程，只需要下载好文件，然后移动即可。

<a href="https://developer.nvidia.com/rdp/cudnn-archive">旧版本cudnn下载地址</a>

<a href="https://blog.csdn.net/sinat_23619409/article/details/84202651">cudnn安装方法</a>

安装pytorch gup版本的时候速度可能会很慢，建议换成阿里云的源或者直接去清华镜像下载 然后安装。

安装普通的库下载过慢，可将源换为豆瓣源 或 阿里源pi

```python
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url 源地址

常用的源
阿里云 https://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
豆瓣(douban) http://pypi.douban.com/simple/
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/
```

换了之后发现没有。于是我又试了这个

```python
首先输这个，更换镜像源（注意顺序，第四条一定要在最后，原因不详）

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
最后，运行执行安装pytorch

conda install pytorch torchvision cudatoolkit=10.0
搞定
```

----

实际上我是按下面的步骤，和参考上面的博客成功的

- 安装N卡驱动，注意N卡驱动支持的cuda版本，这里建议用驱动精灵这些工具，下载18年发行的驱动，然后查看驱动支持的cuda版本。
  - 如果没有n卡的控制面板，不能查看支持的cuda型号，就去下一个<a href="https://www.qqxiazai.com/down/44050.html">一个下载地址,我没用过！！！</a>
- 安装对应版本的cuda
- 安装miniconda <a href="ModuleNotFoundError: No module named 'torch._C'">下载地址</a>
- 下载pytorch的离线包，离线安装。在线安装我试了好多次，总出错，最后还是去清华镜像下载的。<a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/">下载地址</a>
- 下载离线安装包 <a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/">清华镜像下载地址</a>
  - 注意py的版本 一定要一致，不一致会出错！！！
- conda install 下载的压缩包
- over

```python
# 测试代码
import torch

if __name__ == '__main__':
    print(123)
    print(torch.cuda.is_available())
    
# output
# 123
# True
```

