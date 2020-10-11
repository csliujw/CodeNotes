# 概述

工业界推荐用Tensflow

学术界推荐用Pytorch

# 常用的网络

- resnet
- vgg
- FCN（语义分割划时代的网络？？）
- deeplab （目前知道有三种）
- unet（目前知道有很多种）

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

---

# PyTorch基本操作

## 矩阵操作

### 取随机矩阵

api的基本使用方式为：

- 创建各种随机数据  torch.methodName(行，列)

- 给定数据创建 torch.methodName([数据])

**常用api**

**表示元组**

| methodName                        | describe                  |
| --------------------------------- | ------------------------- |
| Tensor(*sizes)                    | 基础构造函数              |
| tensor(data,)                     | 类似np.array的构造函数    |
| ones(*sizes)                      | 全1Tensor                 |
| zeros(*sizes)                     | 全0Tensor                 |
| eye(*sizes)                       | 对⻆角线为1，其他为0      |
| arange(s,e,step)                  | 从s到e，步⻓长为step      |
| linspace(s,e,steps)               | 从s到e，均匀切分成steps份 |
| rand/randn(*sizes)                | 均匀/标准分布             |
| normal(mean,std)/uniform(from,to) | 正态分布/均匀分布         |
| randperm(m)                       | 随机排列列                |

----

### 算术&矩阵运算

- a+b

- torch.add（num1,num2）

- torch.add(x,y,out=result)

- y.add_(x) 创建的那些数据也是一个对象。 <span style="color:red">带下划线表示原地操作</span>

- torch.matmul() 矩阵乘法

  ```python
  # b.T 表示b的转置
  torch.matmul(a,b.T)
  ```

----

### 索引

pytorch的索引赋值为浅拷贝，指向同一块内存。

以下为代码验证

```python
import torch
x = torch.ones(3,3)
# 取每行数据的第一列
y = x[:,0]
y[1] = 5
# out put
"""
tensor([[1., 1., 1.],
        [5., 1., 1.],
        [1., 1., 1.]])
"""
```

### 索引高级函数

| 函数                            | 功能                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| index_select(input, dim, index) | 指定维度选取，如选某些行/列，返回一个新矩阵，深拷贝，互不影响。 |
| masked_select(input, mask)      |                                                              |
| nonzero(input)                  | 非0元素下标                                                  |
| gather(input, dim, index)       |                                                              |

> **index_select(input, dim , index)**

- input 需要选择的那个矩阵

- dim 维数 0为选择每一行，1为选择每一列

- index 为需要选择另一维的那些数据。

  - 如 dim = 0 选择行
  - index = [1 ,3]
  - 选择第一行和第三行的数据

  ```python
  tensor([[0.3346, 0.2801, 0.6146, 0.6209, 0.8886], 第0行
          [0.8055, 0.8360, 0.0765, 0.8212, 0.8358], 第1行
          [0.0052, 0.8295, 0.0566, 0.4908, 0.3620], 第2行
          [0.6363, 0.6391, 0.1611, 0.9173, 0.6518], 第3行
          [0.2913, 0.0361, 0.7733, 0.4480, 0.7769]]) 第4行
  # index = [1,3]
  torch.index_select(data,0,index)
  tensor([[0.8055, 0.8360, 0.0765, 0.8212, 0.8358],
          [0.6363, 0.6391, 0.1611, 0.9173, 0.6518]])
  
  ```

> **nonzoer(input)**

```python
zoers = torch.zeros(3,4)

zoers[0][1] = 1
zoers[1][2] = 2

torch.nonzero(zoers)
```

---

### 改变形状

- view(行，列)
- reshape(行，列)

都是浅拷贝，共用内存。

- clone() 后再改变形状

```python
data = torch.randint(0,1000,size=(3,4))
copy_data = data.clone()
reshape_data = copy_data.reshape(4,3)
print(reshape_data)
```

---

### tensor 转python数据

```python
# 当个数的转
x = torch.randn(1)
x.item()
```

## 线性代数操作

直接看官方文档就是。随用随学。

---

### 广播机制

不同形状做计算，会进行适当的广播（增加维数，一致后再计算）

```python
x = torch.arange(1,3).view(1,2)
print(x)
y = torch.arange(1,4).view(3,1)
print(y)
x + y
```

---

## tensor与numpy的转换

tensor对象调用 xx.numpy即可
numpy转tensor用
torch.from_numpy(xx)

---

## 求梯度

```python
import torch
"""
自动求梯度
.requires_grad 属性 为是否被跟踪
张量的梯度会自动累加到 .grad
阻止跟踪可以使用 .detach()

下面的看不懂
为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad(): 中。
在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，但是
我们不需要在此过程中对他们进行梯度计算。
"""
class grad:

    def __init__(self):
        self.x = torch.randint(0, 100, (2, 2), dtype=torch.float32, requires_grad=True)
        self.y = self.x + 2
        self.z = self.y * self.y * 3

    def demo_1(self):
        # 暂且不管为何种意思
        out = self.z.mean()
        print(self.z, out)
        out.backward()  # 进行求导
        print(self.x.grad)  # 输出对x倒数

    def demo_2(self):
        # 目前测试结果就是把导数的值累加了一下而已。
        out = self.z.sum()
        out.backward()
        print(self.x.grad)

    def demo_3(self):
        # 这里似乎必须调用一个z的方法才可，不然会报错
        # 原因：需要变成标量，所以要做一个处理，如求mean，sum
        out = self.z
        out.backward()
        print(self.x.grad)

    def demo_4(self):
        out = self.z.mean()
        out.backward()
        print(self.x.grad)

    def demo_5(self):
        x = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
        print(x)
        y = x * x * 2
        y.backward()
        # 对x的求导结果为20. 之前那些都是向量求导，还不会好吧。
        # xx.mean()这种链式调用应该是和求导的结果有关，暂时不看
        # 等学了矩阵论再说。
        print(x.grad)

    def help_docs(self, obj):
        help(obj)


if __name__ == '__main__':
    obj = grad()
    # obj.demo_2()
    # obj.help_docs(obj.z)
    obj.demo_5()
```

**pytorch的数据类型。与numpy的array类似。**


```python
import torch
a = torch.tensor([1,2,3],dtype=int)
print(a)
print(a.dtype)
# output
# tensor([1, 2, 3])
# torch.int64
```

**定义二维数据并查看数据类型**


```python
tensor = torch.tensor([[1,2,3],[4,5,6]])
print(tensor.shape)
print(tensor.size())
# torch.Size([2, 3])
# torch.Size([2, 3])
```

**查看维度**


```python
print(tensor.ndim)
# output
# 2
```

**生成数据**


```python
print(torch.zeros(2,3))# 都是0
print(torch.ones(2,3)) # 都是1
print(torch.rand(3,4)) # 生成0-1的随机的3*4的数据
print(torch.randint(0,100,(3,4)))
print(torch.randn(3,4)) # 生成符合正态分布的数据
a = torch.randn(3,4)
b = torch.rand_like(a,dtype=float) #生成和a形状一样的 float类型的随机数
print(a)
print(b)

"""
output
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.2974, 0.5818, 0.2805, 0.6552],
        [0.7131, 0.6139, 0.5760, 0.5053],
        [0.1240, 0.3698, 0.2391, 0.9218]])
tensor([[85, 47, 54, 52],
        [ 6,  3, 18, 36],
        [15, 83,  5, 53]])
tensor([[ 0.7933,  0.8227, -0.4664, -0.1945],
        [ 0.2067, -1.3382,  0.6787,  0.2010],
        [ 1.0566, -1.4890,  0.9925, -1.3757]])
tensor([[-1.2703,  0.4650,  0.2521,  0.0509],
        [ 0.7025,  0.1780,  0.2240, -0.4364],
        [ 2.5631, -0.3438,  1.0812, -0.0662]])
tensor([[0.4620, 0.6949, 0.0714, 0.5057],
        [0.7386, 0.6092, 0.3711, 0.6469],
        [0.2036, 0.4890, 0.6017, 0.4017]], dtype=torch.float64)
"""
```

**查看形状并修改**


```python
print(b.shape)
print(b.size())
# output
# torch.Size([3, 4])
# torch.Size([3, 4])
```

**重新调整大小**

```python
c = b.reshape(4,3)
d = b.view(4,3)
print(c)
print(d)
"""
output
tensor([[0.4620, 0.6949, 0.0714],
        [0.5057, 0.7386, 0.6092],
        [0.3711, 0.6469, 0.2036],
        [0.4890, 0.6017, 0.4017]], dtype=torch.float64)
tensor([[0.4620, 0.6949, 0.0714],
        [0.5057, 0.7386, 0.6092],
        [0.3711, 0.6469, 0.2036],
        [0.4890, 0.6017, 0.4017]], dtype=torch.float64)
"""
```

**tensor数据变为python数据**


```python
d[0][0].item() # 只能转换单个值
# output
# 0.46197234568033185
```

**np和pytorch之间的矩阵转换**


```python
import numpy as np
np.array(d) # 把tensor变成numpy array
# output 
# dtype('float64')
tensor = torch.tensor(np.array(d)) # numpy 变为 tensor
print(tensor)
# output
# tensor([[0.4620, 0.6949, 0.0714],
        [0.5057, 0.7386, 0.6092],
        [0.3711, 0.6469, 0.2036],
        [0.4890, 0.6017, 0.4017]], dtype=torch.float64)
```

---

# NN基本步骤

## NN基本搭建流程

### 文字描述

前馈，反馈，循环。前馈算损失，反馈算梯度，循环找最优解。

用pytorch构建自己的深度学习模型的时候，可考虑采用以下流程：

- 找数据定义
- 找model定义（找损失函数、优化器定义）
- 主循环代码逻辑

深度学习模型过程

- 输入处理模块（把输入的数据变成网络能够处理的Tensor类型）
- 模型构建模块（根据输入的数据得到预测的y，【前向过程】）
  - 前向过程只会得到模型预测结果，不会自动求导和更新
- 定义代价函数和优化器模块
- 构建训练过程（迭代训练过程）

### 代码举例

#### 数据处理

```python
from torch.utils.data import Dataset
# 继承Dataset方法，Override以下方法
class trainDataset(Dataset):
    def __init__(self):
        # 构造函数，初始化常用数据
    
    def __getitem__(self,index):
        # 获得第index号的数据和标签
        
    def __len__(self):
        # 获得数据量   
        
        
# 假设我们把数据集 和 标签 放在了一起，这样我们就可以十分方便地同时遍历数据集和标签了
data_items = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for label,data in data_items
	print(label,data)
    break # 只输出一组
```

#### 构建模型

所有模型都需要继承torch.nn.Module,需要实现以下方法.

其中forward() ⽅法是前向传播的过程。在实现模型时，我们不需要考虑反向传播。

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        
    def forward(self,x):
        return
    
 model = MyModel()
```

####  定义代价函数和优化器

```python
criterion = torch.nn.BCELoss(reduction='sum') # 代价函数
optimizer = torch.optim.Adam(model.paramenters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False) # 优化器
```

#### 构建训练过程

```python
def train(epoch):
    for i,data in data_items:
        x,y = i,data
        y_pred = model(x) # 前向传播
        loss = criterion(y_pred, y) # 计算代价函数
        optimizer.zero_grad() # 清理梯度准备计算
        loss.backward() # 反向传播
        optimizer.step() # 更新训练参数
```

#### 流程图如下

```flow
flow
st=>start: Prepare dataset
op1=>operation: Design model using class
op2=>operation: Construct loss and optimizer
op3=>operation: Training cycle
e=>end
st->op1->op2->op3->e
```

#### 代码模板

```python
import torch
from torch.nn import Module

# 步骤一 预定义数据  建议定义一个类 专门处理数据
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 步骤二 Design model using class  建议定义一个类 专门定义模型
class LogisticRegressionModel(Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


if __name__ == '__main__':  
    model = LogisticRegressionModel()
    # 步骤三 construct loss and optimizer 
    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 步骤四 training cycle
    for epoch in range(1500):
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## NN基本概念&优化器&损失函数

### 反向传播

反向传播：求偏导数。神经网络层数过高的话，求导很麻烦，故采用反向传播。反向传播的核心是计算图。

就是链式法则。

### 激活函数

- sigmoid（常用）
- tanh
- ReLU（NN中常用）只压缩了小于0的数，可有效降低每一层训练神经元的活跃度，能够降低训练的复杂的。
- Leaky ReLU（弥补了ReLU丢弃一半参数的缺陷）
- Maxout
- ELU

### 损失函数

- 均方差损失Mean Squared Error Loss

  ```python
  torch.nn.MSELoss
  ```

- 对数损失-交叉熵

  ```python
  torch.nn.CrossEntropyLoss
  ```

  交叉熵损失函数，刻画的是实际输出（概率）与期望输出（概率）分布的距离，也就是交叉熵的值越小，两个**概率分布就越接近**。

- 二分类交叉熵

  ```python
  torch.nn.BCELoss
  ```

  二分类交叉熵把 y , 1 − y {y, 1-y}*y*,1−*y* 当做两项分布，计算出来的loss就比交叉熵大（因为包含了正类和负类的交叉熵了）。

###  优化器

- 随机梯度下降优化器SGD
  - SGD的问题
    - （W，b）的每一个分量获得的梯度绝对值有大有小，一些情况下，将会迫使路径变成Z字形状。
    - SGD求梯度的策略过于随机，由于上一次和下一次用的是完全不同的BATCH数据，将会出现优化的方向随机的情况。
- AdaGrad&RMSProp解决了各个方向梯度不一致的问题

- 解决梯度随机性问题：
  - Momentum 
- 以上的综合，解决了两个问题
  - Adam
- <a href="https://zhuanlan.zhihu.com/p/32230623">推荐博客</a>

### 学习建议

深度学习的模型很多，不必部学。学其精髓，即如何构造模型/构造模型的套路。可以把某些模型当作基本块，用这些基本块进行组装。

CUDA的安装，如果安装过vs 可能会出错，所以cuda建议选择自定义安装，不要勾选vs支持。

##  NN的参数设置

### 分类

- **随机梯度下降**

  - 不用每输入一个样本就去变换参数，而是输入一批样本（叫做一个BATCH或MINI-BATCH），求出这些样本的梯度平均值后，根据这个平均值改变参数。
  - 在神经网络训练中，BATCH的样本数大致设置为50-200不等。    

- **激活函数选择**

  - 常见非线性函数选择 sigmoid、tanh、ReLU

- **训练数据初始化**

  - 建议做均值和方差归一化，防止使用激活函数时，很多数据集中在边缘，大的数据和很大的数据之间看不到什么差异（给数据做归一化，让数据均匀分布，充分利用信息？）

- **（W，b)的初始化**

  - 梯度消失现象：如果$W^T X$一开始很大或很小，那么梯度将趋近于0，反向传播后前面与之相关的梯度也趋近于0，导致训练缓慢。 
    因此，我们要使       一开始在零附近。

    一种比较简单有效的方法是：（W,b）初始化从区间$(- \frac{1}{\sqrt{d}},\frac{1}{\sqrt{d}})$均匀随机取值。其中d为（W,b）所在层的神经元个数。

    可以证明，如果X服从正态分布，均值0，方差1，且各个维度

    无关，而（W,b）是$(- \frac{1}{\sqrt{d}},\frac{1}{\sqrt{d}})$的均匀分布，则$W^T X + b$是均值为0， 方差为1/3的正态分布。

- **Batch normalization**

  - 既然我们希望每一层获得的值都在0附近，从而避免梯度消失现象，那么我们为什么不直接把每一层的值做基于均值和方差的归一化呢？

  - 每一层FC（Fully Connected Layer）接一个BN（Batch Normalization）层

    <img src="..\pics\pytorch\Batch_normalization_hhj.png" style="float:left">

- **目标函数选择**
  - 可加正则项（Regulation Term）
  - 如果是分类问题，F(W) 可以采用SOFTMAX函数和交叉熵
- **参数更新策略**
  - SGD梯度过于随机，可采用上面提到过的一些梯度下降策略
  - 解决梯度随机性问题
    - momentum （角动量）
      每一次更新一个v（速度），第一次算出的方向让他第二次还有一点点影响（让上一次的方向对这次有影响）
- **训练建议**
  - 1）一般情况下，在训练集上的目标函数的平均值（cost）会随着训练的深入而不断减小，如果这个指标有增大情况，停下来。有两种情况：第一是采用的模型不够复杂，以致于不能在训练集上完全拟合；第二是已经训练很好了。
  - 2）分出一些验证集（Validation Set）,训练的本质目标是在验证集上获取最大的识别率。因此训练一段时间后，必须在验证集上测试识别率，保存使验证集上识别率最大的模型参数，作为最后结果。
  - 3）注意调整学习率（Learning Rate）,如果刚训练几步cost就增加，一般来说是学习率太高了；如果每次cost变化很小，说明学习率太低。
  - 4） Batch Normalization 比较好用，用了这个后，对学习率、参数更新策略等不敏感。建议如果用Batch Normalization, 更新策略用最简单的SGD即可，我的经验是加上其他反而不好。
  - 5）如果不用Batch Normalization, 我的经验是，合理变换其他参数组合，也可以达到目的。
  - 6）由于梯度累积效应，AdaGrad, RMSProp, Adam三种更新策略到了训练的后期会很慢，可以采用提高学习率的策略来补偿这一效应。

----

# 深度学习介绍

##  多层神经网络

###  优势

- 基本单元简单，多个基本单元可扩展为非常复杂的非线性函数。因此易于构建，同时模型有很强的表达能力。

- 训练和测试的计算并行性非常好，有利于在分布式系统上的应用。

- 模型构建来源于对人脑的仿生，话题丰富，各种领域的研究人员都有兴趣，都能做贡献。

### 劣势

数学不漂亮，优化算法只能获得局部极值，算法性能与初始值有关。

- 不可解释。训练神经网络获得的参数与实际任务的关联性非常模糊。

- 模型可调整的参数很多 （网络层数、每层神经元个数、非线性函数、学习率、优化方法、终止条件等等），使得训练神经网络变成了一门“艺术”。

- 如果要训练相对复杂的网络，需要大量的训练样本。

## 自编码器（Auto-encoder）

解决了（W，b）参数初始化问题。

##  卷积神经网络



# 案例

