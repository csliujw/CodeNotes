# 概述

工业界推荐用Tensflow

学术界推荐用Pytorch

----

# LaTex

恶补一波LaTex语法，好写数学公式

<a href="https://zhuanlan.zhihu.com/p/95886235">LaTex语法</a>

----

# 安装

看<a href="https://pytorch.org/get-started/locally/">官网</a>

cuda的安装有点麻烦，请看这篇博客。cuda得版本一定要对上！

<a href="https://blog.csdn.net/marleylee/article/details/81988365?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param">这是MXNET安装CUDA的过程，按这个来就行。</a>

我是按上面那安装成功的。

如果一直试了好几次都失败了，那么大概率是电脑某些服务或驱动有问题【某些安全软件的锅】，建议直接重装电脑。

----

# 基本操作

## 矩阵操作

### 取随机矩阵

api的基本使用方式为：

- 创建各种随机数据  torch.methodName(行，列)

- 给定数据创建 torch.methodName([数据])

**常用api**

*表示元组

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

- y.add_(x) 创建的那些数据也是一个对象。

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

pytorch的数据类型。与numpy的array类似。


```python
import torch
a = torch.tensor([1,2,3],dtype=int)
print(a)
print(a.dtype)
# output
# tensor([1, 2, 3])
# torch.int64
```


定义二维数据并查看数据类型


```python
tensor = torch.tensor([[1,2,3],[4,5,6]])
print(tensor.shape)
print(tensor.size())
# torch.Size([2, 3])
# torch.Size([2, 3])
```


查看维度


```python
print(tensor.ndim)
# output
# 2
```


生成数据


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


查看形状并修改


```python
print(b.shape)
print(b.size())
# output
# torch.Size([3, 4])
# torch.Size([3, 4])
```

重新调整大小

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


tensor数据变为python数据


```python
d[0][0].item() # 只能转换单个值
# output
# 0.46197234568033185
```

np和pytorch之间的矩阵转换


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

# 参考步骤

## 文字描述

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

## 代码举例

> **数据处理部分**

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
```

`torch.utils.data.DataLoader`

```python
# 假设我们把数据集 和 标签 放在了一起，这样我们就可以十分方便地同时遍历数据集和标签了
data_items = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for label,data in data_items
	print(label,data)
    break # 只输出一组
```

> **构建模型**

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

> **定义代价函数和优化器**

```python
criterion = torch.nn.BCELoss(reduction='sum') # 代价函数
optimizer = torch.optim.Adam(model.paramenters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False) # 优化器
```

> **构建训练过程**

----

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

> **具体案例**

流程图如下

```flow
flow
st=>start: Prepare dataset
op1=>operation: Design model using class
op2=>operation: Construct loss and optimizer
op3=>operation: Training cycle
e=>end
st->op1->op2->op3->e
```



仅是梳理过程，不要在意具体的细节。

```python
import torch
from torch.nn import Module

# 步骤一 预定义数据
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 步骤二 Design model using class
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

----

# 流程演变

## Rule-based systems

**人工设计**

```flow
flow
st=>start: start
op0=>operation: input
op1=>operation: Hand-designed program
op2=>operation: output
e=>end
st->op1->op2->e
```



## Classic machine learning

```flow
flow
st=>start: start
op0=>operation: input
op1=>operation: Hand-designed features
op2=>operation: Mapping from features
op3=>operation: output
e=>end
st->op1->op2->op3->e
```

---

```flow
flow
st=>start: start
op0=>operation: input
op1=>operation: features
op2=>operation: Mapping from features
op3=>operation: output
e=>end
st->op1->op2->op3->e
```





## Deep learning

多了一个提取特征的过程。

传统的机器学习方式feature是单独训练的。

深度学习中，训练过程是统一的。【端-->端的训练过程】

```flow
flow
st=>start: start
op0=>operation: input
op1=>operation: simple features
op2=>operation: additional layers of more abstract features
op3=>operation: mapping from features
op4=>operation: output
e=>end
st->op1->op2->op3->op4->e
```

方式在逐渐演变

----

# 基本概念普及

反向传播：求偏导数。神经网络层数过高的话，求导很麻烦，故采用反向传播。反向传播的核心是计算图。

就是链式法则。

深度学习的模型很多，不必部学。学其精髓，即如何构造模型/构造模型的套路。可以把某些模型当作基本块，用这些基本块进行组装。

CUDA的安装，如果安装过vs 可能会出错，所以cuda建议选择自定义安装，不要勾选vs支持。

----

# 线性模型

主要是熟悉API的使用，具体理论方面去看书。

- 准备数据集
- 选择模型（根据数据集进行选择）
- 训练（个别可以不做训练，eg：KNN）
- inferring，看模型的效果

有标签：监督学习

无标签：无监督学习

有些有，有些无：半监督学习。

判断模型的优劣

将数据分为测试集和训练集，进行模型训练和模型效果检测。关于测试集和训练集的划分看书。数据集尽量和真实的环境保持一致。这让我想起了NLP中语料库的重要性。

随机猜测权重，对权重进行评估，然后重新调整。

评估找到的模型与数据集的误差有多大。这种评估模型称之为<span style="color:red">损失</span>
$$
损失函数
loss = (\hat{y}-y)^2 = (x * \omega - y)^2
$$
y_hat 是预测值

ω是权重。

对于整个训练集，我们需要把累加每一个样本的损失函数求均值。
$$
cost = \frac{1}{N}\sum^{n}_{1\to n} (\hat y_n - y_n)^2
$$
根据不同的权重计算MSE 平均平方误差 ，选取最优的ω。通过穷举来做【准确的来说，是通过类似于二分的方式来做吧】。

视频里真的是定义好步长，穷举。

请通过以下数据集，和以下方程计算出一个较优的ω

```shell
x_data = [1,2,3,4]
y_data = [2,4,6,8]

y = x * ω
loss = （y_hat -y）^2
# 拓展 增加一个b（bais）
```

训练的结果通过可视化工具，直观的展示训练效果，看那个区间最优，继续缩小范围进行调参【论可视化工具的重要性 visdom】

深度学习的代码 跑的周期可能很长。怕程序奔溃，丢失训练的部分结果，最好是存盘。

----

# 梯度下降

梯度：求偏导。
$$
\frac{\partial cost}{\partial ω}
$$

$$
ω = ω - a\frac{\partial cost}{\partial ω}
$$

a是学习率

梯度下降：朝梯度下降的地方走，片面来说就是朝导数下降的地方走。

梯度下降 局部最优。如何尽量避免这种片面的情况？

随机取起始点，梯度下降，取最优的。

----

# PyTorch实现线性回归

## 步骤

- 准备数据
- 设计模型，计算y_hat
- 构造损失函数和构造器
- 训练周期
  - forward：算损失
  - backward：算损失的梯度
  - update：更新权重

$$
\hat y = ω*x + b
$$

​				需要确定 ω和b的大小

```python
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的构造
        super(LinearModel, self).__init__()
        # torch.nn.Linear(1, 1) 构造一个对象，包含权重和偏置
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 实现了一个__call__
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# 标准
criterion = torch.nn.MSELoss(size_average=False)
# 优化器   lr是学习率，应该是每次梯度下降的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    y_pred = model(x_data) # 模型
    loss = criterion(y_pred, y_data) # 损失
    print(epoch, loss.item())

    # 更新的时候需要相应的优化器
    optimizer.zero_grad()
    # 反馈 得到梯度 然后进行优化
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.tensor([[4, 0]])
y_test = model(x_test)
print('y_pred', y_test.data)


# * 元组
# ** 字典
def func(*args, **kwargs):
    print(args)
    print(kwargs)
```

## 再次强调！

**步骤**

- 准备数据
- 定义模型
- 计算损失
- 优化

---

# Pytorch实现逻辑斯蒂回归

```python
import torchvision # 集成了常用的那些数据集

# 从里面找数据集 root应该是下载的文件存放在哪里
tran_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=True)
```

## Logistic Regression Model

$$
\hat y = σ(x*ω+b)
$$

加了一个xx函数（名字忘了哈哈哈），让值分布在0-1之间

- Loss Function for Linear Regression
  $$
  loss = (\hat y - y)^2 = (x*ω-y)^2
  $$

- Loss Function for Binary Classification
  $$
  loss = -(y log\hat y + (1-y)log(1-\hat y))
  $$
  
- 优化器公式【梯度下降】
  $$
  ω = ω - a \frac {\partial cost}{\partial ω}
  $$
  

```python
import torch
import torch.nn.functional as F
import torch.nn.modules

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

class LogisiticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisiticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisiticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1200):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()  # 清理
    loss.backward()
    optimizer.step()
```

基本就是这四个步骤。

- 准备数据
- 选择模型
- 构造loss 和 优化器
- 不断迭代

```text
损失最低 就是效果好

一般  实验 要看指标

损失就是通过计算  准确呢  可以作为参考

越不准确  loss越大

实验中  效果最好也不是看你的训练数据

你要用验证数据 去验证

不然损失很低  训练时候过拟合  但是 在测试集上效果非常差

一般是 训练集 训练  指标看验证集的

同时看
```

---

# Pytorch处理多维特征输入

```python
import torch.nn.modules
import numpy as np

"""
测试代码
result = np.random.randint(low=0, high=100, size=25).reshape((5, 5))
print(result)
print(result[:, [-1]])
print("\n\n\n\n")
"""

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
# 这个数据集最后一列是y的取值
# 每行都选  列的话 从0-最后一列，不包括最后一列
x_data = torch.from_numpy(xy[:, :-1])
# 每行都选，但是只取最后一列
y_data = torch.from_numpy(xy[:, [-1]])

# print(x_data)
# print("\n\n")
# print(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # in_features: int, out_features: int
        # 8是特征的数量  1时输出结果的数目
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.010)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 更新
    optimizer.step()

"""
多个特征
公式就变为了向量之间的操作了
$$
\hat y = σ(x^{(i)}*ω + b)
$$
if __name__ == '__main__':
    x = torch.tensor([1, 2, 3, 4])
    w = torch.tensor(3)
    result = x * w
    print(result)
    # output
    # tensor([ 3,  6,  9, 12])
"""
```

---

# Dataset&Dataloader

## API概述

```python
DataLoader: batch_size=2, shuffle=True
# batch_size = 2 2个数据分为一组
# shuffle 打乱数据。
```

## 定义自己的Dataset

```python
# 来看一手源码注释
r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        num_workers (int, optional): how many subprocesses to use for data
"""
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None):
```

## 代码

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # in_features: int, out_features: int
        # 8是特征的数量  1时输出结果的数目
        self.linear1 = torch.nn.Linear(8, 6)  # 第一层
        self.linear2 = torch.nn.Linear(6, 4)  # 第二层
        self.linear3 = torch.nn.Linear(4, 1)  # 第三层
        self.sigmoid = torch.nn.Sigmoid()  # 阶跃函数

    def forward(self, x):
        # 一层一层正向传播
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 自定义的Dataset需要几次 torch的Dataset
class DiabetesDataset(Dataset):
    # 初始化数据集
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = xy[:, :-1]
        self.y_data = xy[:, [-1]]

    # The expression dataset[item] will call this magic function
    # 得看看python的高級语法了····
    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.010)

dataset = DiabetesDataset('diabetes.csv.gz')
# dataset 加载的数据集
# batch_size 多少数据为一个batch
# shuffle 是否打乱原始的数据
# num_workers 使用多少子程序处理数据
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

if __name__ == '__main__':
    for epoch in range(100):
        # 补充 enumerate语法
        # debug 看看
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

----

# 多分类问题

## Softmax概述

softmax

多分类要输出多个。二分类输出两个即可。 

有10个分类，有9个参数就够了。第10个分类用1-前面的求和，但是这样不利于并行计算。

<img src="..\pics\pytorch\softmax01.png">

归一化？

<img src="..\pics\pytorch\softmax02.png">

<img src="..\pics\pytorch\softmax03.png">

激活的操作，在交叉熵损失中完成了。

<img src="..\pics\pytorch\cross01.png">

<img src="..\pics\pytorch\cross02.png">

**交叉熵损失demo测试**

```python
import torch

criterion = torch.nn.CrossEntropyLoss()
y = torch.LongTensor([2, 0, 1])

y_pred1 = torch.Tensor([[0.1, .02, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])
# 1的loss是比较小的 2的loss是比较大的
l1 = criterion(y_pred1, y)
l2 = criterion(y_pred2, y)
print(l1, l2)
```

## MNIST

拿到图像后先把他变成Tensor数据类型

<img src="..\pics\pytorch\soft_pre_data.png">

- 1x28x28 1维 28*28 == 把三阶张量变成一阶，一张图片一共784个像素数。
- ToTensor 把图像变成Tensor类型
- Normalize(mean,σ) 让数据满足0 1 分布

<img src="..\pics\pytorch\design_model01.png">

<img src="..\pics\pytorch\design_model02.png">

<img src="..\pics\pytorch\clao.png">

<img src="..\pics\pytorch\train_and_test.png">

---

晚上把代码捋一捋。

# 卷积神经网络

## CNN基础

由线性层连接起来的叫全连接的网络。

CWH

- C ==> Channle
- W ==> Width
- H ==> High

卷积神经网络保留了原始的空间信息。经过卷积后，得到的图像其通道会变，图像的高度和宽度也会变（高宽也可以不变）。做一个2*2的下采样后，通道数不变，但是图像的宽高会变。做下采样是为了降低计算要求。

<img src="..\pics\pytorch\a_simple_convolution_neural_network.png">

---

RGB 通道 有三个channle。

以下的为 3xWxH

<img src="..\pics\pytorch\cnn02.png">

做数乘！每个通道配一个核！

 <img src="..\pics\pytorch\cnn03.png">

<img src="..\pics\pytorch\cnn04.png">

---

## CNN高级

### GoogLeNet

想减少代码冗余，OOP！代码复用！

抽取出公共的内容。

1*1的卷积改变通道数。

<img src="..\pics\pytorch\GoogleNet01.png">

什么是1*1的卷积？

CWH = 100 * 500 * 500

filter = 20，filter的通道数要和输入的通道数一样。

一个filter相乘相加就是1 * 500 * 500

二十个就是 20 * 500 * 500.

---

### Implementation of Inception Module

<img src="..\pics\pytorch\GoogleNet02.png">

算出各个走法的通道后，再进行通道的拼接。

<img src="..\pics\pytorch\GoogleNet03.png">

<img src="..\pics\pytorch\GoogleNet04.png">"



# 循环神经网络

## 基础

## 高级

