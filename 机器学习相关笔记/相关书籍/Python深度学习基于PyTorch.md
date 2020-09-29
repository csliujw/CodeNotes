# 第一章 Numpy基础

## 不熟悉的方法

### np.random中的方法

```python
import numpy as np

x = np.random.randint(10, 20, 10)
# replace 不可重复抽取
np.random.choice(x, size=(2, 2), replace=False)
x = np.random.normal(0, 1, size=(5, 5))
# numpy几个记得不是很清楚的方法
r = np.random
# 标准正态分布
x1 = r.randn(3, 2)
# 符合0 10 状态分布的 形状为3 2 的随机数
x2 = r.normal(0, 10, size=(3, 2))
# 打乱数据
r.shuffle(x2)
```

### 算数运算&矩阵运算

```python
data1 = np.random.randint(0, 10, size=(2, 2))
data2 = np.random.randint(0, 10, size=(2, 2))
print(data1)
print(data2)
# 数乘
print(data1 * data2)
print(np.multiply(data1, data2))
# 数加
data1 + data2
# 矩阵运算
print(np.dot(data1, data2))
```

### 数组变形

- reshape() 调整维度，不修改本身
- resize() 会修改本身
- ravel() 多维变一维，不修改本身
- flatten() 多维变一维，不修改本身

### 合并数组

- 1

```python

```

































----

# 第二章 PyTorch基础

## numpy与Tensor

很相似。numpy通过cup加速计算，Tensor通过GPU加速计算。

- Tensor为类 ==C

- tensor为方法==function

- 带下划线的方法会修改本身【这样命名清晰啊！】

  - ```python
    x = torch.tensro([2,3])
    y = torch.tensro([2,9])
    # 修改原数组
    x.add_(y)
    ```

> **注意点**

- torch.Tensor是torch.empty() 和 torch.tensor 之间的一种混合。全局默认使用dtype（FloatTensor），而torch.tensor是从数据中推断数据类型

- torch.tensor(1)返回一个固定值1，而torch.Tensor(1)返回一个大小为1的张量，它是随机初始化值。

  ```python
  print(torch.tensor(1).type())
  print(torch.Tensor(1).type())
  torch.LongTensor
  torch.FloatTensor
  ```

## Tensor不熟悉的操作

- numel(input) 计算Tensor的元素个数

- view(*shape) 共享内存，共享内存。。那他的表示岂不是只是修改了访问的索引，这样访问不是特别耗时吗

- view(-1) 展平数组

  ```python
  import torch
  
  x = torch.randn((3, 2))
  
  print(x.view(2,3))
  print(x.view(-1))
  
  x[1][1] = 10
  # output 发现共享内存
  print(x)
  print(x.view(3,2))
  ```

- item() 为单个变量则放回Python的标量



## 自动求导



 























----

# 第三章 PyTorch神经网络工具











----

# 第四章 PyTorch数据处理工具

