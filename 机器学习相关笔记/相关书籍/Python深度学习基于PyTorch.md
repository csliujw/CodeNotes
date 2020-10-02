# 第一章 Numpy基础

## 不熟悉的方法

### np.random中的方法

| 函数                          | 说明                    |
| ----------------------------- | ----------------------- |
| random.randint(1,20,10)       | 1-20,左闭右开，10个数据 |
| random.normal(0,1,size=(5,5)) | 0 1正态分布，5*5的矩阵  |
| random.randn(2,3)             | 2*3的标准状态分布       |
| random.randn(100,2,3)         | 100组2*3的标准状态分布  |
| random.shuffle(data)          | 打乱data数据            |

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
# 1000组2*3的矩阵
x_1 = r.random.randn(1000,2,3)
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

### 合并数组【处理数据常用 】

| 函数           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| np.append      | 内存占用大                                                   |
| np.concatenate | 没有内存问题,不会提升维数                                    |
| np.stack       | 沿着新的轴加入一系列数组，会提升维数，把传入的参数当整体哦！ |
| np.hstack      | 堆栈数组垂直顺序<span style="color:red">(行)</span>  horizontal 水平的英文，和concatenate类似 |
| np.vstack      | 堆栈数组垂直顺序<span style="color:red">(列)</span> vertical 垂直的英文，和concatenate类似 |
| np.dstack      | 堆栈数组按顺序深入（沿第三维）                               |
| np.vsplit      | 将数组分解成垂直的多个子数组列表                             |

**注意**

- append、concatenate以及stack都有一个axis参数，用于控制数组合并方式.
  - axis的意思是轴。
  - 向量默认为列向量
  - <span style="color:red">axis = 0为默认值，默认以列向量（y轴自顶向下合并）</span>
  - <span style="color:red">axis = 1，以行向量（x轴，自左向右合并）</span>
- 对应append和concatenate，待合并数组必须有相同的行数或列数（满足一个即可）
- stack、hstack、dstack 要求待合并的数组必须具有相同的形状（shape）

```python
import numpy as np

data1 = np.random.randint(0, 100, size=(2, 2))
data2 = np.random.randint(0, 100, size=(2, 2))
# print(data1, data2)
print("=====================\n")
# 直接追加，返回一个新数组，耗内存
# copy = np.append(data1, data2)
# print(copy)
# 测试发现 不是同一个对象
# print(id(copy), id(data2))

print("=" * 50)
print(data1, "\n===\n===\n", data2)
copy = np.concatenate((data1, data2), axis=0)

copy1 = np.stack((data1, data2), axis=0)

print("=" * 50)
# y轴，自顶向下 合并
print(np.vstack((data1, data2)))
# x轴，自左向右 合并
print(np.hstack((data1, data2)))

```

## <span style="color:red">批量处理</span>

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

