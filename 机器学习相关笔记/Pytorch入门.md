# 概述

工业界推荐用Tensflow

学术界推荐用Pytorch

----

# 安装

看<a href="https://pytorch.org/get-started/locally/">官网</a>

----

# 基本操作

## 矩阵操作

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


## 基本运算操作

加减删除


```python
import torch
```


```python
a = torch.randint(1,10,(3,2))
b = torch.randint(1,10,(3,2))
print(a)
print(b)
```

    tensor([[8, 6],
            [8, 8],
            [3, 6]])
    tensor([[4, 9],
            [8, 6],
            [2, 3]])

```python
a + b
```


    tensor([[12, 15],
            [16, 14],
            [ 5,  9]])


```python
c = torch.add(a,b)
print(c)
```

    tensor([[12, 15],
            [16, 14],
            [ 5,  9]])

```python
# 这么麻烦做什么，直接a+b不好吗
result = torch.zeros(3,2)
torch.add(a,b,out=result)
```


    tensor([[12., 15.],
            [16., 14.],
            [ 5.,  9.]])


```python
a - b
```


    tensor([[ 9, -2],
            [-1, -1],
            [ 2, -3]])


```python
a // b
```


    tensor([[2, 0],
            [1, 1],
            [1, 2]])

矩阵运算


```python
torch.matmul(a,b.T) # b.T 转置  mat矩阵的缩写 mul乘法的缩写
```


    tensor([[ 86, 100,  34],
            [104, 112,  40],
            [ 66,  60,  24]])



