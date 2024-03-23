# NumPy

NumPy 是使用 Python 进行科学计算的基础软件包，可以十分方便的进行统计、切片和索引，并且具备强大的线性代数、傅立叶变换和随机数功能。

其中，类似于 NumPy 的切片和索引功能在机器学习和深度学习中是经常需要用到的。PyTorch tensor 的功能就和 NumPy 类似。

## 快速入门

NumPy 的数据类型我们称之为 ndarray（n-dim array）多维数组。下面是一个创建 n-dim-array 的 numpy 代码示例

```python
import numpy as np

array = np.array([
                    [1,2,3],
                    [4,5,6],
                    [7,8,9]
                ])
print(array)
# 下面是 ndarray 的一些属性，基本上见名知意。
print(array.ndim) # 维度 [[ 二维]] 看括号！！
print(array.shape) # 形状
print(array.size) # 形状
print(array.dtype) # 元素类型 dtype data type
```

### 创建

ndarray 数组的创建方式有很多种，下表列出了多种常见 ndarray 的方式。

| 创建方式                             | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| np.array([1,2,3,4,'abc'])            | 根据列表中的内容创建 array，由于 array 中的数据类型要一样，如果不一样会发生强转。<br>强转规则如下：str > float > int |
| np.ones()                            | 创建一个全 1 的 array                                        |
| np.zeros()                           | 创建一个全 0 的 array                                        |
| np.full()                            | 创建一个为指定数值的 array                                   |
| np.eye()                             | 创建一个单位矩阵                                             |
| np.linspace(1, 10, 30)               | 创建一个等差 array                                           |
| np.arange()                          | 创建指定范围的 array                                         |
| np.random.randn()                    | 创建一个符合标准正态分布的 array                             |
| np.random.randint(1, 10, size=(3,4)) | 创建一个数值在指定范围的，大小为指定 shape 的 array          |
| np.random.normal()                   | 创建一个符合正态分布的 array                                 |
| np.random.rand()                     | 0~1 的随机数                                                 |
| np.random.random()                   | 0~1 的随机数                                                 |

经验之谈：上述创建 array 的方法多用于创建 fake data 模拟如 `机器学习 / 深度学习` 模型的输入数据，用于快速测试模型是否能正常运行。

如利用 np.random.randint() 创建一个大小为 $1*3*224*224$ 的数组

```python
np.random.randint(low=0,high=255,size=(1,3,224,224))
```

<b>创建数组时指定元素类型</b>

```python
import numpy as np
a = np.array([1,2,3],dtype=np.int32) # 指定元素数据类型

b = np.array([1,2,3],dtype=np.float) # 指定元素数据类型
```

<b>创建元素全为 0 的数组</b>

```python
zero = np.zeros((2,3)) # 生成2行3列全为0的矩阵
empty = np.empty((3,2)) # 生成3行2列 接近0的矩阵
```

<b>创建元素全为 1 的</b>

```python
one = np.ones((3,4)) # 生成3行4列全为1的矩阵
```

<b>生成 0~10 的数组，并将其改为 $2*5$ 的二维数组</b>

```python
e = np.arange(10) # output [0 1 2 3 4 5 6 7 8 9]

h = np.arange(10).reshape(2,5) # 转换形状【重新调整形状】 2行4列
```

> <b>习题</b>

1、创建一个长度为 10 的一维全为 0 的 ndarray 对象，然后让第 5 个元素等于 1

```python
data = np.zeros(shape=10)
data[4] = 1
display(data)
```

2、创建一个元素为从 10 到 49 的 ndarray 对象

```python
# data = np.array(range(10,50))
data = np.arange(10,50,1)
display(data)
```

3、使用 np.random.random 创建一个 10*10 的 ndarray 对象，并打印出最大最小元素

```python
data = np.random.random(size=(10,10))
data.min(), data.max()
```

4、创建一个每一行都是从 0 到 4 的 5*5 矩阵

```python
data = np.zeros(shape=(5,5))
# 就是每一行的元素都变成 [0,1,2,3,4]
data[:,:] = np.arange(0,5)

# 利用 full 直接创建
data = np.full(shape=(5,5),fill_value=range(5))
```

### 访问

numpy 中数据的访问方式和列表的访问方式类似，都是使用索引下标获取/修改数据，并且，与 Python 的原生列表相比，numpy 具备更加强大的切片功能。

<b>访问并修改 numpy 中的元素</b>

```python
import numpy as np
data = np.arange(0,10)
data[8] = 111
```

<b>切片访问</b>

```python
data = np.zeros(shape=(10,10))
data[2,:] = range(10,20) # 修改第二行的数据为 10~19
data[2,:]
```

<b>访问多行数据</b>

```python
data = np.zeros(shape=(10,10))
# 获取第 3 行和第 6 行的数据
data[[2,5],:]
# 给第 3 行和第 6 行的数据重新赋值，赋值为 0~9
data[[2,5],:] = range(10)
```

<b>交换 data 中第 2 行和第 5 行的数据</b>

```python
data = np.zeros(shape=(10,10))
data[2,:] = range(10)
data[5,:] = range(20,30)
# 交换数据
data[[2,5],:] = data[[5,2],:]
```

### 切片⭐

<b>numpy 中的切片 -- 以图片为例</b>

假定有一个三维的 ndarray（$3*224*224$）data（图片）， numpy 中的切片方式如下表

| 切片方式              | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| `data[0]`             | 访问第 0 个维度对应的数据（$1*224*224$），即第一个通道的所有像素 |
| `data[1,1,1]`         | 访问 1,1,1 索引对应的像素                                    |
| `data[:,0:100,0:100]` | 在第 2 和第 3 维度进行数据筛选，筛选的范围是 0~100           |
| `data[data<50] = 0`   | 布尔切片，data 中所有小于 50 的像素都赋值为 0                |
| `data[:,:,::-1]`      | 就是将 n 中第三个维度的元素顺序完全反转                      |

在机器学习和深度学习中最常用的切片方式包括：布尔切片、索引切片。

> <b>习题</b>

1、创建一个元素为从10 到 49 的 ndarray 对象并将其反转

```python
np.arange(10,50,1)[::-1]
```

2、创建一个 10*10 的 ndarray 对象，且矩阵边界全为 1，里面全为 0

```python
data = np.zeros(shape=(10,10))
data[0,:] = 1 
data[-1,:] = 1
data[:,0] = 1
data[:,-1] = 1
```

3、创建一个 $3*3$ 的数组，交换第一行和第二行的数据

```python
data = np.random.randint(low=0,high=5,size=(3,3))

data[0,],data[1,] = data[1,],data[0,]

# 也可以这样，trick 学过就会了 ',' 可以省略
data[[0,1],] = data[[1,0],]
```

4、给定数组 [1,2,3]，在每个元素之间插入 3 个 0 后的新数组

```python
data = np.array([1,2,3])
new_data = np.zeros(shape=9)
new_data[::4] = data
```

### reshape⭐

reshape 是重新组织数据，修改数据的 shape。

```python
n = np.arange(1, 21)
n.shape # 20
# 变形:满足元素个数不不变
n.reshape(4, 5)
n.reshape(2, 2, 5)
```

在机器学习和深度学习中，reshape 这个 api 也使用的非常频繁。

### 广播机制

numpy 的运算存在一个广播机制，如果两个 ndarray 的 shape 不一样，会先尝试让它们变成一样的形状然后再做运算（numpy 做了运算符重载）

<b>【重要】ndarray 广播机制的两条规则</b>

- 规则一：为缺失的维度补维度
- 规则二：缺失元素用已有值填充

data.shape = [1,3]; data2.shape=[3,1] 两者相加时会先进行广播，都广播成 $3*3$ 的形状。

```python
data1 = np.random.randint(low=0,high=10,size=(1,3))
data2 = np.random.randint(low=0,high=10,size=(3,1))
data1 + data2
```

### 数学运算

<b>加减乘除</b>

都是相同位置的进行操作，即按位相除。

```python
a = np.array([[1,2],
             [3,4]])
b = np.array([[1,2],
             [3,4]])
a + b
a - b
a * b
a / b
a ** b  # [次幂b中的数为a的次方]
a % b # a中对应的元素 对b中对应元素取余
a // b # 取整
a + 2 # a中的所有元素+2 减乘除都是一样的。
a > 3 # a中的所有元素与3比较，返回值是一个bool 矩阵
```

<b>矩阵运算</b>

```python
np.dot(arr1,arr2) # 矩阵乘法
# or
arr1.dot(arr2)
# or 可用 @ 进行矩阵乘法 py3.5及以上版本
a @ b 

arr1.T # 转置矩阵
np.transpose(arr1) # arr1的转置矩阵

# 其他
np.exp(2) # e^2 e的平方
```

矩阵求行列值、求秩、求逆 (很少用到)

```python
# 方阵: 3行3列 
# 阶数: 3阶
n = np.array([[1, 2, 3], 
              [2, 3, 4],
              [4, 5, 7]])

# 主对角线: 1*3*6 + 2*4*4 + 3*2*5 = 18 + 32 + 30 = 80
# 副对角线: 3*3*4 + 2*2*6 + 1*4*5 = 36 + 24 + 20 = 80
# 行列式的值 : (主对角线 -  副对角线) = 80 - 80  = 0

# 线性代数
np.linalg.det(n)   # 行列式的值 : (主对角线 -  副对角线)

# 矩阵的逆: 数学中倒数
# np.linalg.inv(n)

# 秩: 最高阶非零子式的阶数
np.linalg.matrix_rank(n)

# 满秩矩阵: 秩 == 矩阵的阶数 3
# 降秩矩阵(奇异矩阵): 秩 < 矩阵的阶数 3

np.linalg.inv(n)
```

<b>其他数学运算</b>

abs、sqrt、square、exp、log、sin、cos、tan、round、ceil、floor、cumsum

```python
np.abs(-10)  # 10
np.sqrt(25)  # 5
np.square(25)  # 625

np.exp(2)  # e=2.718281828459045
np.e

# 对数:
#    底数,真数
np.log(np.e)  # ln(e)

# π = 3.141592653589793
np.pi
np.sin(3)
np.cos(3)
np.tan(3)

np.round(3.4567)
np.round(3.4567, 2)  # 四舍五入

np.ceil(3.1)  # 向上取整
np.floor(3.9)  # 向下取整

# cumsum(): 累计
n = np.array([1, 2, 3, 4, 5, 6, 7])
np.cumsum(n) # array([ 1,  3,  6, 10, 15, 21, 28])
```

### 随机数

| 方式                                 | 说明                                                |
| ------------------------------------ | --------------------------------------------------- |
| np.linspace(1, 10, 30)               | 创建一个等差 array                                  |
| np.arange()                          | 创建指定范围的 array                                |
| np.random.randn()                    | 创建一个符合标准正态分布的 array                    |
| np.random.randint(1, 10, size=(3,4)) | 创建一个数值在指定范围的，大小为指定 shape 的 array |
| np.random.normal()                   | 创建一个符合正态分布的 array                        |
| np.random.rand()                     | 0~1 的随机数                                        |
| np.random.random()                   | 0~1 的随机数                                        |

<b>生成从 0-1 的随机数</b>

```python
import numpy as np
sample = np.random.random((3,2)) # 3行2列
print(sample)
```

<b>生成符合标准正态分布的随机数</b>

```python
sample2 = np.random.normal(size=(3,2))# 3行2列
```

PS：标准正太分布 N(0，1)； 期望=1，方法差=1

<b>生成指定范围的int类型的随机数</b>

```python
sample3 = no.random.randint(0,10,size=(3,2)) # 3行2列 整数
```

<b>生成等份数据</b>

```python
np.linspace(0,2,9) # 9个数组 从0-2中等份取
```

----

### 元素求和⭐

numpy 中的求和 api 有两种 np.sum 和 np.nansum（nan: not a number） 

- nan：数值类型，not a number 不是一个正常的数值，表式空
- np.nan：float 类型

```python
n3 = np.array([1, 2, np.nan, 4, 5])
display(n3)  # array([ 1.,  2., nan,  4.,  5.])
display( np.sum(n3) )  # nan
display( np.nansum(n3) )  # 12.0
```

需要注意的是 sum 求和的时候可以指定求和的维度，请看下面的例子

- axis=0 表示沿着列的方向，做逐行的操作 ==> 对行做操作
- axis=1 表示沿着行的方向，做逐列的操作 ==> 对列做操作

```python
n2.sum()
np.sum(n2)

np.sum(n2, axis=0)  # 行,  行和行之间求和

np.sum(n2, axis=1)  # 列,  列和列之间求和
```

<b>对每一列求和</b>

你可以这样认为，numpy 中向量默认是列向量，axis=0 即对默认的向量求和

```python
np.sum(sample,axis=0)
# eg：
data = np.array([[1,2,3,4],
                 [5,6,7,8]])
np.sum(data,axis=0)
# output
# [6 , 8 , 10 , 12]
```

<b>对每一行求和</b>

```python
np.sum(sample,axis=1)
# eg：
data = np.array([[1,2,3,4],
                 [5,6,7,8]])
np.sum(data,axis=0)
# output
# [10 , 26]
```

### 其他常见聚合函数

| 函数          | 说明             |
| ------------- | ---------------- |
| np.min⭐       | 最小值           |
| np.max⭐       | 最大值           |
| np.mean       | 平均值           |
| np.average    | 平均值           |
| np.median     | 中位数           |
| np.percentile | 百分位数         |
| np.argmin     | 最小值对应的下标 |
| np.argmax⭐    | 最大值对应的下标 |
| np.std        | 标准差           |
| np.var        | 方差             |
| np.power      | 次方，求幂       |
| np.argwhere⭐  | 按条件查找       |

```python
np.min(n2)
np.max(n2)

np.mean(n2)
# np.average(n2)

np.median(n1)
# np.percentile(n1, q=50)  # q= 0~100 

np.argmin(n2)
np.argmax(n2)

np.std(n1)  # 标准差
np.var(n1)  # 方差

np.power(3, 2)
3**2
pow(3, 2)

n3 = n2.reshape(-1)
print(n3)
display(n3)
ret = np.argwhere(n3 == 4)
display(ret)
display(ret.reshape(-1))
```

### 拼接操作⭐

拼接操作在深度学习中使用的频率非常高~ numpy 中常见的拼接操作如下表，假定 arr.shape = (3,3)

| 方法                        | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ |
| np.concatenate()            | 参数是列表或元组<br>级联的数组维度必须相同<br>可通过 axis 参数改变级联的方向 |
| np.hstack                   | 水平级联                                                     |
| np.vstack                   | 垂直级联                                                     |
| np.expand_dims(arr, axis=0) | 新数组 shape 为 （1，3，3）                                  |

会用 `np.concatenate()` 即可

```python
n1 = np.random.randint(1, 10, size=(3, 5))
n2 = np.random.randint(1, 10, size=(3, 5))

np.concatenate( (n1, n2) )  # 上下合并,垂直合并,默认
np.concatenate( (n1, n2), axis=0 )  # axis: 轴,表示第几个维度,从0开始, 0表示行,1表示列 shape=(6,5)

# 左右合并,水平合并: axis=1
np.concatenate( (n1, n2), axis=1 )  # axis: 轴,表示第几个维度,从0开始, 0表示行,1表示列 shape=(3,10)
```

<b>垂直合并【垂直拼接】<span>vstack  vertical stack【垂直】</span></b>

垂直的方式叠起来

```python
import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr3 = np.vstack((arr1,arr2))
# output
# [1 2 3]
# [4 5 6]
```

<b>水平合并【水平拼接】hstack horizontal【水平】</b>

水平方向叠起来

```python
np.hstack((arr1,arr2))
# output
# [1 2 3 4 5 6]
```

<b>新增维度</b>

```python
arr = np.array([1,2,3,4])
arr_1 = arr[np.newaxis,:] # 新增一个维度
```

把行向量变成列向量

```python
arrs = np.array([1,2,3])
arrs_1 = arrs[np.newaxis,:]
arrs_1.T
```

<b>维度扩展</b>

常用的 API 还是 `np.expand_dims`

```python
arrs_2 = np.atleast_2d(arrs) # 如果低于2d（2 dim）则会扩充为2dim 反之不改变
# 这个常用哦
arr = np.array([1, 2, 3])
arr = np.expand_dims(arr, axis=0)
print(arr)
```

----

### 分割

<b>水平分割</b>

```python
import numpy as np
arr1 = np.arange(12).reshape((3,4))
print(arr1)
arr2,arr3 = np.split(arr1,2,axis=1) # 水平分割 分2份
```

<b>垂直分割</b>

```python
arr4,arr5 = np.split(arr1,3,axis=0) # 垂直方向 分3份
```

矩阵中的向量一般默认为列向量。所以axis默认为0，垂直方向分割。

<b>无法等份切割</b>

```python
arr6,arr7,arr8 = np.array_split(arr1,3,axis=1) # 水平切割 分三份 不等份分割
```

----

### 拷贝⭐

numpy 可以进行深拷贝，使用 copy 方法即可，拷贝出来的数据和原数据使用不同的内存，互不干扰。

```python
import numpy as np
data = np.array([1,2,3])
data_copy = data.copy() # 是两个完全独立的数据，互不影响
```

### 排序

np.sort() 与 ndarray.sort() 都可以，但有区别：
- np.sort() 不改变输入
- ndarray.sort() 本地处理，不占用空间，但改变输入

默认是使用快速排序，可以自己指定排序规则。

```python
n = np.array([1, 5, 4, 88, 77, 6, 99, 2, 3, 4])
# 会直接对原数组排序
n.sort()
```

```python
n = np.array([1, 5, 4, 88, 77, 6, 99, 2, 3, 4])
n2 = np.sort(n)
display(n, n2)
```

# maplotlib

主要用 pyplot 包

### 入门案例

> **生成数据**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,100) # 生成从-1 ~ 1的100个数据

y = 2*x + 1

plt.plot(x,y)
plt.show()
```

> **绘制图像figure**

```python
x = np.linspace(-1,1,100) # 生成从-1 ~ 1的100个数据

y1 = 2*x + 1
y2 = x**2 +1

# 绘制的第一个图像
plt.figure()
plt.plot(x,y1)

# 绘制的第二个图像
plt.figure(figsize=(8,5)) # 设置要创建的图像的大小
plt.plot(x,y2,linewidth=1.0,linestyle='--')

plt.show()
```

两个函数画在同一个坐标系，标明颜色，线条样式。

```python
x = np.linspace(-1,1,100) # 生成从-1 ~ 1的100个数据

y1 = 2*x + 1
y2 = x**2 +1
plt.figure()
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.plot(x,y2,color='black',linewidth=2.0,linestyle=':')
plt.show()
help(plt.plot) # 查看plt.plot的帮助文档
```

### 设置坐标轴

PS：四条边框就是四条脊梁。

限制坐标轴范围

```python
plt.xlim((-1,2))
plt.ylim((-5,10))
```

设置x y描述

```python
plt.xlabel('This is X')
plt.ylab('This is Y')

plt.plot(xxxx)
```

指定x y下方显示的尺度/用文字表示尺度

```python
plt.xticks([-1,-0.5,0,0.25,0.5,1])
# -1 0 1 2 3 与 leavel对应
plt.yticks([-1,0,1,2,3],['leave1','leave2','leave3','leave4','leave5'])
```

获取坐标轴

```python
ax = plt.gca() # gca get current axis
# 选择对应的脊梁 右边的脊梁设置为红色
ax.spines['right'].set_color('red')
ax.spines['top'].set_color('none') # top无边框
```

将坐标轴移至中间 spines：脊梁

```python
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') # 用下边的坐标轴代替x
ax.yaxis.set_ticks_position('left') # 用左边的坐标轴代替y
ax.spines['bottom'].set_position(('data',-1))
```

**移动脊柱文件**【菜鸟教程】

坐标轴线和上面的记号连在一起就形成了脊柱（Spines，一条线段上有一系列的凸起，是不是很像脊柱骨啊~），它记录了数据区域的范围。它们可以放在任意位置，不过至今为止，我们都把它放在图的四边。

实际上每幅图有四条脊柱（上下左右），为了将脊柱放在图的中间，我们必须将其中的两条（上和右）设置为无色，然后调整剩下的两条到合适的位置——数据空间的 0 点。

```python
...
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
```

### legned图例

> **基础代码**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2,2,100)

# 设置函数
y_1 = x*2
y_2 = x**2
axis = plt.gca() # 获取当前坐标轴

# 隐藏多余边框
axis.spines['top'].set_color('none')
axis.spines['right'].set_color('none')

axis.spines['bottom'].set_position(('data',0))
axis.spines['left'].set_position(('data',0))

# 注意有逗号
l1, = plt.plot(x,y_1,color='blue')
l2, = plt.plot(x,y_2,color='red')

plt.legend(handles=[l1,l2],labels=['111','222'],loc='best')
```

### 图像标注/annotation标注

- 画出函数
- 画出散点
- 根据散点画出垂直的线
  - `plt.plot([x0,x0],[y0,0],'k--',lw=2.5)`连接点`[x0,y0] [x0,0]`
    - k：黑色 -- 虚线
    - lw：线宽 2.5
  - 注解：`plt.annotate(r'$2x+1=%s' % y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30)，textcooords=’offset points‘，fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2))`
    - r‘ 是正则表达式
    - xy是xy的值
    - xytext 距离点的位置 x方向+30 y方向-30
    - connectionstyle：弧度，角度
  - 普通的文字注解
    - plt.text(x,y,r'$This is the some text. $'，fontdict={'size':16,'color':'red'})
    - 可以用数学公式，但是很麻烦

 ### tick 能见度

设置坐标轴数字的能见度

### 散点图

看API，实验楼实验

> **绘制散点图**

```python
import numpy as np
import matplotlib.pyplot as plt

N = 1024
X = np.random.normal(0,1,N)
Y = np.random.normal(0,1,N)

T = np.arctan2(Y,X) # 颜色的值

# s --> size ; c --> color  alpah --> 透明度
# plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.scatter(np.arange(5),np.rarange(5))
# plt.xlim((-1.5,1.5))
# plt.ylim((-1.5,1.5))

plt.xticks(())
plt.yticks(())
plt.show()
```

### 柱状图/直方图

> **基本用法**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y = 2**x + 10
plt.bar(x,y)
plt.bar(x,-y)
plt.show()
```

> **方法补充**

```python
seq1 = ['one', 'two', 'three']
seq2=[1,2,3]

list(zip(seq1,seq2))
# output [('one', 1), ('two', 2), ('three', 3)]

dict(zip(seq1,seq2))
# output {'one': 1, 'two': 2, 'three': 3}

list(zip(*zip(seq1,seq2)))
# output [('one', 'two', 'three'), (1, 2, 3)]
```

> **为柱状图标数值**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y = 2**x + 10

plt.bar(x,y)
plt.bar(x,-y)

for x,y in zip(x,y):
    plt.text(x,y,'%.2f' %y ,ha='center',va='bottom')
    plt.text(x,-y,'%.2f' %y ,ha='center',va='top')
    
plt.show()
```

---

# Pandas

对 numpy 进行了封装，Pandas 有两种数据类型，一维数组和维数组；二维数组用的更多，二维数组中包含行名和列名。

<span style="color:blue">Pandas 中涉及到删除或修改原数据的操作，一般都不会直接修改原数据。</span>

案例驱动学 API 更佳~

数据分析流程~

- 收集数据，加载数据
- 理解数据
  - 数据的大小，头尾数据，随机采样部分数据观察，数据的描述，有多少空值
  - shape、head、tail、sample、describe、info、isnull
- 数据清洗
  - 缺失值处理、重复值处理、异常值处理
- 数据分析 / 特征工程
  - 分析数据，如数据各个特征的相关性，剔除冗余的特征
- 算法训练
  - 选取合适的算法模型 ，集成学习
- 数据可视化

## 数据结构

Pandas 是基于 Numpy 的，用于数据分析，可对 csv tsv xlsx 等格式的数据进行处理和分析。Pandas 主要使用的数据结构为：Series 和 DataFrame 类

Pandas 中常见的知识点如下

- 排列
- 索引
- 交叉表
- 透视表
- 数据探索

<b>Series：类似于一维数组对象，由两个部分组成</b>

- values：一组数据（ndarray 类型）
- index：相关的数据索引标签（我们自定义的索引）， 类似于 dict 中的 key

```python
series = [1,2,3,41,123,123]

# output
# index  value
#   0      1
#   1      2
#   2      3
#   3     41
#   4    123
```

<b>DataFrame：类似于二维数据结构</b>

DataFrame 是一个【表格型】的数据结构，可以看做是【由 Series 组成的字典】（共用同一个索引）。DataFrame 由按一定顺序排列的多列数据组成。设计初衷是将Series的使用场景从一维拓展到多维。DataFrame 既有行索引，也有列索引。
- 行索引：index
- 列索引：columns
- 值：values（NumPy 的二维数组）

## 读写数据

Pandas 读取 csv 数据，并设置指定列的数据为索引

```python
import pandas as pd
# 设置第 0 列的数据为索引
pd.read('xx.csv', index_col=0, sep=',')
```

## Series

### 创建

<b>Serires 有两种创建方式</b>

- 由列表或 NumPy 数组创建
- 由字典创建

<b>列表或 NumPy 数组创建</b>

Series 默认索引为 0~N-1 的整数型索引

```python
s1 = pd.Series(np.array([10, 20, 30]),index=['a','b','c'])
s2 = pd.Series([10, 20, 30],index=['a','b','c'])
# 获取 a index 中的数据
s1['a'],s2['a']
```

Series 中的 index 和 values 可以直接获取

```python
s1.values, s1.index
```

<b>字典创建</b>

字典的 key 作为默认的 index，value 作为 Series 的值

```python
my_dict = {
    'a':10,
    'b':20,
    'c':30
}
# dict 的 key 作为自定义 index value 作为 Series 的 value
data = pd.Series(my_dict)
```

### 索引&切片

Series 可以通过<b>自定义的 index 获取数据</b>，也可以使用<b>数组本身的索引获取数据</b>。如果自定义的索引是数字，可能会出现冲突。

切片语法与 numpy 一致。

#### <b>索引</b>

可以使用中括号取单个索引（此时返回的是元素类型），或者中括号里一个列表取多个索引（此时返回的仍然是一个Series类型）。分为显示索引 [我们自定义的索引] 和隐式索引

<b>(1) 显式索引</b>

- 使用 index 中的元素作为索引值
- 使用.loc[]（推荐，可以避免 pd 的 index（自定义的索引）和数组的索引冲突）
- 如果企图使用数组默认索引访问数据请使用隐式索引 `.iloc`

```python
s = pd.Series(np.arange(10,15),index=list('ABCDE'))
s['A':'D'] # s.loc['A':'D'] 效果一样
s[0] # 爆出警告 In a future version, integer keys will always be treated as labels, To access a value by position, use `ser.iloc[pos]`
```

<b>(2) 隐式索引</b>

- 使用整数作为索引值
- 使用.iloc[]（推荐）

```python
s = pd.Series([10,20,30],index=[1,2,3])
# pure index(非自定义索引) access element
s[1] # 10 使用的自定义索引访问的
s.iloc[1] # 20 使用的数组索引访问的
```

#### 切片

- 支持 nmpy 方式的切片
- 也支持通过自定义索引进行切片（可能重写了 numpy 的 getitem 方法）

```python
s = pd.Series(np.arange(10,20),index=list('abcdefghjk'))
s['a':'g'], s[:7]
```

### 属性&方法

Series 包含下面的基本属性

| 属性   | 说明 |
| ------ | ---- |
| shape  | 形状 |
| size   | 长度 |
| index  | 索引 |
| values | 值   |
| name   | 名字 |

```python
s = pd.Series(np.arange(10,20),index=list('ABCDEFGHJK'))
s.shape
s.size
s.index # s.index = list('abcdefghjk') # 可以获取就可以修改
s.values
s.name = 'kkx'
s.name # Series 的 name 没什么用
```

<b>常用且实用的方法有</b>

| 方法   | 说明                      |
| ------ | ------------------------- |
| head() | 查看前几条数据，默认 5 条 |
| tail() | 查看后几条数据，默认 5 条 |

```python
s = pd.Series(np.arange(10,20),index=list('ABCDEFGHJK'))
s.head()
s.tail()
```

### 运算

适用于 NumPy 的数组运算也适用于 Series，也支持广播

```python
s1 = pd.Series(np.arange(10,15),index=list('ABCDE'))
s1.sum()
s+1
```

<b>Series 之间的运算</b>

- 在运算中自动对齐索引, 索引名对应的数据相加
- 如果索引不对应，则计算结果为 NaN (两个 Series 之间如果存在不对应的索引，则不对应索引数据的计算结果是 NaN)

```python
s1 = pd.Series(np.arange(10,12),index=list('AB'))
s2 = pd.Series(np.arange(100,120,10),index=list('BC'))
s1+s2
"""
A      NaN
B    111.0
C      NaN
dtype: float64
"""
```

- add 方法，索引不对应的数据，如果也希望保留它们的值，可以用 add 方法进行计算，索引不对应的数据可以与指定的 fill_value 进行加法计算

```python
s1.add(s2,fill_value=0) # 索引不对应的数据用 fill_value 计算

"""
A     10.0
B    111.0
C    110.0
dtype: float64
"""
```

## DataFrame⭐

DataFrame 是一个【表格型】的数据结构，可以看做是【由 Series 组成的字典】（共用同一个索引）。DataFrame 由按一定顺序排列的多列数据组成。设计初衷是将 Series 的使用场景从一维拓展到多维。DataFrame 既有行索引，也有列索引。
- 行索引：index
- 列索引：columns
- 值：values（NumPy 的二维数组）

<b>注意：</b>DataFrame 是优先列的，更关心的是一列一列的数据，因此直接对 DataFrame 进行切片是<span style="color:blue">默认是先对列做切片</span>，再对行做切片；若希望像 numpy 一样操作需要使用 DataFrame 的 `iloc/loc` 方法。

### 创建

最常用的方法是传递一个字典来创建。DataFrame 以字典的键作为每一列的名称，以字典的值（一个数组）作为每一列的值。

此外，DataFrame 会自动加上每一行的索引（和 Series 一样），同 Series 一样，若传入的列与字典的键不匹配，则相应的值为 NaN。

```python
# 字典的 key 作为列名，字典的 value 作为列的值
my_dict = {
    'name':['tom','jerry'],
    'age':[8,7]
}
df = pd.DataFrame(my_dict)
```

通过 list / numpy 创建

```python
# 3 行 5 列
df2 = pd.DataFrame(
    [[1,1,1,1,1], # 行
     [2,2,2,2,2], # 行
     [3,3,3,3,3]])# 行
df2.index=['A','B','C']
df2.columns = ['Java','Python','C Plus Plus','C#','js']
```

### 修改/新增列

Pandas 列的新增和修改方式和字典新增/修改元素的方式一样

```python
df2 = pd.DataFrame(
    [[1,1,1,1,1], # 行
     [2,2,2,2,2], # 行
     [3,3,3,3,3]])# 行

df2.index=['A', 'B', 'C']
df2.columns = ['Java', 'Python', 'C Plus Plus', 'C#', 'js']

# 有 Kotlin 这列则为修改，没有则为删除
df2['Kotlin'] = [4,4,4]
# OK
df2['js'][df2['js']>2]=100
# 但是不推荐，3.0 默认用 Copy-On-Write A typical example is when you are setting values in a column of a DataFrame, like:
# df["col"][row_indexer] = value
```

### Series中的_mgr

DataFrame 是由 Series 组成的，Series 中的数据存储在哪里呢？存储在 `_mgr` 中，我们修改 Series 时，其实修改的是 `_mgr`。知道了这个，我们再来看 CoW 技术。

### Copy-On-Write

关于直接使用 DataFrame 切片修改元素的问题

- 官方文档：在 Copy-On-Write 模式下，这种是无法修改原视图的

Copy-On-Write：在我尝试修改数组的时候，我会先复制一份副本，然后修改副本，并不会修改原视图。

#### Java中的CoW

Java 中 Copy-On-Write 的典型实现

```java
public boolean add(E e) {
    synchronized (lock) {
        Object[] es = getArray();
        int len = es.length;
        es = Arrays.copyOf(es, len + 1); // 复制一份数据
        es[len] = e;	// 在副本上做修改
        setArray(es);	// 让数组的指针指向新的副本
        return true;
    }
}
```

#### 体验Pandas中的CoW

Pandas 中的 copy_on_write 也是类似的，在副本上做修改，不会改变原视图，我们可以写代码验证下这点。

1️⃣给定一个 list，我们向里面插入元素，然后比较插入元素前后 list 的地址，地址都是一样的

```python
l = [1,2,3]
print(id(l))    # 0528
l.insert(0,100)
# l.append(22)
print(id(l))    # 0528
```

2️⃣给定一个 Pandas#DataFrame，不开启 CoW，我们修改里面的元素，然后比较被修改的列的地址.

- 测试直接切片赋值（使用 pandas 3.0 以下的版本，还支持切片修改原数据）

```python
import pandas as pd

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})

print(id(df['js']))  # 7584
df['js'][df['js'] < 60] = 60
print(id(df['js']))  # 7584
```

- 我们开启 `mode.copy_on_write` 再次进行测试发现无法更改

```python
import pandas as pd

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})

df['js'][df['js'] < 60] = 60
```

#### 理解CoW（1）

我们先了解下 CoW 模式和非 CoW 模式下访问 DataFrame 中的列是什么样的调用情况

1️⃣我们先看一个未开启 CoW 的代码 `df['js'] is df['js']` 每次都是返回 True，我们 debug 看看为什么每次都是 True

```python
import pandas as pd

# 开启 Copy-On-Write 技术
# pd.options.mode.copy_on_write = True

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})
# print(df['js'] is df['js'])  # True 每次返回的都是相同的对象

data = df['js']
data = df['js']
```

调用流程如下

```python
"""
data = df['js']  
    ==> 调用 __getitem__ 方法
        ==> _get_item_cache(key)，如果没有使用 CoW，就走 _ixs + cache[item] = res
            ==> _ixs 走 _mgr.iget(i) 用于获取一个新的 mgr
                ==> _mgr.iget()
                    ==> block = self.blocks[self.blknos[1]] 用于获取原始数据的
                    ==> 然后将数据包装成一个新的 mgr
                ==> _box_col_values(col_mgr, i)，将 mgr 包装成一个 Series
                    ==> _box_col_values 走 _constructor_sliced_from_mgr
                    ==> _constructor_sliced_from_mgr 走 _sliced_from_mgr 生成一个新的 Series
            ==> cache[item] = res 缓存这个新的 Series
            
    ==> 再次调用 __getitem__ 方法
        ==> res = cache.get(item) 中能查到，返回的都是相同的 Series 了
        
总结：
    第一次获取 Series 时，会根据原始数据生成 mgr ==> Series, 然后缓存这个 Series
    后面获取时都是从缓存中获取了，mgr 中的 self.blocks 中保存了最原始的 NumpyBlock
"""
```

从上面的调用流程可以看出来，如果未开启 CoW 则第一次访问时会缓存，后面访问会直接读取缓存中的 Series，因此每次返回的都是一个同一个 Series 对象。

2️⃣开启 CoW 的代码 `df['js'] is df['js']` 每次都是返回 False，我们 debug 看看为什么每次都是 False

```python
import pandas as pd

# 开启 Copy-On-Write 技术
pd.options.mode.copy_on_write = True

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})
# print(df['js'] is df['js'])  # False, 每次返回的都是不同的对象
data1 = df['js']
data2 = df['js']
print(id(data1) == id(data2))  # False
print(id(data1._mgr) == id(data2._mgr))  # False
print(id(df['js']) == id(df['js']))  # False

# debug 走是 False, 正常运行 True
print(id(df['js']._mgr) == id(df['js']._mgr))


for item in range(5):
    # 用变量接收每次的地址都是不一样的
    # 直接输出，地址有时候可能一样
    d = df['js']
    print(id(d))
```

调用流程

```python
"""
data = df['js']  
    ==> 调用 __getitem__ 方法
        ==> _get_item_cache(key)，如果使用了 CoW，就不走 cache 了
            ==> 走 _ixs 方法
                ==> 走 _mgr.iget(i) 获取一个新的 mgr
                    ==> iget 从 block 中拿数据，然后返回一个 BlockPlacement 对象
                    ==> BlockPlacement 对象被包装成 SingleBlockManager
                    ==> 返回一个新的 mgr（每次都是返回新的 mgr）
                ==> 走 _box_col_values(col_mgr, i) 将 mgr 包装成一个新的 Series
                    ==> _box_col_values 走 _constructor_sliced_from_mgr
                        ==> _constructor_sliced_from_mgr 走 _sliced_from_mgr 生成一个新的 Series
        ==> 每次都是返回新的 Series
"""
```

从上面的调用流程可以看出来，如果开启 CoW 则每次访问都会创建一个新的 Series

#### 理解CoW（2）

我们再了解下 CoW 模式和非 CoW 模式下修改 DataFrame 中的列数据是什么样的情况

1️⃣非 CoW 模式修改数据，修改的是 `_mgr`，series 共享 `_mgr` 因此能被感知到

```python
import pandas as pd

# 开启 Copy-On-Write 技术
# pd.options.mode.copy_on_write = True

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})
data = df['js']
data[0] = 500
# print(id(data._mgr))  # 一样的地址
print(df['js'][0])
# print(id(df['js']._mgr))  # 一样的地址
```

调用流程

```python
"""
data[0] = 500  
    ==> 调用 __setitem__ 方法
        ==> 没有使用 CoW， 所以走 sys.getrefcount(self) 这块不用管做什么的，和主线任务无关
        ==> 走 _set_with_engine(key, value, warn=warn)
            ==> 走 _mgr.setitem_inplace(loc, value, warn=warn)
                ==> 走 setitem_inplace 直接修改的 ndarray 中对应 index 中的值
                ==> _mgr 修改成功
        ==> _maybe_update_cacher(inplace=True) 更新 cache
            ==> _maybe_cache_changed(cacher[0], self, inplace=inplace) 需要修改 js 的 cache
                ==> × _ixs(loc, axis=1) ==> 这部分的代码未使用到，只是执行了
                    ==> _box_col_values(col_mgr, i) 创建一个新的 Series，但是是共享的 _mgr
                    ==> result._set_as_cached(label, self) 这是一个缓存的值，将其标记为已缓存
                ==> _mgr.iset(loc, arraylike, inplace=inplace)
        
    ==> df['js'][0]
        ==> 调用 __getitem__ 方法
            ==> 从 cache 中获取的数据，cache 中的 Series 是共享的 _mgr
            ==> 由于 data[0] = 500 的过程中修改了 _mgr 所以 cache 中可以感知到值修改了
"""
```

2️⃣CoW 模式修改数据

```python
import pandas as pd

# 开启 Copy-On-Write 技术
pd.options.mode.copy_on_write = True

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})
data = df['js']
data[0] = 500
print(df['js'][0])
```

调用流程

```python
"""
data[0] = 500  
    ==> 调用 __setitem__ 方法
        ==> 使用 CoW， 所以走一下 warnings
        ==> 走 _set_with_engine(key, value, warn=warn)
            ==> 走 _mgr.setitem_inplace(loc, value, warn=warn)
                ==> 走 setitem_inplace，由于启用了 CoW，因此走了一个复制的代码
                    ==> self.blocks = self._block.copy(), 确保 self.blocks 中的是旧数据
                    ==> 清除了 _cache.clear()
                    ==> 修改原始的 _mgr,修改成功
        
    ==> df['js'][0]
        ==> 调用 __getitem__ 方法
            ==> 尝试从 _get_item_cache 获取，发现使用了 CoW，没从缓存中拿
            ==> _ixs(loc, axis=1)
                ==> _mgr.iget(i) 从 blocks 中拿数据，blocks 中存储的旧数据
                ==> _box_col_values 得到一个新的 Series
        ==> 所以拿到的是旧数据
    
    ==> 在更改过程中确保了 self.blocks 中的是旧数据
"""
```

#### 理解CoW（3）

- 为了深入理解 Pandas 中的 CoW 机制，我们调试下列代码进行观察

```python
import pandas as pd

# 开启 Copy-On-Write 技术
pd.options.mode.copy_on_write = True

df = pd.DataFrame({'js': [50, 80, 90], 'python': [90, 80, 99]})
# print(df['js'] is df['js'])  # False, 每次返回的都是不同的对象
data = df['js']  # 每次返回的都是一个全新的对象
bool_slice = [True,True,False]

data[bool_slice] = 60   # generic.py 10747 行 if inplace: 原地更新的，实际上是更换了 series 的 _mgr，series 对象没有改变，只是存储值的 _mgr 变了

"""
generic.py 10747 行
        if inplace:
            # we may have different type blocks come out of putmask, so
            # reconstruct the block manager
            # 尝试将新值复制过去
            new_data = self._mgr.putmask(mask=cond, new=other, align=align, warn=warn)
            result = self._constructor_from_mgr(new_data, axes=new_data.axes)
            return self._update_inplace(result) # 更新的数据
            
            _update_inplace() 中的 self._mgr = result._mgr 更改的数据
"""
```

Pandas 官方推荐：Try using `.loc[row_indexer, col_indexer] = value` instead, to perform the assignment in a single step.

### 属性&方法

属性和方法与 Series 类似，不过多了一个 columns。

| 属性   | 说明 |
| ------ | ---- |
| shape  | 形状 |
| size   | 长度 |
| index  | 索引 |
| values | 值   |
| name   | 名字 |

<b>常用且实用的方法有</b>

| 方法     | 说明                      |
| -------- | ------------------------- |
| head()   | 查看前几条数据，默认 5 条 |
| tail()   | 查看后几条数据，默认 5 条 |
| sample() | 随机采样数据              |
| map()    | 可以用 map 修改属性       |

```python
# 修改 df 语言这列的值，增加一个后缀 _1
df.语言.map(lambda x: x.strip()+'_1')
"""
汉语	==> 汉语_1
粤语	==> 粤语_1
"""
```

### 索引&切片

DataFrame 取数据的方式与 numpy 略有不同，直接利用 DataFrame 取元素时，<b style="color:red">DataFrame 优先取列索引</b>

数据准备

```python
data = np.random.randint(low=60, high=90, size=(5,5))
df = pd.DataFrame(data)
df.index=['A', 'B', 'C', 'D', 'E']
df.columns = ['Java', 'Python', 'C Plus Plus', 'C#', 'js']
```

#### 索引

DataFrame 直接索引数据时支持下面这几种操作

- 对行进行切片，支持使用自定义索引进行切片操作
- 检索单独一列数据，支持使用自定义索引检索列
- 检索连续/不连续的列数据，支持使用自定义索引检索列
- <span style="color:red">不支持列切片！不支持同时检索行和列！其功能完全可以被 loc 和 iloc 替代，可不记！</span>

DataFrame 使用 loc / iloc 是默认行优先，可以执行任何切片操作，也支持使用自定义索引进行切片操作~

<b>取一列数据</b>

DataFrame 优先取列元素的代码示例，访问 df 中的一列数据（优先对列做索引）

```python
df['Java'] # Series 类型
df.Java # 也可以，不过要符合变量命名规则，不推荐🤣

df[ ['Java'] ] # DataFrame 类型
```

返回 DataFrame 类型方便链式调用。

<b>取行数据</b>⭐

如果想<span style="color:blue">取一行</span>元素的话需要借助 `loc / iloc` 这两个方法

- 使用 .loc[] 加自定义 index 来进行行索引
- 使用 .iloc[] 加整数来进行行索引

访问 df 中某一行的数据

```python
df.loc['A']	# 取一行, Series
df.loc[['A','B']]	# 取多行 DataFrame

df.iloc[0]	# 取一行, Series
df.iloc[[0,1]]	# 取多行 DataFrame
```

<b>如何取元素？</b>⭐

取列元素呢？那就恢复到了类似于 numpy 切片的方式了，那取指定行列的具体元素呢？也是类似于 numpy 的访问方式。

分别利用 loc 和 iloc 取出 Python 和 C# 这两列的数据。

```python
df.loc[:,['Python','C#']]
df.iloc[:,[1,3]]
```

取出 C 行对应的 Python 和 C# 成绩

```python
df.loc[['C'],['Python','C#']] # DataFrame 类型

df.iloc[[2],[1,3]]
```

<b>DataFrame 的 values</b>

也可以使用 DataFrame 的 values 获取数据的 numpy 数组，然后对 numpy 进行`取列/行/元素`的操作。

#### 切片

- 直接使用中括号对 DataFrame 进行操作，DataFrame 支持行切片，不支持列切片；同时，dataframe[] 可以索引选取不连续的列，但是由于默认列优先，不能索引选行~
- 使用自定义 index 进行切片的时候，左右都是闭区间
- 使用 `loc/iloc` 时与 numpy 的方式类似，索引优先对行进行操作，`:` 切片时是左闭右开

```python
data = np.random.randint(low=60,high=90,size=(5,5))
df = pd.DataFrame(data)
df.index=['A','B','C','D','E']
df.columns = ['Java','Python','C Plus Plus','C#','js']

# 取 ’A‘~'D' 行的数据（闭区间，包含D）
df['A':'D'] # 直接使用 DataFrame 进行切片

df['Java'] # 直接使用 DataFrame 获取 Java 这一列的所有数据，返回 Series

df[['Java']] # 直接使用 DataFrame 获取 Java 这一列的所有数据，返回 DataFrame

df[['Java','Python']] # 直接使用 DataFrame 获取 Java Python 两列 的所有数据，返回 DataFrame
```

<b>切片的使用建议</b>⭐

- 要么取一行或一列：索引
- 要么取连续的多行或多列：切片
- 要么取不连续的多行或多列：中括号

<b>注意：</b>取连续的多行多列还是用切片方便些

> <b>习题</b>

2 种方式创建一个 DataFrame，包含 6 个学生(小明，小红，小绿，小白，小黑，小黄)，每个学生有 4 门课(语文，数学，英语，物理)，成绩随机值

- 找到小红的语文成绩
- 找到小红和小白的数学和物理成绩

```python
columns = ['语文','数学','英语','物理']
index = ['小明','小红','小绿','小白','小黑','小黄']
df = pd.DataFrame(data=np.random.randint(low=50,high=98,size=(6,4)),columns=columns,index=index)

# 找到小红的语文成绩
# 注意，这里使用 [ ['语文'] ] 是希望返回值为 DataFrame
# 希望查找的数据符合原始数据的格式
df[['语文']].loc[['小红']]

# 同上，也是希望返回值为 DataFrame,让查找的数据符合原始数据的格式
df[['数学','物理']].loc[['小红','小白']]
```

### 运算

和 Series 类似，和标量计算时会广播，两个 DataFrame 计算时不会进行广播，行列不匹配的数据计算时会出现 NaN。

如果不希望计算结果出现 NaN 可以使用 add，用法和 Series 的 add 一样。

### 数据合并

#### concat 合并

和 numpy 中的矩阵合并是类似的，指定合并的维度，在这个维度上堆叠数据。如两个 shape=(3,4) 的 DataFrame 在维度 0 上进行合并，堆叠后的结果就是 shape=(6,4)。

准备数据

```python
def make_df(index, columns):
    df = pd.DataFrame(data=np.random.randint(1,100, size=(len(index), len(columns))),
                      index=index, 
                      columns=columns
                     )
    return df
```

<b>简单合并/级联</b>

合并两个 DataFrame

```python
df1 = make_df(['A','B','C'],[1,2,3,4])
df2 = make_df(['A','B','C'],[1,2,3,4])
pd.concat((df1,df2))
```

合并时重置索引

```python
pd.concat((df1,df2), ignore_index=True)
```

合并时使用多重索引

```python
pd.concat((df1,df2), keys=['x', 'y'])
```

<b>不匹配合并/级联</b>

不匹配指的是级联的维度的索引不一致。例如纵向级联时列索引不一致，横向级联时行索引不一致；对于不一致的数据会用 NaN 填充

外连接：补 NaN（默认模式）

内连接：只连接匹配的项

#### merge 合并规则

- 类似 MySQL 中表和表直接的合并
- merge 与 concat 的区别在于，merge 需要依据某一共同的行或列来进行合并
- 使用 pd.merge() 合并时，默认根据两者相同 column 名称的那一列，作为 key 来进行合并。（默认做内连接，只显示公共部分）
- 每一列元素的顺序不要求一致

<b>按公共字段进行合并</b>

- 只有一列字段相同，则只比对这个字段的值
- 有两列字段相同，则会同时比较两个字段的值
- 多列相同，指定某列作为连接字段，使用 `on`

- 一般只会用一个公共字段或指定字段进行合并

仅一列字段相同

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'sex': ['男', '女', '男'],
    'job': ['Saler', 'CEO', 'Programer']
})
# 只会合并公共字段相同的数据
pd.merge(df1, df2) 
```

两列字段相同，同时比较两列字段

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'age': [33, 33, 44],
    'job': ['Saler', 'CEO', 'Programer']
})
# 只会合并公共字段相同的数据
pd.merge(df1, df2) 
```

两列字段相同，指定使用其中一列进行合并

```python
pd.merge(df1, df2,on='id') 
```

<b>没有公共字段，按指定字段进行合并（left_on / right_on）</b>

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id1': [1, 2, 3],
})
df2 = pd.DataFrame({
    'id2': [2, 3, 4],
    # 'sex': ['男', '女', '男'],
    'job': ['Saler', 'CEO', 'Programer']
})
# 只会合并公共字段相同的数据
pd.merge(df1, df2,left_on='id1',right_on='id2') 
```

#### merge 合并关系

合并关系可以分为三种

- 一对一合并：A、B 两个数据直接是一对一的关系
  - eg：A,B 有一个公共字段，A 中的数据的 id 在 B 中出现了一次
- 一对多合并：A、B 两个数据是一对多的关系
  - eg：A,B 有一个公共字段，A 中的数据的 id 在 B 中出现了两次（一对多）
- 多对多合并：A、B 两个数据是多对多的关系
  - eg：A,B 有一个公共字段，A、B 中的数据的 id 分别在其他地方出现了两次

注意与上面的 merge 合并进行区分，上面的 merge 合并只是说的合并规则，而非合并关系。

一对一合并

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'sex': ['男', '女', '男'],
    'job': ['Saler', 'CEO', 'Programer']
})

pd.merge(df1, df2) 
```

一对多合并

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 3],
    'sex': ['男', '女', '男'],
    'job': ['Saler', 'CEO', 'Programer']
})
# 一对多，一会逐个和多进行匹配
display(pd.merge(df1, df2))
```

多对多合并

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 3, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 3],
    'sex': ['男', '女', '男'],
    'job': ['Saler', 'CEO', 'Programer']
})
# 也是逐个合并
display(pd.merge(df1, df2))
```

#### 内/外合并

merge 有两种合并方式，内合并和外合并。

- 内合并：只保留两者都有的 key（默认模式）
- 外合并 how='outer'：不匹配的也显示，补 NaN
- 左合并、右合并：how='left'，how='right'，类似于左/右外连接
  - 左合并，显示左表的所有数据和匹配数据

内合并，默认就是内合并

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'age':[22, 33, 44],
    'job': ['Saler', 'CEO', 'Programer']
})
# 默认就是 inner 合并
pd.merge(df1, df2,on='id',how='inner')
```

外合并，保留所有数据

```python
df1 = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'id': [1, 2, 3],
    'age': [22, 33, 44]
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'age':[22, 33, 44],
    'job': ['Saler', 'CEO', 'Programer']
})
# 默认就是 inner 合并
pd.merge(df1, df2,on='id',how='outer')
```

#### 总结

<b>重难点是 merge，merge 合并总结</b>

   - 合并有三种现象：一对一, 多对一, 多对多.
   - 合并默认会找相同的列名进行合并, 如果有多个列名相同，用 on 来指定.
   - 如果没有列名相同,但是数据又相同，可以通过 left_on, right_on 来分别指定要合并的列.
   - 如果想和 index 合并，使用 left_index, right_index 来指定.
   - 如果多个列相同，合并之后可以通过 suffixes 来区分.
   - 还可以通过 how 来控制合并的结果，默认是内合并，还有外合并 outer，左合并 left，右合并 right.

## 缺失值处理

一般有两种空值：None 和 np.nan

- None 是 Python 自带的，是 Python 中的空对象。None 不能参与到任何计算中；object 类型的运算要比 int 类型的运算慢得多
- np.nan 是浮点类型，能参与到计算中。但计算的结果总是 NaN；但可以使用 np.nan*() 函数来计算 nan，此时会过滤掉 nan

不过 Pandas 中的 None 和 np.nan 都视为 np.nan

### 操作-查找 nan

<b>pandas 的 axis 的计算方式和 numpy axis 的计算方式类似，对于二维矩阵</b>

- axis=0 表示沿着列的方向，做逐行的操作
- axis=1 表示沿着行的方向，做逐列的操作

对于二维矩阵 torch dim 维度的计算方式也是一样的

```python
import torch
import numpy as np
data = [
    [1,1],
    [2,2]
]

nd = np.array(data)
tensor = torch.tensor(nd)
print(nd.shape, tensor.shape, '\n')

"""
1 1
2 2
"""

print(nd.sum(axis=0))			# 3 3
print(tensor.sum(dim=0).data)	# 3 3
```

Pandas 常用操作

| 方法      | 说明                                                         |
| --------- | ------------------------------------------------------------ |
| isnull()  | 判断每个元素是否<b>为 null</b>                               |
| notnull() | 判断每个元素是否<b>不为 null</b>                             |
| all()     | 都为 true 则为 true                                          |
| any()     | 有 true 则为 true                                            |
| dropna()  | 过滤丢失数据，默认删除有空的行，不是运算所以 axis = 0 是删除行<br>可以选择过滤的方式 how = 'all', 必须全部为空才能删除 |
| fillna()  | 填充丢失数据，fillna(0) 进行 0 填充；<br>一般会选择均值 / 中位数 / 众数填充； |

<span style="color:blue">Pandas 中涉及到删除或修改原数据的操作，一般都不会直接修改原数据。如果希望修改原数据，需要设置 inplace=True</span>

```python
import numpy as np
import pandas as pd

score = pd.DataFrame(data=np.random.randint(low=10,high=90,size=(4,5)))
score.index = list('ABCD')
score.columns=['Java','Python','JS','Go','Kotlin']

score.loc['A','Python'] = np.nan
score.loc['B','JS'] = np.nan

score.isnull().any(axis=0) # DataFrame 列优先，所以 axis=0 是判断每列的数据是否有空的
score.isnull().any(axis=1) # DataFrame 列优先，所以 axis=1 是判断每行的数据是否有空的
```

### 填充缺失值-fillna

计算出每一列的的均值 / 中位数 / 众数等，然后填充对应列的空值。

准备数据

```python
import numpy as np
import pandas as pd
data = np.random.randint(low=10,high=90,size=(4,5))
score = pd.DataFrame(data=data)
score.index = list('ABCD')
score.columns=['Java','Python','JS','Go','Kotlin']

# 填充 nan 值
score.loc['A','Python'] = np.nan
score.loc['B','JS'] = np.nan
score.loc['B','Python'] = np.NaN
score.loc['C','Python'] = np.NaN
score.loc['B','Java'] = np.NaN
score.loc['C','Java'] = np.NaN
```

填充缺失值，如果希望在原数据上填充缺失值，可以使用 `inplace=True`，不过 Pandas 3.0 未来不会支持这种原地修改的操作

```python
sc1 = score.copy()

# scores_copy1
sc1['Java'].fillna(sc1['Java'].mean(),inplace=True)
sc1['Python'].fillna(sc1['Python'].mean(),inplace=True)
sc1['JS'].fillna(sc1['JS'].mean(),inplace=True)
```

上面这种方式要一个一个计算每列的值，太麻烦了，下面是一种简便的写法。

```python
sc2 = score.copy()
sc2.fillna(sc2.mean(),inplace=True)
```

### 处理重复值和异常值

#### 处理重复值

| 方法              | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| duplicated()      | 检测重复的行（一般不会也没必要检测重复列）<br>默认所有数据重复才算重复，可以用 subset 指定部分重复就算重复<br>keep='first' 默认保留第一个<br>keep='last' 默认保留最后一个 |
| drop_duplicates() | 删除重复的行（一般不会也没必要检测重复列）<br/>使用方式和 duplicated() 一致 |

使用 duplicated 检测重复的子集，并保留最后一个

```python
import numpy as np
import pandas as pd
df = pd.DataFrame(data=np.random.randint(1, 100, size=(4, 4)), 
                  index=list('ABCD'),
                  columns=['Python', 'Java', 'Go', 'C'])

# 设置重复数据, 让 B 和 D 的子集 [Python 和 Java] 相似
df.loc[['B','D'],['Python','Java']] = [43,90]

# 检测子集上那些数据重复了，最后出现的重复数据保留（既视为 False，不重复）
df.duplicated(subset=['Python','Java'], keep='last')
```

使用 drop_duplicates 删除子集重复的行，并保留最后一行

```python
df.drop_duplicates(subset=['Python','Java'], keep='last')
```

#### 寻找异常值

如何寻找异常值？先统计所有数据的统计量，然后看每个量和统计量的差异，差异大的就可能是异常值

| 方法           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| describe()     | 查看每一列的描述性统计量<br>用于了解数据的统计分布           |
| info()         | 查看数据缺失情况和数据类型                                   |
| value_counts() | 统计数组中每个值的次数，次数最多的会出现在第一行<br>也可以设置 subset |
| unique()       | 唯一，去重 (用于 Series)                                     |
| nunique()      | 查看不同元素的个数（用于 DataFrame）                         |

#### 剔除异常值

| 方法   | 描述           |
| ------ | -------------- |
| drop() | 删除指定行或列 |
| bool   | 值索引进行过滤 |

准备数据

```python
df = pd.DataFrame(data=np.random.randint(0,100,size=(4,4)))

df.index = list('ABCD')
df.columns=['Python', 'Java', 'Go', 'C']
```

删除行 / 列

```python
df.drop(columns=['B','C']) 	# 删除行
df.drop(index=[0, 2, 3])	# 删除列
```

布尔切片

```python
bool_slice = df['Python']>52
bool_slice.shape # 4,

df[bool_slice] # bool_slice 有四行，行中为 true 的会被切出来

df[~bool_slice] # bool_slice 有四行，行中为 false 的会被切出来
```

## 分组聚合

数据聚合是数据处理的最后一步，通常是要使每一组数据生成一个单一的数值。

数据分类处理

 - 分组：先把数据分为几组
 - 用函数处理：为不同组的数据应用不同的函数（sum、avg）以转换数据
 - 合并：把不同组得到的结果合并起来

### groupby

<b>使用 groupby 进行分组，分组后利用其他函数进行统计。groupby 中常见的聚合函数位于 pandas#core#groupby#groupby.py#GroupBy</b>

| 方法      | 说明                                                         |
| --------- | ------------------------------------------------------------ |
| count     | 数量统计                                                     |
| mean      | 均值                                                         |
| median    | 中位数                                                       |
| sum       | 求和                                                         |
| max / min | 最大最小值                                                   |
| agg       | 使用指定的方式进行聚合，The aggregation is for each column<br>agg 的操作比较灵活，这里列举几个操作<br>`df.groupby('A').agg(['min','max'])`<br>`df.groupby('A').agg({'weight':'sum', 'price':'mean' })` 按照 A 分组，统计 weight 的 sum 和 price 的 mean<br/> `df.groupby('A').agg(lambda x: sum(x)+2)` 按照 A 分组，然后求和，对求和后的结果再 + 2 |

<b>分组后使用场景聚合函数进行统计</b>

eg：将数据按颜色进行分组，统计每组颜色的价格总和。

```python
df = pd.DataFrame(
    {
        'color': ['green', 'green', 'yellow', 'blue', 'blue', 'yellow', 'yellow'],
        'price': [4, 5, 3, 2, 1, 7, 6],
        'price2': [4, 5, 3, 2, 1, 7, 6]
    }
)

# 先按 color 分组，分组后得到新的数据
# 我们是要统计价格的总和，所以要拿到价格这列的数据
# 拿到数据后进行求和
df.groupby(by='color')['price'].sum() # Serise 类型
df.groupby(by='color')[['price']].sum() # DataFrame 类型
```

### 练习题

假设商场的商品包含以下属性

| 属性         | 内容                   |
| :----------- | ---------------------- |
| 菜品(item)   | 萝卜，白菜，辣椒，冬瓜 |
| 颜色(color)  | 白，青，红             |
| 重量(weight) |                        |
| 价格(price)  |                        |

要求如下

1. 要求以属性作为列索引，新建一个 df
2. 进行聚合操作，求出颜色为白色的价格总和
3. 进行聚合操作，分别求出萝卜的所有重量以及平均价格
4. 使用 merge 合并总重量及平均价格

以属性作为列索引，新建一个 df

```python
df = pd.DataFrame(
    data={
        "item": ["萝卜","白菜","辣椒","冬瓜","萝卜","白菜","辣椒","冬瓜"],
        'color':["白","青","红","白","青","红","白","青"],
        'weight': [10,20,10,10,30,40,50,60],
        'price': [0.99, 1.99, 2.99, 3.99, 4, 5, 6,7]
    }
)
```

进行聚合操作，求出颜色为白色的价格总和

```python
df[ df.color == '白' ]['price'].sum()
df.groupby(by='color')['price'].sum() # series 类型
df.groupby(by='color')[['price']].sum() # DataFrame 类型

# as_index = False 让分组字段不变成行索引
df.groupby(by='color', as_index=False)[['price']].sum()

# 先求出每种颜色的总价，再找白色的价格也一样
df.groupby(by='color').sum().loc[['白'],['price']]
```

进行聚合操作，分别求出萝卜的所有重量以及平均价格

```python
df.groupby(by='item')['weight'].sum()['萝卜']
df.groupby(by='item')['price'].sum()['萝卜']
```

使用 merge 合并总重量及平均价格

```python
w_sum = df.groupby('item')[['weight']].sum()
p_mean = df.groupby('item')[['price']].mean()
# 两者的 index 值是一一对应的，所以用 index 进行匹配
pd.merge(w_sum, p_mean, left_index=True, right_index=True)
```

使用 agg 进行多种聚合操作，求总重量及平均价格（不指定列默认是对所有的列都进行操作）

```python
df.groupby(by='item').agg({'weight':'sum', 'price':'mean'})

# 在 agg 中使用别名
df.groupby(by='item').agg(总重量=('weight','sum'), 平均价格=('price','mean'))
```

## 分箱操作

分箱操作就是将连续型数据离散化。例如，根据身高的范围，将人群的身高分为矮、中、高，这就是将连续性的身高离散化成三种取值。

<b>分箱操作分为等距分箱和等频分箱</b>，需要详细解释下两种分箱操作的概念

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体，让汉字不乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
# 支持负数
plt.rcParams['axes.unicode_minus'] = False
```

### 等距分箱

按学生的成绩，将学生分为：优良中

```python
data = np.random.randint(50, 100, size=(50, 3))
df = pd.DataFrame(data=data, columns=['Python', 'Pandas', 'PyTorch'])


# 对 Python 这一列的数据进行分箱

# 自动给你分成四组，在(] 范围的归为一组，不推荐这种分组方式
df['bin'] = pd.cut(df.Python, bins=4) 

# 自动给你分成四组，在(] 范围的归为一组，0~30 一组 30~60 一组 60~100 一组，这种更直观
df['bin'] = pd.cut(df.Python, bins=[0,30,60,100]) 

# ⭐labels 给分组一个标签值，推荐这种⭐
df['bin_label'] = pd.cut(df.Python, bins=[0,60,85,100], labels=list('中良优'))
```

### 等频离散

等频离散意思是，使划分的区间中，样本数量尽量保持一致。

## 绘图

Pandas 内部集成了 matplotlib 可以绘图，但是复杂的图还是用 matpotlib，简单了解下此处的绘图功能即可。以上面的代码为例，绘制成绩的柱状图。

```python
df['bin_label'] = pd.cut(df.Python, bins=[0,60,85,100], labels=list('中良优'))

df.bin_label.value_counts().plot(kind='pie', autopct='%1.1f')
```

kind 用于指定画何种图，autopct 用于指定在图中显示的数据格式

| kind 参数 | 说明                           |
| --------- | ------------------------------ |
| line      | line plot (default)            |
| bar       | vertical bar plot              |
| barh      | horizontal bar plot            |
| hist      | histogram                      |
| box       | boxplot                        |
| kde       | Kernel Density Estimation plot |
| density   | same as 'kde'                  |
| area      | area plot                      |
| pie       | pie plot                       |
| scatter   | scatter plot (DataFrame only)  |
| hexbin    | hexbin plot (DataFrame only)   |

用 kind 画图可能不太方便，Pandas 还提供了其他 API 用于绘制各种图。用其他 API 画出一个 sin 曲线。

```python
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
dict_sin = {'x':x,'y':y}
df = pd.DataFrame(dict_sin)

# 用 kind 指定画散点图
df.plot(kind='scatter',x='x',y='y')
# 直接指定画 scatter 散点图
df.plot.scatter(x='x',y='y', color='red')
```

随机生成 50 个点，绘制散点图

```python
x = np.random.normal(size=50)
y = np.random.normal(size=50)
df = pd.DataFrame({'x':x,'y':y})
df.plot.scatter(x='x',y='y', color='red')
```

<b>题外话</b>

0 维：点

1 维：线

2 维：面

3 维：立体

