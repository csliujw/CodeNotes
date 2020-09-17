# 概述

> **Python基础语法入门**，便于后期用python写代码

shift+tab 查看帮助文档。 jupyter notebook.

Annconda：开源得Python发行版本，集成了众多的三方库，不用自己去安装（减少了版本冲突，很多框架也用了这些方式，典型的有SpringBoot的约定大于配置）。Annconda中也包含了conda。conda可以当作一个虚拟环境，为不同的项目继承不同的依赖，彼此互不干扰（类似于沙箱机制吗？）

Python的效率和语法有关。

列表推导式的效率远高于for循环。

```python
# 采用列表推导式遍历文件
file_obj = open("IO_Demo.txt")
[print(x.rstrip()) for x in file_obj]

list = [1,2,3,4,5,6,7,7,4,4]
[ print(x) for x in list if x%2 == 0 ]
# 遍历的基本模板
# [ 操作 for x in xxx ]

# 数据筛选模板
# [ 操作 for x in xxx 条件判断 ]
```

---

# 基本语法

### **基本数据类型**

- Integer： integer转string  str(123)
- float
- string

---

### **运算**

- +加

- -减

* *乘

- /除  5/3 = 1.666667 和 java不一样
- %取余
- // 取整
- 3**2 = 3的2次方
- 2**4 = 2的四次方

---

### **prinf**

```python
print(123)  # output 123
print(1+2) # output 3
print(1,2) # output 1 2
```

---

### 循环

- while 不确定循环次数
- for  确定循环次数

```python
condition = 100
while condition > 1:
    print(condition)
    condiion-=1
    
for i in range(10): # range(10)产生0 1 2 3 4 5 6 7 8 9
    print(i)
    
    
range(2,10) #左闭 右开  包含2 不包括10

range(2,10,2) # start end step
```

---

### **布尔**

True False 首字母大写

----

### **数据结构及API**

#### list

**列表用法【list】**

```python
data_list1 = [] # 空列表
data_list2 = [1,2,3,4] # 带初始数据的
data_list3 = [1,2,3,"123",2.3] # 可以是不同的数据类型

data_list3[0] # 数据的访问同数组一样，通过下标  output 1
data_list3[-1] # output 2.3 从右向左数 【输出的倒数第一个元素】

# 打印元素
data_list3[::] # 打印所有
data_list3[1:4] # 左闭右开 output: [2, 3, '123']
data_list3[:-2] # output [1,2,3]
# 打印后三个
data_list3[-3:]

# 获得长度
len(data_list1)
# 获得指定元素第一次出现的位置
data_list3.index(2) # output:1
# 统计元素个数
data_list3.count(2) # output:1

# 排序 会改变原来的列表。 排序要求 元素的类型一致！
data_list3.sort() # 默认从小到大
data_list3.sort(reverse=True) # 逆序 即从大到小
```

**列表操作&多维列表**

``` python
list = [1,2,3]
list.append(10) # 尾部追加
list.insert(2,10) # 指定索引处插入元素
del list[0] # 输出0索引的元素
list.remove(2) # 移除列表中 2 这个值 只移除一个

list.pop() # 弹出尾部元素

mutl_list = [[1,2,3],[3,4,5]] # 类似多维数组 常用二维，表示矩阵
```

#### tuple

元组不可变，不可追加删除元素，不可修改元组中的数据

修改元组只能是该引用，给引用重新赋值。

#### 字典

> key -- value

```python
dictionary = {"key1":"value1","key2":"value2"} 
```

> 获取字典中的值

```python
dictionary[key] # 获得对应key的value
dictionary['key1'] == 'value1' # output true
dicitionary['key1'] = 'new value' # 修改值
```

> 删除字典的key--value对

```python
del dicitionary['key1'] # 删除 key1 -- value1
```

> 字典的遍历

```python
for key,value in dicitionary.items():
    print("key:",key,"\t","value:",value)
```

```python
for key in dicitionary in d.keys():
    print("key==",key)
```

> 对字典里的key进行排序

```python
for key in sorted(dicitionary.keys()):
    print("key == ",key)
```



----

### 条件判断

```python
num1 = 10
num2 = 20
num3 = 30
if num1<num2<num3:
    print("OK")
else:
    print("sorry")
    
if num1>10:
    print("num1>10")
elif num2>20:
    print("num2>20")
else:
    print("sorry")

if num1 == 100:
    print("num1==100")
elif num1==50:
    pass # 不想执行任何语句的话要加pass
else:
    print(num1)
```

```python
if 1>2 and 3<4:
    print("1111")
    
if 1>2 or 3<4:
    print("1111")

# 判空
if x is not None
```

---

### 函数

> **定义**

```python
# 无参数
def function():
    # 代码
    
# 有参函数
def function2(a,b):
    return a+b

# 设置默认值
def function3(a=10,b=20)
	return a+b
```

> 使用

```python
function()
function2(1,5) # output 6
function3() # output 30
```

----

### 模块

> **示例**

建议安装 annconad，包含常用的库。

- 安装numpy ==  pip install numpy
- 卸载numpy == pip uninstall numpy

```python
# 定义模块 保存文件名未max
def fun_max(a,b):
    if a>b:
        return a
    else:
        return b
    
# 把模块和要调用模块的程序放在一起（同一级目录下）
import max# 导入模块
max.fun_max(10,20)

from max import fun_max # 从max模块导入fun_max函数
fun_max(10,20)

from max import * # 从max模块导入其所有的内容

import max as m # 给模块取别名

import os # 导入os模块
os.getcwd() # 获得当前文件所在的路径
```

----

### 类

> **定义类**

```python
class human:
    # 类属性（成员变量）
    name = "YZK"
    age = 23
    def __init__(self,name,age):
        self.name = name
        self.age = age
        print("init~~~")
        
    def my_name(self): # self 表示当前类
        print("my name is ",self.name)
```

> **创建类**

```python
person = human("zzx",18)
person.name
person.name="!2223"
person.my_name()
```

**PS**： 

- python无重载的语法，但可通过转发实现重载。
- python中的类方法一定要写上self参数，表示当前类（与this类似，但是要显式说明）

> **类的继承**

```python
class father:
    def __init__(self,name='human',age=15):
        self.name = name
        self.age = age
     
    def get_name(self):
        return self.name
    
    def get_age(self):
        return self.age
    
class son(father): # 继承父类father
    pass

class son2(father):
    def __init__(self,grade=1,school='UST')
    	super().__init__() # 调用父类初始化. 注意！！不会自动调用父类的初始化方法
        self.grade = grade
        self.school = school
        
```

```python
son1 = son()
son1.get_name() # output huname
son1.get_age() # output 15
```

----

### input 输入

> **基本使用**

```python
a_input = input()
# 输入结束后按回车
b_input= input("please input a number:")
```

----

### IO

基本语法及其参数

```python
file object = open(file_name, access_model, buffering)
- file_name  文件名
- access_model 访问级别
- buffering 每次缓冲的量

访问级别的选项，help(open)看帮助文档即可
'r'       open for reading (default)
'w'       open for writing, truncating[先清除数据] the file first
'x'       create a new file and open it for writing
'a'       open for writing, appending to the end of the file if it exists
'b'       binary mode
't'       text mode (default)
'+'       open a disk file for updating (reading and writing)
'U'       universal newline mode (deprecated)
```

IO = input output

```python
# 常规写入
text = "Writing a text \n\n hello world"
my_file = open("file.txt","w") # 写入的方式打开文件 [以清空的方式写入]  不记得了就help(open)查下api，写的挺清楚。
my_file.write(text)
my_file.close()
```

```python
# 不用手动关闭IO的写法
with open('a.txt','w') as f: # 无需手动close流
    f.write("1213")
# 循环遍历文件，一次读一行。
with open(filename) as f:
    for line in f:
        print(line)
```

可读可写案例，结合上面的参数即可

```python
file_obj = open('IO_Demo.txt','r+')
file_obj.readline()
file_obj.write("1231231asf\n") # 写入一行数据
file_obj.flush() # 写入后刷新
```

----

### Exception

异常

```python
try:
    with open('aaa','r+') as f:
        f.write(123)
except Exception as e:
    print(e)
else:
    print("No exception")
# output [Errno 2] No such file or directory: 'aaa'
```

> **语法模板**

```python
try:
<语句>        #运行别的代码
except <名字>：
<语句>        #如果在try部份引发了'name'异常
except <名字>，<数据>:
<语句>        #如果引发了'name'异常，获得附加的数据
else:
<语句>        #如果没有异常发生
```

### json存储

```python
import json
dict = {'user_id':'ssf','user_name':'xxks'}
with open('example.json','w') as f:
    json.dump(dict,f) # 把json数据写入f中
    
with open('example.json') as f:
    content = json.load(f)
    print(content)
```

# Numpy

### 入门

```python
import numpy as np

array = np.array([
                    [1,2,3],
                    [4,5,6],
                    [7,8,9]
                ])
print(array)
# 基本上见名知意
print(array.ndim) # 维度 [[ 二维]] 看括号！！
print(array.shape) # 形状
print(array.size) # 形状
print(array.dtype) # 元素类型 dtype data type
```

### 创建array

> **指定元素类型**

```python
import numpy as np
a = np.array([1,2,3],dtype=np.int32) # 指定元素数据类型

b = np.array([1,2,3],dtype=np.float) # 指定元素数据类型
```

> **创建二维矩阵**

```python
c = np.array([
     [1,2,3],
     [1,2,3],
     [1,2,3]
	])
```

> **创建元素全为0的**

```python
# 注：传入的为元组，不可变！
zero = np.zeros((2,3)) # 生成2行3列全为0的矩阵
empty = np.empty((3,2)) # 生成3行2列 接近0的矩阵
```

> **创建元素全为1的**

```python
one = np.ones((3,4)) # 生成3行4列全为1的矩阵
```

> **常用api**

```python
e = np.arange(10) # output [0 1 2 3 4 5 6 7 8 9]

f = np.arange(4,8) # output [4 5 6 7]

g = np.arange(1,20,3) # 1-20 步长为3

h = np.arange(8).reshape(2,4) # 转换形状【重新调整形状】 2行4列
```

### numpy的运算

> **加减乘除**

都是相同位置的进行操作，即按位相除。

```python
a = np.array([[1,2],
             [3,4]])
b = np.array([[1,2],
             [3,4]])
a + b
a- b
a * b
a / b
a**b  # [次幂b中的数为a的次方]
a % b # a中对应的元素 对b中对应元素取余
a // b # 取整
a + 2 # a中的所有元素+2 减乘除都是一样的。
a > 3 # a中的所有元素与3比较，返回值是一个bool 矩阵
```

> **矩阵运算**

```python
np.dot(arr1,arr2) # 矩阵乘法
or
arr1.dot(arr2)

arr1.T # 转置矩阵
np.transpose(arr1) # arr1的转置矩阵

# 也可用@ 进行矩阵乘法 py3.5及以上版本
a @ b

# 其他
np.exp(2) # e^2 e的平方
```

### 随机数生成★

**形式参数说明：**

```python
np.random.xxx(low,high,size,dtype)
```

> **生成从0-1的随机数**

```python
import numpy as np
sample = np.random.random((3,2)) # 3行2列
print(sample)
```

> **生成符合标准正态分布的随机数**

```python
sample2 = np.random.normal(size=(3,2))# 3行2列
```

PS：标准正太分布 N(0，1)； 期望=1，方法差=1

> **生成指定范围的int类型的随机数**

```python
sample3 = no.random.randint(0,10,size=(3,2)) # 3行2列 整数
```

> **生成等份数据**

```python
np.linspace(0,2,9) # 9个数组 从0-2中等份取
```

----

### 元素求和

> **对矩阵内的元素求和**

```python
np.sum(sample3)
```

> **对每一列求和**

**向量 默认是列向量！**

```python
np.sum(sample,axis=0)
```

> **对每一行求和**

```python
np.sum(sample,axis=1)
```

> **最小/大值的索引**

命名风格arg开头

```python
np.argmin(sample3) # 求最小值的索引
np.argmax(sample3) # 最大值的索引
```

> **求均值/中位数/开方**

```python
np.mean(sample3)
sample3.mean()
np.median(sample3)
np.sqrt(sample3)
```

> **排序**

```python
np.sort(sample3) # 每行的数据排序，不同行之间不影响
```

> **修改**

```python
np.clip(sample3,1,4) # 小于1的都变成1 大于4的都变成4
```

### 索引

> **仅一行的话，与python的切片语法一致。**

> **二维**

```python
array = np.array([
    			 [1,2,3],
                  [4,5,6]
				])

array[1] # 取到第一行 与二维数组类似
array[; ,2] # 取每一行的第二列 [3,6] / 每行都取，每列只取第0 1个元素
```

> **迭代**

```python
for i in array.flat: # 迭代每一个元素
    print(i)
    
# 列表推导式
[ print(x) for x in aaray.flat ]
```

----

### 合并★

> **垂直合并【垂直拼接】**

<span style="color:white">vstack  vertical stack【垂直】</span>

```python
import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr3 = np.vstack((arr1,arr2))
# output
# [1 2 3]
# [4 5 6]
```

> **水平合并【水平拼接】**

<span style="color:white">hstack horizontal【水平】</span>

```python
np.hstack((arr1,arr2))
# output
# [1 2 3 4 5 6]
```

> **连在一起**

默认水平连接

与其他拼接的区别：concatenate 合并的array维度要相同，array形状要匹配。

```python
np.concatenate((array1,array2),axis=0) # 垂直合并
np.concatenate((array1,array2),axis=1) # 水平合并
```

> **新增维度**

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

> **维度扩展**

```python
arrs_2 = np.atleast_2d(arrs) # 如果低于2d（2 dim）则会扩充为2dim 反之不改变
```

----

### 分割

> **水平分割**

```python
import numpy as np
arr1 = np.arange(12).reshape((3,4))
print(arr1)
arr2,arr3 = np.split(arr1,2,axis=1) # 水平分割 分2份
```

> **垂直分割**

```python
arr4,arr5 = np.split(arr1,3,axis=0) # 垂直方向 分3份
```

矩阵中的向量一般默认为列向量。所以axis默认为0，垂直方向分割。

> **无法等份切割**

```python
arr6,arr7,arr8 = np.array_split(arr1,3,axis=1) # 水平切割 分三份 不等份分割
```

----

### 深拷贝浅拷贝

> **浅拷贝**

使用同一块内存

> **深拷贝**

使用不同的内存，互不干扰。

```python
arr2 = arr1.copy();
```

---

# matplotlib

主要用pyplot包

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

### 概述

是基于Numpy的，用于数据分析

知识点

- 排列
- 索引
- 交叉表
- 透视表
- 数据探索

可用类似于SQL的方式对 csv tsv xlsx等格式的数据进行处理和分析。

主要使用的数据结构为：Series和DataFrame类

- Series：类似于一维数组对象

  - ```python
    series = [1,2,3,41,123,123]
    
    # output
    # index  value
    #   0      1
    #   1      2
    #   2      3
    #   3     41
    #   4    123
    ```

- DataFrame：类似于二维数据结构
  - key - value
  - key为表头，value为表头的各种取值。value为list数据类型

### Series

### DataFrame

### 选择数据

### 赋值及操作

### 处理丢失数据

### 读取及写入文件

### 合并

### plot



```python
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')
print(data.head())
# 查看形状
print(data.shape, '\n')
# 查看列名
print(data.columns, '\n')
# 查看数据的信息！
print(data.info(), '\n')

print("===================\n\n")
# 修改数据类型
data['Churn'] = data['Churn'].astype('int32')
# 查看数据的信息！
print(data.info(), '\n')

# 显示基本的统计特征 如期望 方差 max min  数据类型为 pandas.core.frame.DataFrame
print(data.describe())
# 显示出类型为 object 和 int32的数据 【显示对应类型的数据】
print(data.describe(include=['object', 'int32']))

# 查看类别和布尔值的特征  对Churn属性的值进行统计
print(data['Churn'].value_counts())
"""
output
0    2850
1     483
Name: Churn, dtype: int64
"""
# 归一化  统计各种类型的比例
print(data['Churn'].value_counts(normalize=True))

print("================ \n\n\n")
# 排序 by=排序字段  ascending 上升的=False  就是降序
print(data.sort_values(by='Account length', ascending=False).head())
print("================ \n\n\n")
# 索引和获取数据
print(data['Churn'])
"""
output
index      value
0            0
1            0
2            0
3            0
"""

# 求均值
print(data['Churn'].mean())

# 布尔索引
# 布尔值索引同样很方便，语法是 df[P(df['Name'])]，
# P 是在检查 Name 列每个元素时所使用的逻辑条件。
# 这一索引的输出是 DataFrame 的 Name 列中满足 P 条件的行。
# 求为真的均值
print(data['Churn'] == 1)
print(data[data['Churn'] == 1].mean())  # 筛选出data中Churn == 1的所有数据的均值

# 组合使用 求均值。 如求离网用户在白天打电话的总时长的均值
# 离网用户 data[data['Churn'] == 1]
# 离网用户中白天打电话 data[data['Churn'] == 1]['Total day minutes']
print(data[data['Churn'] == 1]['Total day minutes'].mean())

# DataFrame 可以通过列名、行名、行号进行索引。loc 方法为通过名称索引，iloc 方法为通过数字索引。
# 打印0-5行、State州至Area code区号的数据
print(data.loc[0:5, 'State':'Area code'])
# 打印前10行，前三列的數據
print(data.iloc[0:10, 0:3])

# 第一行的切片 和第一列的切片 data[行切片，列切片]  看官方文档的解释  当成一个list[list]的切片？
print("This is the first \n")
print(data[:1])  # 首行
print(data[-1:])  # 末行

# 应用函数到单元格、列、行
data.apply(np.max)  # 求每一列的最大值 感觉有点像lambda表达式 传个方法过去

# 把数据中的旧值替换为新值{'old value': 'new value'}
# 把原先的No替换为False 把原先的Yes替换为True
replace_data = {'No': False, 'Yes': True}
data['International plan'] = data['International plan'].map(replace_data)
print(data['International plan'])

"""
分组 group by
"""
print("******************************\n")
columns_to_show = ['Total day minutes', 'Total eve minutes',
                   'Total night minutes']
print(data.groupby(['Churn'])[columns_to_show].describe())
print(data.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max]))
```

