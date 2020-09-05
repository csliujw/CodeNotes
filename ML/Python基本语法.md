# 概述

> **Python基础语法入门**，便于后期用python写代码

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

> 定义

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

### input

输入

> **基本使用**

```python
a_input = input()
# 输入结束后按回车
b_input= input("please input a number:")
```

----

### IO

IO = input output

```python
text = "Writing a text \n\n hello world"
my_file = open("file.txt","w") # 写入的方式打开文件 [以清空的方式写入]
my_file.write(text)
my_file.close()
```

```python
with open('a.txt','w') as f: # 无需手动close流
    f.write("1213")
```

> **open的参数**

open(args1,args2)

args2的取值

- w【write】 清空再写入
- a【append】 追加写入
- r【read】读取文件内容
  - file.read() 读取全部内容
  - file.readline() 只读一行
  - file.readlines() 读取所有行，存入一i个列表，并返回这个列表。
- r+ 读取到了文件，就写入，读不到就报异常。

```python
# 循环遍历文件，一次读一行。
with open(filename) as f:
    for line in f:
        print(line)
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

array = np,array([
                    [1,2,3],
                    [4,5,6],
                    [7,8,9]
                ])
print(array)

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

h = np.arange(8).reshape(2,4) # 转换形状 2行4列
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

```

### 随机数生成★

> **生成从0-1的随机数**

```python
import numpy as np
sample = np.random.random((3,2)) # 3行2列
print(sample)
```

> **生成符合标准正太分布的随机数**

```python
sample2 = np.random.normal(size=(3,2))# 3行2列
```

PS：标准正太分布 N(0，1)； 期望=1，方法差=1

> **生成指定范围的int类型的随机数**

```python
sample3 = no.random.randint(0,10,size=(3,2)) # 3行2列 整数
```

### 元素求和

> **对矩阵内的元素求和**

```python
np.sum(sample3)
```

> **对每一列求和**

```python
np.sum(sample,axis=0)
```

> **对每一行求和**

```python
np.sum(sample,axis=1)
```

> **最小值的索引**

```python
np.argmin(sample3) # 求最小值的索引
```

> **最大值的索引**

```python
np.argmax(sample3)
```

> **求均值**

```python
np.mean(sample3)
sample3.mean()
```

> **求中位数**

```python
np.median(sample3)
```

> **开方**

```python
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

> 仅一行的话，与python的切片语法一致。

> **二维**

```python
array = np.array([
    			  [1,2,3],
                  [4,5,6]])

array[1] # 取到第一行 与二维数组类似
array[:,2] # 取每一行的第二列 [3,6]
```

> **迭代**

```python
for i in arrar.flat: # 迭代每一个元素
    print(i)
```

----

### 合并★

> **垂直合并【垂直拼接】**

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

```python
np.hstack((arr1,arr2))
# output
# [1 2 3 4 5 6]
```

> **连在一起**

```python
np.concatenate((array1,array2),axis=0) # 垂直合并
np.concatenate((array1,array2),axis=1) # 水平合并
```



### 分割

### 深拷贝浅拷贝

