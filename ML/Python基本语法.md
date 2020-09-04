# 概述

> **Python基础语法入门**，便于后期用python写代码

# 基本语法

### **基本数据类型**

- Integer： integer转string  str(123)
- float
- string

### **运算**

- +加

- -减

* *乘

- /除  5/3 = 1.666667 和 java不一样

- %取余

- // 取整
- 3**2 = 3的2次方
- 2**4 = 2的四次方

### **prinf**

```python
print(123)  # output 123
print(1+2) # output 3
print(1,2) # output 1 2
```

### **循环**

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

### **布尔**

True False 首字母大写

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

### 模块

### 类的继承

### input

### IO

### Exception

