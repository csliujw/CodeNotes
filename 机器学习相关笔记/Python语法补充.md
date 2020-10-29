# 概述

补充代码中看到的部分语法。

# 语法层面

## 函数形参类型

| 参数类型                  | 说明                                         |
| ------------------------- | -------------------------------------------- |
| def func( arg1, *arg2 )   | 可变长参数，带*号的参数会以元组的形式导入    |
| def func( arg1, **arg2 )  | 可变长参数，带**号的参数以字典的形式传入     |
| def func( a , b , * , c ) | 单独出现* 后的参数必须使用关键字传入，看code |

```python
def func1(*args):
    print(args)

def func2(**args):
    print(args)

def func3(a, *, c):
    print(c)

if __name__ == '__main__':
    func1(1)
    func1(1, 2, 3)
    func2(a=1, b=2)
    func3(1, c=2)
# output
# (1,)
# (1, 2, 3)
# {'a': 1, 'b': 2}
# 2
```

---

## 设置/获取属性

| 方法                       | 说明     |
| -------------------------- | -------- |
| setattr(obj, name,  value) | 设置属性 |
| getattr(obj , name)        | 获取属性 |

```python
class A:
    def __init__(self):
        pass
    
    def set_attribute(self):
        setattr(self, 'number', 5)
        
    def set_loop(self):
        for i in range(10):
            setattr(self, 'fc%i' % i, i)# %i 和 i对应，名字也要一样！！

if __name__ == '__main__':
    a = A()
    a.set_attribute()
    a.set_loop()
    print(a.fc2)

```

# `API`层面

## enumerate

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
# output
#[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
list(enumerate(seasons, start=1))       # 下标从 1 开始
# output
# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

## id

查看python对象的内存地址

```python
id(python_obj)
```

# 高级语法层面

## 魔法函数

# `Numpy`补充

```python
# numpy axis
import numpy as np

np_array = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])
print('我们的数组是：')
print(np_array, '\n')
print('调用 argmax() 函数：')
"""
By default, the index is into the flattened array, otherwise along the specified axis.
查看注释，默认是按展平进行计算的。
按照数组展平后计算索引。 展平后90对应的索引是 7（索引从0开始）
"""
print(np.argmax(np_array), '\n')
print('展开数组：')
print(np_array.flatten(), '\n')  # 展开数组 [30 40 70 80 20 10 50 90 60]
print('沿轴 0 的最大值索引：')
"""
沿0轴就是 沿着x轴。以列向量为基本单位。
[[30 40 70]            30                                40
 [80 20 10]   表示为:   80 最大的值的索引为 1(对应元素80)   20  最大值的索引为 2(对应元素90) 
 [50 90 60]]           50                                90
"""
maxindex = np.argmax(np_array, axis=0)
print(maxindex, '\n')
print('沿轴 1 的最大值索引：')
"""
沿1轴就是 沿着y轴。以行向量为基本单位。
[[30 40 70]             [30,40,70] 最大值的索引为 2(值为70)
 [80 20 10]   表示为:    [80 20 10] 最大值的索引为 0(值为80)
 [50 90 60]]            [50 90 60] 最大值的索引为 1(值为90)
"""
maxindex = np.argmax(np_array, axis=1)
print(maxindex, '\n')
print('调用 argmin() 函数：')
minindex = np.argmin(np_array)
print(minindex, '\n')
print('展开数组中的最小值：')
print(np_array.flatten()[minindex], '\n')
print('沿轴 0 的最小值索引：')
minindex = np.argmin(np_array, axis=0)
print(minindex, '\n')
print('沿轴 1 的最小值索引：')
minindex = np.argmin(np_array, axis=1)
print(minindex)
```

```python
"""
output:
我们的数组是：
[[30 40 70]
 [80 20 10]
 [50 90 60]] 

调用 argmax() 函数：7 

展开数组：[30 40 70 80 20 10 50 90 60] 

沿轴 0 的最大值索引：[1 2 0] 

沿轴 1 的最大值索引：[2 0 1] 

调用 argmin() 函数：5 

展开数组中的最小值：10 

沿轴 0 的最小值索引：[0 1 1] 

沿轴 1 的最小值索引：[0 2 0]
"""
```

