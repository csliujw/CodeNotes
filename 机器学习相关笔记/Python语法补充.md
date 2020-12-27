# 述

补充代码中看到的部分语法。

# 博客中看到的

- `*args`；`**kwargs`；`glob`；`uuid`；`serialization序列化与反序列化`

```python
"""带任意数量参数的函数"""
import torch
import glob
import os
import uuid
import pickle

"""第一个的写法不明所以，还以为是可变长参数"""
def test1(*args):
    for index, emelent in enumerate(args):
        print(f"index={index}, elements is {emelent}, element type is {type(emelent)}")


"""字典"""
def test2(**kwargs):
    # 字典的迭代
    for key, value in kwargs.items():
        print(f"key is {key}, value is {value}")


def test3():
    # 只查找当前目录匹配到的以 *py结尾的文件名. 这个在一个github 关于cv的开源项目里看到过~
    files = glob.glob("*.py")
    # 把它转成绝对路径
    realPath = [os.path.realpath(path) for path in files]
    print(realPath)


"""生成uuid，随机字符~"""
def test4():
    print(uuid.uuid1())


"""
序列化，可以理解为按某种规则的持久化存储,dumps变成pickle对应格式的文件后，再用io存储起来即可~ 反序列化时，io读取内容，然后loads加载还原数据
"""
def serialization():
    print("===================start serialization===================")
    target = {'a': 1, 'b': 2, 'c': 3}
    dumps_result = pickle.dumps(target)
    print(f"dumps result is {dumps_result}")
    loads_result = pickle.loads(dumps_result)
    print(f"loads result is {loads_result}")
    print("===================end  serialization===================")


if __name__ == '__main__':
    # *args
    test1(torch.tensor(1), torch.tensor(2), torch.tensor(3))
    # **kwargs
    test2(a=1, b=2, c=3)
    test3()
    test4()
    serialization()
```



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

## `__new__`

python对象的初始化：

- `__new__`进行内存空间分配
- `__init__`进行数据初始化操作

可以通过`__new__`实现单例模式

## 对象

- 类对象
- 实例对象

类是一个特殊的对象`--------`类对象。

在程序运行的时候，类对象在内存中只有一份，使用一个类可以创建出很多个对象实例。（和Java的字节码对象很像？）

类对象有自己的属性和方法

- 类属性
- 类方法

通过类名.的方式可以访问类的属性 或者 调用类的方法。

```python
"""类方法"""
class Tool(object):
    # 类属性
    instance = None
    count = 0

    def __init__(self):
        Tool.count += 1

    @classmethod
    def show_tool_count(cls):
        print("当前工具类的数目%d" % cls.count)


if __name__ == '__main__':
    t1 = Tool()
    t2 = Tool()
    t3 = Tool()
    Tool.show_tool_count()
```

## 单例模式

```python
class Tool(object):
    # 类属性
    instance = None

    def __init__(self):
        print("!23")

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


if __name__ == '__main__':
    t1 = Tool()
    t2 = Tool()
    t3 = Tool()
    print(t1, t2, t3)
```

## 捕获异常

```python
try:
    # 尝试代码
    pass
except:
    # 提示错误内容
    
==============
try:
    # 尝试代码
    pass
except 错误类型1：
	pass
except (错误类型2, 错误类型3):
    pass
except Exception as result:
    print("未知错误 %s" % result)
===============
try:
    # 尝试代码
    pass
except 错误类型1：
	pass
except (错误类型2, 错误类型3):
    pass
except Exception as result:
    print("未知错误 %s" % result)
else:
    # 没有异常才执行
finally:
    # 无论是否有异常都会执行
```

## 主动抛出异常

- Python提供了一个Exception异常类
- 在开发时，如果满足特定业务需求时，希望抛出异常，可以：
  - 创建一个Exception对象。
  - 使用raise关键字抛出异常对象。

```python
# 这块的操作比较迷~
def throw_exception():
    pwd = input("please input you password")
    if len(pwd) >= 8:
        return
    exception = Exception("捕获异常,密码长度不够")
    raise exception


if __name__ == '__main__':
    throw_exception()
```

## 模块

模块名就是文件名。

```import moduleName```导入模块中所有的类，方法，变量等

```from moduleName import 类``` 从模块中导入具体的一个类

```python
import moduleName # 导入模块中的所偶工具 -- 全局变量、函数、类
from moduleName import xx # 导入模块中的xx工具 导入后无需 模块名.工具
from moduleName import * # 导入模块中的所有工具
```

- 模块名称相同的话，用别名进行标识

**模块搜索顺序**

如果在当前目录则直接导入，没有则再搜索系统目录。

python中的每个模块都有一个内置属性`__file__`可以查看模块的完整路径

这点上，没有Java的安全。Java的类加载机制保证了类加载，和类库调用的安全！！

```python
print(random__file__)
```

```python
"""代码风格"""
# 导入模块
# 定义全局变量
# 定义类
# 定义函数

def main():
    """测试代码"""
    pass

def if __name__ == '__main__':
    main() # 测试模块的作用！
```

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

