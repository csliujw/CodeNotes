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