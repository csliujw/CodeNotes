import numpy as np

"""
numpy 的索引最重要！
"""

class demo:
    def __init__(self):
        self.arr = np.array([[1,2,3,4,5],
                             [4,5,6,7,8],
                             [7,8,9,10,11]])

    def test1(self):
        print(self.arr[1:,3:])

    def test2(self):
        print(self.arr[1:,2:])

    def test_demo1(self):
        # output (0,3)  (1,4)  use tuple
        print(self.arr[((0,1),(3,4))])

    def test_demo2(self):
        # 使用切片和元组组合的方式
        print(self.arr[0:,(1,2,4)])

    def test_demo3(self):
        # 比较 得布尔数组
        print(self.arr >=4)


"""
数组的展平
"""
class demo2:
    def __init__(self) -> None: self.arr = np.array([[1,2,3,4,5],
                                                    [4,5,6,7,8],
                                                    [7,8,9,10,11]])
    
    def reshape_demo(self):
        print(self.arr.reshape(5,3))

    def ravel_demo(self):
        print(self.arr.ravel())

    def flatten_demo1(self):
        # 按行一个一个取
        print(self.arr.flatten())

    def flatten_demo2(self):
        # 按列一个一个取
        print(self.arr.flatten('F'))

"""
数組合并
"""
class demo3:
    def __init__(self) -> None:
        self.arr1 = np.array([[0,1,2,3],
                            [4,5,6,7],
                            [8,9,10,11]])
        self.arr2 = np.array([[0,3,6,9],
                              [12,15,18,21],
                              [24,27,30,33]])
    
    # 横向合并 x轴 （列向量为合并的基本单位）
    def hstack_demo(self):
        print(np.hstack((self.arr1,self.arr2)))

    # 纵向合并 y 轴 （行向量为合并的基本单位）
    def vstack_demo(self):
        print(np.vstack((self.arr1,self.arr2)))
    
"""
分割
"""
class demo4:
    def __init__(self) -> None:
        self.arr = np.array([[0,3,6,9],
                              [12,15,18,21],
                              [24,27,30,33]])

    # h還是以x轴 为基准 划2份 把x轴对半开   只能均分
    def hsplit_demo1(self):
        print(np.hsplit(self.arr,2))

    # v还是以y轴 为基准 划2份 把y轴对半开   只能均分
    def vsplit_demo1(self):
        print(np.vsplit(self.arr,3))
        
"""
简便的合并与分割
"""
class demo5:
    def __init__(self) -> None:
        self.arr1 = np.array([[0,1,2,3],
                            [4,5,6,7],
                            [8,9,10,11]])
        self.arr2 = np.array([[0,3,6,9],
                              [12,15,18,21],
                              [24,27,30,33]])

    # 合并 axis = 0 按y轴方向 以行向量为单位进行合并
    def stack_demo(self):
        print(np.concatenate((self.arr1,self.arr2),axis = 0))
        
    # 分割 axis=0 按y轴方向 以列向量为单位进行拆分 只能均分
    def split_demo(self):
        print(np.split(self.arr1,3,axis=0))

if __name__ == "__main__":
    obj = demo5()
    obj.stack_demo()
    print("="*50)
    obj.split_demo()
    
