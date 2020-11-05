import numpy as np
from numpy import random
from numpy.core.numeric import load

"""
读写文件
"""
class read_write_file:
    def __init__(self) -> None:
        self.arr = np.arange(100).reshape(10,10)
    
    # 保存单个数组
    def save_as_file(self):
        np.save("./arr_data",self.arr)
        np.savetxt("./arr_data.txt",self.arr)
        print("保存的数组为：\n",self.arr)


    # 只能一次保存多个，能不能追加，循环存取？  npz好像是压缩文件
    def save_as_file2(self):
        np.savez("./arr_data_mutil",self.arr,self.arr)

    # 二进制的读取
    def read_npy(self):
        np_data = np.load("./arr_data.npy")
        print(np_data)
    
    # 多个numpy数组存进去的时候，key-value形式存取，key的规律 arr_0 arr_1
    def read_npz(self):
        np_data = np.load("./arr_data_mutil.npz")
        for key in np_data:
            print(key)


"""
还是读写文件
"""
class save_read:

    def __init__(self) -> None:
        self.arr = np.arange(0,12,0.5).reshape(4,-1)
        print(self.arr)

    def save_read_file(self):
        # fmt 存储的数据格式    delimiter 用什么进行分割
        np.savetxt("./save_read_file.txt",self.arr,fmt="%d",delimiter=",")
        # 元素之间以 ， 为分界
        load_data = np.loadtxt("./save_read_file.txt",delimiter=",")
        print("读取的数据为：\n",load_data)

    # 与loadtxxt相似，面向结构化数组和缺失数据
        load_data = np.genfromtxt("./save_read_file.txt",delimiter=",")
        print(load_data)


class analysis:
    def  __init__(self) -> None:
        np.random.seed(45)
        self.arr1 = np.random.randint(1,10,size=10)
        self.arr2 = np.random.randint(1,10,size=(3,3))
        print(self.arr1)

    def sort_demo(self):
        self.arr1.sort()

    def sort_demo2(self):
        # 沿着橫軸排序（x軸） x轴上的元素是有序的
        self.arr2.sort(axis=1)
        print(self.arr2)

    def sort_demo3(self):
        # 沿着竖轴，  y轴上个的元素是有序的
        self.arr2.sort(axis=0)
        print(self.arr2)

    def sort_arg(self):
        # 根据value 对索引（index）进行排序。返回值为拍好序的索引，原数组的值的位置不会变
        data = np.array([1,3,6,72,324,567,34,567])
        data2 = data.copy()
        index = data.argsort()
        print(data)
        print(index)

if __name__ == "__main__":
    obj = analysis()
    # obj.sort_demo2()
    print("="*50)
    obj.sort_arg()