class list_demo:
    def __init__(self) -> None:self.arr = [[1,2,3,4],[5,6,7,8]]

    def test1(self):
        # 无法使用 arr[0:1,2:3]这种形式，报错提示是：
        # TypeError: list indices must be integers or slices, not tuple
        print(self.arr[0:1])


if __name__ == "__main__":
    obj = list_demo()
    obj.test1()