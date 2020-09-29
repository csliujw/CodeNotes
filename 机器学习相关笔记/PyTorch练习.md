# 第一次跑代码

```python
import numpy as np
import torch
# torch自己集成的一些数据集
from torchvision.datasets import mnist
# 预处理模块
import torchvision.transforms as transforms
# 数据加载模块
from torch.utils.data import DataLoader
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

"""
写完后再抽取模块，改为OOP
"""
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
# 迭代次数
num_epoches = 20
# 书中为lr 学习率 应该是loss梯度下降的学习比例
loss_rate = 0.01
# momentum ==>动量
momentum = 0.5
# 预处理函数定义好预处理规则                                                   mean  std
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# 应该是train下载好了 test只需要在指定的目录中读取指定的数据
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

"""
可视化数据源
"""
import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (examples_data, examples_targets) = next(examples)
fig = plt.figure()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(examples_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth:{}".format(examples_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig.show()

"""
构建网络
"""


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden_1, n_hidden_2):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(28 * 28, 300, 100, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=loss_rate, momentum=momentum)

    losses = []
    # acc 准确率 es 复数
    acces = []
    # eval 评价
    eval_losses = []
    eval_acces = []

    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()

        if epoch % 5 == 0:
            # 看不懂
            optimizer.param_groups[0]['lr'] *= 0.5
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            # 忘记了
            img = img.view(img.size(0), -1)
            # 前向传播算loss
            out = model(img)
            loss = criterion(out, label)
            # 后向传播算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录误差
            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        model.eval()
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            # 忘记了
            img = img.view(img.size(0), -1)
            out = model(img)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        eval_loss = 0
        eval_acc = 0

        print("Train Loss", train_loss / len(train_loader))
        print("Train acc", train_acc / len(train_loader))
```

