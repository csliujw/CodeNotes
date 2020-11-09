import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation
from FCN import FCN
import cfg

# 通用代码，用cuda还是cup
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

# 训练数据
Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
# 验证集
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

# PyTorch读数据的函数 batch_size 一次读取几张图片 num_workers 线程数
train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

# FCN 12是类别数目
fcn = FCN(12)
# 把fcn放在设备上跑
fcn = fcn.to(device)
# 损失函数  和 交叉熵函数差不多，但是这个损失函数没有封装 softmax
# 要保证模型和损失的计算环境是一样的。用NLLLoss自定义激活函数的时候 改比较方便
criterion = nn.NLLLoss().to(device)

# 2d数据集试试Adam，rgbd试试SGD
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def train(model):
    # 保存最好的权重
    best = [0]
    # 训练
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        #  print常用的打印方式记一下
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        # 每 50次降低一次学习率
        if epoch % 50 == 0 and epoch != 0:
            # 更改学习率
            for group in optimizer.param_groups:
                # 学习率  下降 1/2
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        # 训练批次 i->索引  sample->字典
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = sample['img'].to(device)   # [4, 3, 352, 480]
            img_label = sample['label'].to(device)    # [4, 352, 480]
            # 训练 out为预测图
            out = net(img_data)     # [4, 12, 352, 480]
            # 用的NNLoss 没有softmax分类  所以要手动进行一次分类。
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            # 梯度清零，不清零的话会进行累加
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估  求预测图的最大值 dim=1 y轴方向， 行向量为基本单位
            # 他会返回两个之 一个是值本身 一个是值所在的索引
            # 固定写法 记住哦！
            pre_label = out.max(dim=1)[1].data.cpu().numpy()    # (4, 352, 480)
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()   # (4, 352, 480)
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )

        print(metric_description)
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), '{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou
    # eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(fcn)

