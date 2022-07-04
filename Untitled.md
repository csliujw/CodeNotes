# E组实验

E 组实验是在 UDA（UDA 加上 1/16 的 labeled target 数据后）的基础上增加了这些损失项：

- 原型预测结果和网络预测结果的一致性损失
- ProDA 的正则化损失

> 原型计算方式：用的主分类器的预测结果计算的原型

在网络训练前，先计算每个类别的原型，原型的计算方式如下：feat\*(mask\*pred) ==> 找出 ground truth 和网络预测结果的交集，求交集部分特征的均值，作为当前类别的原型。

```python
def calculate_mean_vector(self, feat_cls, outputs, labels_val=None):
    """
    feat_cls：特征
    outputs：模型最后的输出
    其实，feat_cls 和 output 我都是取得模型最后的输出结果，即 batch*7*512*512 的结果
    """
    outputs_softmax = F.softmax(outputs, dim=1)
    outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
    N,C,H,W = outputs_argmax.shape

    outputs_argmax = self.process_label(outputs_argmax.float())

    if labels_val is None:
        outputs_pred = outputs_argmax
    else: 
        N,H,W = labels_val.shape
        labels_val = labels_val.reshape(N,1,H,W)
        labels_val[labels_val==-1]=7 # 排除背景类
        labels_expanded = self.process_label(labels_val.float()) # 处理成 one-hot 编码
        outputs_pred = labels_expanded * outputs_argmax # 取 mask 和 pred 的交集

        scale_factor = F.adaptive_avg_pool2d(outputs_pred.float(), 1) # 平均池化求出 outputs_pred 中各个类别取了多少个像元
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):# image num
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                    if (outputs_pred[n][t] > 0).sum() < 10:
                        continue
                        s = feat_cls[n] * outputs_pred[n][t] # 求类别第 n 个图中类别 t 中的原型 
                        s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t] # 求出当前 样本 类别 t 的原型
                        vectors.append(s)
                        ids.append(t)
                        return vectors, ids # 存好，后面求均值
```

后面，在模型不断的训练过程中，用移动平均更新原型，更新的方式是

old_prototype = old_prototype\*(1-0.01) + 当前 labeled target 计算出的原型 \* 0.01 ，moment 我最后取得是 0.01

ProDA 原型的移动平均更新做法是，用一个动量编码器对整张图（没有裁剪的图）进行预测，然后计算当前 batch 的原型。而我没有用整张图，用的是随机裁剪后 512\*512 大小图片的特征求的原型，所以我的代码跑起来比较快。

> ProDA 正则化损失

正则化类型用的是默认的 MRKLD，且只计算主分类器的正则化损失

```python
def regular_loss(activation,regular_type='MRKLD'):
    logp = F.log_softmax(activation, dim=1)
    if regular_type == 'MRENT':
        p = F.softmax(activation, dim=1)
        loss = (p * logp).sum() / (p.shape[0]*p.shape[2]*p.shape[3])
    elif regular_type == 'MRKLD':
        loss = - logp.sum() / (logp.shape[0]*logp.shape[1]*logp.shape[2]*logp.shape[3])
    return loss
```

> 一致性损失：只计算主分类器和原型预测结果的损失

```python
def prototype_consistance_loss(proto,out,feat,class_numbers,type="ProDA"):

    N, C, H, W = out.shape
    feat_proto_distance = -torch.ones((N, class_numbers, H, W)).cuda()

    for i in range(class_numbers):
        # F.consine_similarity 计算的结果在 -1~1 之间，此处乘以 20. 便于优化。从 PANET 这篇论文和我调试的结果来看，乘以 15~20 的话
        # feat_proto_distance 和 out 对应像元的值会比较接近，便于优化。
        feat_proto_distance[:, i, :, :] = F.cosine_similarity(proto[i].reshape(-1,1,1).expand(-1, H, W).cuda(),feat,dim=1)*20
        # 计算两个结果的一致性。（model 的输出作为伪标签）
        loss = F.cross_entropy(feat_proto_distance,out.argmax(dim=1))
        return loss
```

> 各个损失的权重设置

```python
loss = loss_gt + loss_pseudo*0.5 + loss_lt + proto_consistent_loss*0.5 + reg_loss*0.1
```

- loss_gt， 源域损失
- loss_pseudo，unlabeled target 损失
- loss_lt，labeled target 损失
- proto_consistent_loss，原型一致性损失
- reg_loss，正则化项

> 结果

不加一致性损失最高的 IoU：45.76

E 组实验目前最高的 IoU：45.96165

还有就是，不加一致性损失的实验，在 Val 上的 mIoU 特别高，到了 57.x；而 E 组实验的 mIoU 只有 51.7 左右。

加入原型对齐损失后，在 Val 上的 mIoU 57.x。