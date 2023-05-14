[知乎简介](https://zhuanlan.zhihu.com/p/337479181)

[论文地址](https://arxiv.org/pdf/2011.09763.pdf)

# 摘要

生物医学图像分割，细胞分割是瓶颈。

Attention-based transformers are state-of-the-art in a range of deep learning fields. 

基于注意力的transformers在深度学习领域效果表现很好，`SOTA`。

将注意力机制加分割中，效果很好，比其他方法更优秀。

因此，本文提出了一种基于注意力机制的`Cell DETR（细胞变换检测器）`，可进行端到端的实例分割，在特定数据下，分割效果与`MaskRCNN`相当，且速度更快。

| name       | iou  | time   |
| ---------- | ---- | ------ |
| Mask R-CNN | 0.84 | 29.8ms |
| Cell-DETRA | 0.83 | 9.0ms  |
| Cell-DETRB | 0.84 | 21.2ms |

`思路来源：``“End-to-end object detection with transformers,” arXiv:2005.12872, 2020.`  提出了一种新颖的`基于注意力的transform DETR`用于`全景分割`，这个方法简单且高效。可以考虑实际应用。

# 简介

Instance segmentation is a major bottleneck in quantifying single cell microscopy data and manual analysis is prohibitively labour intensive （实例分割是量化单细胞显微镜数据的主要瓶颈，手工分析过于费力。）

注意力机制方法，如最近提出的`detection transformer DETR`（用它做的检测分支），正日益超过其他方法。

- 本文提出了Cell-DETR，一种用于生物医学样本实例分割的基于注意力的detection transformer（a novel attention-based detection transformer for instance segmentation of biomedical samples based on DETR ）

- 解决了分割瓶颈
    - 我们解决了微结构化环境中酵母细胞的自动细胞实例分割瓶颈（图1），

![image-20210412175739872](..\..\pics\CV\ISG\Untitled\image-20210412175739872.png)

- 介绍以前的分割方法
- 介绍显微镜实验装置、测试过的架构以及train和evaluation方法在
- 分析方法性能，与实例分割baseline进行对比，超越了之前的方法（`fps或iou精度`），可进行在线检测。

# 背景

Background

`UNet`效果好！为了实例分割要对分割图进行额外的处理！NLP里，基于注意力机制的方法在CV中大放异彩！最近，一个transformer-based method（DETR）用于物体检测和全景分割可以与Faster R-CNN媲美，这给了我们新的希望，可以进一步改进自动目标检测和分割性能（automated object detection and segmentation performance`）【我觉得更多的是性能上的改进】`

减少了细胞观察的环境，怎么处理的，杂七杂八的。

# 方法

## 细胞标标注

需要分割的细胞周围会有一些trap细胞！！

![image-20210412203348924](..\..\pics\CV\ISG\Untitled\image-20210412203348924.png)

带背景（浅灰色）的注释用于语义分段训练，例如U-Net。 对实例分割训练我们引入了无对象类∅来代替背景类。

----

来自各种实验的带注释的419个样本图像集被随机分配用于网络训练，验证和测试（分别为76％，12％和12％）

trap instances in shades of dark grey, cell instances in shades of violet and transparent background; 

trap实例细胞用灰色，细胞实例用紫色（violet）

## Cell-DETR架构

> 概述

提出了基于`DETR`全景分割架构的`Cell-DETR`模型A和模型B

`DETR`模型与`Cell-DETR A/B`模型的区别：

- 激活函数
- 卷积核
- `参数量少了一个数量级！`

![image-20210412205421479](..\..\pics\CV\ISG\Untitled\image-20210412205421479.png)

`Cell-DETR`变体的参数比原始参数大约少一个数量级。

> `Cell-DETR`网络结构概述！★★★★★

They are the backbone CNN encoder, the transformer encoder-decoder, the bounding box and class prediction heads, and the segmentation head.

- 主要的网络内容如下
    - CNN编码器
    - transformer编码-解码器
    - 包围盒（bounding box），论文中好像说了，没有明显的检测框？？？
    - 类预测头（class prediction heads）
    - 分割头（segmentation head）

> 网络结构图！★★★★★

![image-20210412210336781](..\..\pics\CV\ISG\Untitled\image-20210412210336781.png)

- 主干网络：CNN编码器，提取图像特征。
    - 用的四个类似于`ResNet`的，64、128、256、256的四个卷积块。每个块后用$2*2$的平均池化（average pooling）进行下采样。`Cell-DETR`采用不同的激活函数核卷积，详情见表一。
- transformer编码-解码器：确定图像特征之间的关注度。
    - The transformer decoder predicts the attention regions for each of the N = 20 object queries.
    - transformer解码器为每20个对象查询预测一次他们的注意力区域。
    - 我们把transformer encoder blocks 减少到了三个，decoder blocks减少到了两个，每个前馈神经网络中有512个hidden features.
    - 128的backbone feature在输入transformer前会被展平（展平不就失去了空间信息？）`与最初的DETR相反`，我们`使用了学习过的位置编码`
    - 包围盒和分类的预测头都是前馈神经网络。他们将transformer encoder-decoder输出映射到bounding box和classification prediction。【应该就是`Fig.5` 右侧下方的部分】。这些`FFNN（前馈神经网络）`并行处理每个查询，并共享查询中的参数。（除了细胞和trap类，classification head还可以预测无对象类∅）

- segmentation head（分割头）：由multi-head注意力机制 和 CNN 解码器组成，用于预测每个对象实例的分割。
    - 在transformer encoder and decoder features上我们使用的是二维的multi-head注意力机制。
    - 注意力机制图的结果按通道连接到图像特征上，然后输入到CNN解码器。（The resulting attention maps are concatenated channel-wise onto the image features and fed into the CNN decoder.）
    - 三个类似`ResNet`的解码器块减小了特征通道的大小，同时增加了空间维度。
    - 在`CNN encoder` 和 `CNN decoder block` 输出间使用了跳跃连接。
        - 这些特征在`Cell-DETR A`模型中通过元素相加进行融合
        - 在`Cell-DETR B`中通过像素自适应卷积进行融合
        - 第四个卷积块将查询合并特征维度中，并为每个查询返回原始输入空间维度。所有查询的softmax保证了不重叠的分割

## 模型训练

- 联合损失函数（combined loss function）。[ 多任务的网络 ]

- direct set prediction

    - the set prediction
    - $\hat{y} = \{\hat{y}_i = \{ \hat{p}_i, \hat{b}_i, \hat{s}_i \}\}^{N=20}$
    - 包括对类别概率的相应预测
        - $\hat{p}_i∈R^k$  ；$\hat{p}_i$ 属于类别k的概率；k=3，因为这里仅三类（no-object，trap，cell）
        - $\hat{b}_i∈R^4$；包围盒的概率
        - $\hat{s}_i∈R^{128*128}$；`这里没看懂~`

- 我们用匈牙利算法将每个实例集标签yσ(i)分配给相应的查询集预测，索引σ识别标签的最佳匹配排列

- 联合损失函数的公式

    ![image-20210413185013503](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413185013503.png)

    - $L_p$ 分类损失。
    - $L_b$ 包围盒损失，只算非空对象的。
    - $L_s$ 分割损失，只算非空对象的。

- 我们使用分类加权交叉熵来计算分类损失，权重为β=[ 0.5, 0.5, 1.5 ] K=3时（noobject，trap，cell classes）

    ![image-20210413185118585](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413185118585.png)

- 包围盒损失本身由两个加权损失项组成。【文献33】

    ![image-20210413190232224](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413190232224.png)

    - $λ_J$= 0.4
    - $λ_{L1}$= 0.6
    - These are a generalised intersection-over-union $L_J$ 

- 分割损失是焦点损失（focal loss）和Sorensen Dice loss的加权和。【文献6 8】

    ![image-20210413190206025](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413190206025.png)

    - $λ_F$= 0.05 and $λ_D$= 1；focusing parameter γ = 2 and ? = 1 for numerical stability

# 实验结果

- 分割上，B模型的效果稍好。

- bounding box和分类性能 B模型和A模型效果一样。

    ![image-20210413181914706](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413181914706.png)

- 模型B分割性能稍好，但是参数量稍高
    - 模型A参数量：$4*106$；运行时间`9ms`
    - 模型B参数量：$5*106$；运行时间`21.2ms`

## 与SOTA比较

- UNet
- Mask R-CNN

![image-20210413182338557](..\..\pics\CV\ISG\Attention-Based Transformers for Instance Segmentation of Cells in Microstructures\image-20210413182338557.png)

- UNet速度快，但是实例检测还需要进一步的操作，这个实例检测耗时20ms左右。
- Mask R-CNN可以实例分割，但是速度慢`29.8ms`
- `Cell-DETR 9ms-21.2ms`

## 讨论

`Cell-DETR`对比Mask R-CNN，速度更快，更简单，更易实现。

`Cell-DETR`不依赖于明确的region proposals，但它确实利用了突出潜在空间中相关特征的注意力映射

# 参考

这些论文稍微看下

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

[loss for bounding box regression](https://arxiv.org/pdf/1902.09630.pdf)

*arXiv:2003.07853*