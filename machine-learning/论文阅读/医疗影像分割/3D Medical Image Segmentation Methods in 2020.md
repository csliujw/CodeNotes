# 2020 3D Medical Image Segmentation Methods
## Introduction

10个顶级医疗影像分割。五个单模态的，五个多模态的。

- 单模态：图像分割任务，包括3个CT图像分割任务和2个MR图像分割任务
- 多模态：图像分割任务，包括两个双模态任务、两个三模态任务和一个四模态任务。

<img src="..\..\pics\CV\Medical\image-20210619210707617.png">

> 本文主要贡献

提供了最近十个国际三维医学图像分割挑战的综合回顾，包括任务描述、数据集，更重要的是，参与者团队的顶级解决方案，这些解决方案代表了目前最前沿的分割方法.

我们在顶级方法中识别广泛使用的“happy-families”组件，这对于开发强大的分割方法很有用。

总结了几个尚未解决的问题和潜在的研究方向，以期促进医学图像分割领域的发展。

## preliminaries

预备知识。

### Network Architectures

==nnU-Net不是一个新的Net，而是一个动态的全自动医学图像分割框架==，它基于广泛使用的U-Net架构。它可以为任何新的分段任务自动配置预处理、网络架构、训练、推理和后处理。在没有人工干预的情况下，nnUNet超越了大多数现有方法，在53个细分任务中的33个任务中实现了最先进的性能，并且在其他方面显示出与顶级排行榜条目相当的性能。==目前，nnU-Net已经成为最流行的3D医学图像分割任务的主干，因为它功能强大、灵活、开箱即用和开源==

### Loss functions

损失函数用于指导网络学习有意义的预测，并指示网络如何权衡错误。==Cross entropy损失和Dice损失是分割任务中最流行的两种损失函数==。具体来说，交叉熵旨在最小化两个分布之间的差异，其定义如下：
$$
L_{CE} = -\frac{1}{N} \sum^{C}_{c=1}\sum^{N}_{i=1}g^c_ilogs^c_i ------(1)
$$
$g^c_i$是真实二进制类别x像素的标签c，$s^c_i$是对应的预测概率。（意思就是，一张图有很多需要分割的类别，A1类别对应c1标签；A2类别对应c2标签）

Dice loss可以直接优化最常用的分割评估指标Dice Similarity Coefficient（DSC）。

一般来说，Dice loss有两种变体：

- 一种变体是在分母中使用平方项，定义如下：

$$
L_{Dice-square}=1-\frac{2\sum^C_{c=1}\sum{^N_{i=1}g^c_i}s^c_i}{\sum^C_{c=1}\sum{^N_{i=1}(g^c_i)^2+\sum^C_{c=1}\sum^N_i(s^c_i)^2}}------(2)
$$

- 另一个不使用分母中的平方项，该分母由下式定义：

$$
L_{Dice}=1-\frac{2\sum^C_{c=1}\sum{^N_{i=1}g^c_i}s^c_i}{\sum^C_{c=1}\sum{^N_{i=1}g^c_i+\sum^C_{c=1}\sum^N_is^c_i}}------(2)
$$

nnU-Net中的默认损失函数是：$L_{CE}+L{Dice}$

### 评估指标

Dice Similarity Coefficient (DSC) 和Hausdorff Distance (HD) 是两种广泛使用的分割度量，可以分别度量区域重叠率和边界距离。设G和S分别为地面真值和分割结果。DSC的定义如下：
$$
DSC = \frac{2|G∩S|}{|G|+|S|}------（4）
$$
有时候用IoU替代DSC：
$$
IoU = \frac{|G∩S|}{G∪S}------（5）
$$
设$\partial G和\partial S$分别是ground truth和segmentation的边界点。Hausdorff Distance 定义为：
$$
HD(\partial G,\partial S) = max(hd(\partial G,\partial S),hd(\partial S,\partial G))------(6)
$$

$$
hd(\partial G,\partial S) = max_{x∈ \partial G} min_{y∈ \partial S} ||x-y||_2
$$

$$
hd(\partial S,\partial G) = max_{x∈ \partial S} min_{y∈ \partial G} ||x-y||_2
$$

To eliminate the impact of the outliers, 95% HD is also widely used, which is based on the calculation of the 95th percentile of the distances between boundary points in $\partial G$ and $\partial S$.

## Single Modality

### CADA:脑动脉瘤分割

从3D  CT图像中分割动脉瘤。主要困难是标签的高度不平衡。如图2(第一行)所示，动脉瘤非常小，大多数体素是CT图像中的背景。

<img src="..\..\pics\CV\Medical\image-20210619214952004.png">

采用6个指标对分割结果进行定量评价，包括Jaccard (IoU)、Hausdorff距离(HD)、平均距离(MD)、所有动脉瘤预测体积与参考体积的皮尔逊相关系数(Volume Pearson R)、预测体积与参考体积的平均绝对差(Volume Bias)、预测体积与参考体积之差的标准差(Volume Std)。对于排名，根据所有参与者执行最大-最小归一化。这样，每个单独的度量取一个介于0(所有参与者中的最坏情况)和1(参考和预测分割之间的完美匹配)之间的值。排名分数计算为标准化指标的平均值。

==最佳方法的做法：==在nnU-Net的基础上改进，其中主要的修改是在训练和推理过程中使用大的补丁大小($192*224*192$)。五个模型在五重交叉验证中训练，每个模型在TITAN  V100 32G GPU上训练。每一个测试用例都是通过训练好的五个模型的集合来预测的。$max_{iou} = 0.759$

### ASOCA:冠状动脉的自动分割

共60个数据。主要困难是不平衡的问题和外观的变化。一方面，冠状动脉在整个CT图像中只占很小的比例。另一方面，健康和不健康病例的动脉有不同的外观。图2(第二行)展示了一个可视化的例子。用DSC和HD95对分割结果进行评价和排序。

==最佳方法的做法：==

- $max_{DSC} = 0.867$：nnU-Net为backbone，整个管道包括三个独立网络，用于三个任务:心外膜分割、动脉分割和比例图回归。最终的分割结果由动脉分割结果和比例图回归结果的集合生成，然后去除心外膜外的血管。
- $min_{HD95}=2.336$：提出了一种改进的具有选择性核的 2D U-Net ，其中在编码部分，常规的卷积块被 SE-Res 模块替代。在解码器部分使用了不同的convolution filters 和 kernel size，以利用多尺度信息。（**特征金字塔？**）

### VerSe：大规模椎骨分割挑战

难点：病例视野(FoV)高度变化、扫描尺寸大、相邻椎骨形状高度相关、扫描噪音、椎骨骨折、金属植入物等。使用DSC和HD对分割结果进行评估和排名。

<img src="..\..\pics\CV\Medical\image-20210619222122067.png">

==最佳方法的做法：==SpatialConfiguration-Net and U-Net，提出了一种由粗到细的方法。包括三阶段。

- stage 1：通过基于热图回归的3D U-Net定位整个脊柱，该网络可以去除背景；网络输入大小在$32*32*32 \ to \ 128*128*128$之间。
- stage 2：通过3D空间配置网络同时定位和识别所有椎骨标志，该网络将标志的局部外观与空间配置相结合；训练时网络输入大小为$64*64*64 -> 96*96*256$，推理时为$128*128*448$。为了解决错过的椎骨，一个基于核磁共振的图形模型被用来改进定位结果。
- stage 3：用3D U-Net分割每块椎骨。输入大小是$128*128*96$

