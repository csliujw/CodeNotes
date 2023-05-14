# ERFNet

## 概述

ERFNet: Effificient Residual Factorized ConvNet for Real-time Semantic Segmentation

Factorized Convolution Operator（因式分解的卷积操作）

卷积层提速Factorized Convolutional Neural Networks：作用为简化卷积运算。

<img src="https://img-blog.csdn.net/20160907101751622">

[卷积提速方式](https://blog.csdn.net/shenxiaolu1984/article/details/52266391?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=1328641.7666.16155186513616605&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)

## 摘要

目前的语义分割方法无法同时做到高质量和平衡计算资源。很难运用到实时应用中。本文提出的网络架构，速度快，精度还不错。本文网络架构的核心是使用残差网络的factorized convolution构建一个新的网络层，这个新的网络层可以同时保证效率和较好的精度（fps高，精度还不错，70多的miou）。

## 引言

语义分割难，它把密集像素精度与多尺度的上下文推理结合起来。但是，在最新的分割架构中，高质量和计算资源之间尚未取得良好的折衷。 最近，提出的残差层在ConvNets设计中已成为一种新趋势。 他们对卷积层的重新设计避免了深度体系结构的降级问题，这使得最近的工作可以在堆叠大量层的网络上实现很高的精度。 这种策略已普遍用于在图像分类挑战和语义分割挑战中都获得最高准确性的新架构中。尽管取得了这些成就，但我们认为这种设计策略并不是在准确性和效率之间取得良好折衷的有效方法。考虑到合理的图层数量，通过更多卷积来扩大深度只会在精度上获得很小的收益，同时会显着增加所需的计算资源。 <span style="color:green">（残差网络效果好，但是计算量大，计算量加大了很多，但是精度的提升却很小。所以认为最近提出的残差层并不适用于对速度和精度要求高的实时应用）</span>

现在的一些方法，旨在开发可以实时运行的高效架构。 但是，这些方法通常集中于通过积极减少参数来获得这种效率，这极大地损害了准确性。

我们提出了一种实时、准确的卷积网络ERFNet（Efficient Residual Factorized Network）。核心在于设计了一种新的网络层，<span style="color:red">用一维卷积核进行跳跃连接和卷积 that leverages skip connections and convolutions with 1D kernels。</span>跳跃连接允许卷积学习有助于训练的残差函数，一维分解的卷积可以显着降低计算成本，同时保持与2D相比相似的准确性。我们用我们所提出的网络块顺序的构建我们编码节码架构，产生端到端的分割结果。（图1为示例结果）

<img src="../../pics/CV/ISG/ERFNet/image-20210312115151250.png">

<img src="../../pics/CV/ISG/ERFNet/image-20210312115333134.png">

​				<img src="../../pics/CV/ISG/ERFNet/image-20210312115349713.png" > 



$$
Non-bottlenck：(3*3*w+1)*w+(3*3*w+1)*w = 18w^2 + 2w = 18w_0^2 + 2w_0
$$

$$
Bottlenck：(11w+1)\frac{w}{4} +(33\frac{w}{4} + 1)\frac{w}{4} + (11\frac{w}{4} + 1)*w = \frac{17w^2}{16}+ \frac{3w}{2} = 17w_0^2+ 6w_0
$$

$$
Non-bt-1D：(3*1*w+1)*w+(1*3*w+1)*w+(3*1*w+1)*w+(1*3*w+1)*w=12w^2+4w=12w_0^2+4w_0
$$





## 相关工作

<span style="color:red">**补充知识：Residual Block的设计**</span>

$F(x)+xF(x)+x$构成的block称之为**Residual Block**，即**残差块**，如下图所示，多个相似的Residual Block串联构成ResNet。

<img src="https://s2.ax1x.com/2020/02/21/3uUio4.png">

​		一个残差块有2条路径$F(x)和x$，$F(x)$路径拟合残差，不妨称之为残差路径，$x$路径为identity mapping恒等映射，称之为”shortcut”。**图中的⊕为element-wise addition，要求参与运算的$F(x)$和$x$的尺寸要相同**。所以，随之而来的问题是：

- 残差路径如何设计？
- shortcut路径如何设计？
- Residual Block之间怎么连接？

​		在原论文中，残差路径可以大致分成2种，一种<span style="color:green">**有bottleneck结构**</span>，即下图右中的1×11×1 卷积层，用于先降维再升维，主要出于**降低计算复杂度的现实考虑**，称之为“**bottleneck block**”，另一种<span style="color:green">**没有bottleneck结构**</span>，如下图左所示，称之为“**basic block**”。basic block由2个3×33×3卷积层构成，bottleneck block由1×1

<img src="https://img-blog.csdn.net/20180114183212429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast">

左边的图：

右边的图：第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632。

用$1*1$卷积减少参数量。

<span style="color:red">**补充结束**</span>

-----

卷积神经网络最初是为图像分类任务设计的。后来出现了FCN，利用VGG16作为主干网络，实现了端到端的语义分割。然而，直接用这些网络会导致粗糙的像素输出（较低的像素精度），因为在分类任务中为了获得更多的上下文进行了下采样操作。为了改进输出效果，作者建议使用跳跃连接与浅层的特征图进行融合。SegNet网络中提出用一个大的解码器段对这些特征进行上采样，该段通过使用编码器的最大池块的索引来执行更好的池化操作。还有人使用过CRF（条件随机场）等方式。然而，依赖于CRF这种细化分割，大大增加了计算负载。

最近的一些工作表明，通过调整ResNets来适应分割任务，获得了较大的精度提升。这些网络的核心元素是一个 residual block，它包含了从输入到输出的身份连接，以缓解在具有大量层的网络中存在的退化问题。基于ResNet的实验表明，residual连接的更薄（更简单的层）但更深（使用更多层）的体系结构可达到最佳效果（state-of-the art），并且可以胜过VGG16 [15]等更复杂的设计，这些设计使用常规的卷积层而非跳跃连接。

利用了下采样过程中所有可用的信息，可以通过long-range residual连接实现高分辨率的预测。Pohlen等人提出了ResNet-like结构，它通过使用两个处理流将多尺度上下文与像素精度结合起来。这两个处理流，一个是以全分辨率处理的，另一个是执行下采样操作的。Ghiasi等人，提出了一种复杂的体系结构，它构造了一个Laplacian金字塔来处理和组合多个尺度的特征。所有这些方法都达到了最高的精度，但所需的资源很多，在实时应用中显得不可行。ENET是速度快，但是精度低，ENET的作者adapt ResNet to the segmentation task，但是同其他方法相比，它的精度低了很多。

## ERFNET网络结构

在这一部分，我们介绍了我们的实时分割网络的结构。我们旨在解决现有残差网络的效率限制，这些残差网络在分类和分割任务中性能表现很好。为了解决这种限制，我们设计了一个语义分割结构，比起现有架构，它能更高效的使用参数，且允许我们的网络获得一个高质量的分割精度，同时保持高效率，以满足实时应用的需求。

**A. Factorized Residual Layers**

残差层具有允许卷积层近似残差函数的特性，因为层向量输入x的输出向量y变为：

$y = F(x,{W_i})+W_sx$

$W_s$通常是标识映射（identity mapping），$F(x,{W_i})$代表要学习的residual mapping。这个residual formulation有利于学习，且显著地减少了，在那些叠加了大量layers的架构中的退化问题（越是深层次的提取，特征细节丢失的越多的意思嘛？）。

（a）图 -->最开始提出的是每个residual层两个实例：the non-bottleneck；两个$3*3$的卷积

（b）图 --> 这两个版本的参数数量相似，精度几乎相等。

<img src="../../pics/CV/ISG/ERFNet/image-20210312202643816.png">

然而，bottleneck需要的计算资源更少，网络越深，这种资源的节约更明显。因此那些优秀的网络架构普遍采用bottleneck设计。然而，根据reported，如果是增加深度，non-bottleneck ResNets比起bottleneck来说，获得的精度提升更高，这表明non-bottleneck和bottleneck不等价。<span style="color:red">**【bottleneck需要的计算资源更少，non-bottleneck需要的计算资源更多，加深网络深度，non-bottleneck的精度会更好； the bottleneck design仍然存在退化问题！！】**</span>

我们建议完全使用一维滤波器的卷积，以更优化的方式重新设计 non-bottleneck residual module，如图（C）

<img src="../../pics/CV/ISG/ERFNet/image-20210312203213560.png">

任何二维滤波器都可以用以下方式，用一维滤波器的组合来表示。

$R^{C*d^h*d^v*F}$表示一个典型的二维卷积层的权重。

C是输入通道数，F（feature maps）是使出通道数$d^h*d^v$代表每个特征图的核大小。

$(3*1*w+1)*w*4$ ；+1是带有偏置。

原有$3*3$的是$(3*3*w+1)*2$

**Factorized Residual Layers**

一维因子分解来加速和减少原始non-bottleneck的参数。（use of the described 1D factorization to accelerate and reduce the parameters of the original non-bottleneck layer），<span style="color:red">缩写为 non-bt-1D</span>，图（c）即non-bt-1D的结构图。该模块速度快，参数比bottleneck少，学习能力与精确度和non-bottleneck一样。

**表一**总结了每个residual block的权重总数，并将他们的原始权重与我们提出的ID factorizations进行了对比。Both non-bottleneck and bottleneck都可以factorized into 1D kernels。但是non-bottleneck更好，因为它减少了33%的卷积，执行速度更快。<span style="color:red">（non-bottleneck的factorized into 1D kernels更快）</span>。

<img src="../../pics/CV/ISG/ERFNet/image-20210312210827422.png">

<div style="text-align: center">表一</div>

- FM是特征图的数目
- ext好像是外部，int好像是内部。

这种设计可以增加特征图的数量，同时减少计算资源。在以分类为目标的residual网络中，增加特征图是非常有效的。增加特征图的数量也可以提高分割质量（大致是这种意思 semantic segmentation can also

benefifit from the increased width）

***Architecture design***

设计一个精度与效率共存的网络结构。使用上问提出的 non-bt-1D层。

网络架构的详细描述如**表二**

<img src="../../pics/CV/ISG/ERFNet/image-20210312212020179.png">

<div style="text-align: center">表二</div>

<img src="../../pics/CV/ISG/ERFNet/image-20210312115151250.png">

- OUT-F：输出的特征图数目
- out-Res：输出图的大小

采用编码解码架构，与SegNet和ENet类似；与FCN这种需要融合来自不同层特征以获得细粒度输出的结构相反，我们的方法遵循一种更顺序的体系结构，它基于产生下采样特征映射的编码器段和随后的解码器段，对特征映射进行采样以匹配输入分辨率。没有跳跃连接，只有编码解码。

1-16是编码部分，由residual blocks和downsampling blocks组成；下采样会降低像素的精度（降低空间分辨率），但是也有两个优点：下采样可以获得深层次的语义信息收集更多的上下文信息<span style="color:red">（ it lets the deeper layers gather more context (to improve classifification)）</span>，并有助于减少计算。因此，为了保持良好的平衡，我们执行三个下采样：在第1层、第2层和第8层。**我们的下采样块，灵感来自于 ENet的初始块**

<img src="https://img2018.cnblogs.com/blog/1229928/201811/1229928-20181123200423168-1352538922.png">

（a）ENet的initial block。

通过连接单个3x3，步长为2的卷积和最大池化模块的并行输出来执行下采样。ENet只将它用于执行早期下采样的初始块，但是我们所有的下采样都使用这种模块。此外，我们还在non-bt-1D层中插入了一些扩展的卷积（dilated convolutions），以收集更多的上下文，这就提高了我们的实验的准确性。这种技术已经被证明（在计算成本和参数方面）比使用更大的核大小更有效。

解码部分是17-23。它主要的任务是对编码器的特征映射进行上采样，以匹配输入分辨率。有一个小型的解码器，其唯一目的是通过微调细节对编码器的输出进行上采样。与SegNet和ENet相反，我们没有使用max-unpooling操作进行上采样，我们使用的是步长为2的简单反卷积。使用反卷积的有点事不需要共享编码器的pooling indexes。因此，deconvolutions简化了内存和计算需求

# 实验

## 通用设置

未使用注释粗糙的图片。

使用注释良好的进行训练，训练过程中没有使用验证集。

Iou作为衡量标准。

$Iou = \frac{TP}{TP+FP+FN}$

- TP == True positives
- FP == False positives
- FN == False positives

随机梯度下降的 Adam 优化器

BatchSize = 12， momentum=0.9，weight decay of $2e^{-4}$，学习率$5e^4$

that the training error becomes stagnant, in order to accelerate convergence

## Comparison of residual layers

the bottleneck (bt)

the non-bottleneck (non-bt) 

our proposed non-bottleneck-1D (non-bt-1D) designs.

三种对比实验。

 (Fig. 2 (b)), which uses 1x1 convolutions to reduce the number of feature maps computed internally in the 3x3 convolution by a factor of 4。

参数量是：weights和biases的总数（ The number of parameters (#Par) is calculated as the total number of weights and biases of each network.）

图片是$2048*1024$,但是输入模型的图片大小是$1024*512$，最终会还原到原图的大小。

 

| Residual Block    | without bias                                  | with bias                                                 |
| ----------------- | --------------------------------------------- | --------------------------------------------------------- |
| bottleneck        | 1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632   | (1x1x256+1)x64 + (3x3x64+1)x64 + (1x1x64+1)x256 = 70016   |
| non-bottleneck    | 3x3x64x64x2 = 73728                           | (3x3x64+1)x64x2 = 73856                                   |
| bottleneck-1D     | 1x1x256x64 + 3x1x64x64x2 + 1x1x64x256 = 57344 | (1x1x256+1)x64 + (3x1x64+1)x64x2 + (1x1x64+1)x256 = 57792 |
| non-bottleneck-1D | 3x1x64x64x4 = 49152                           | (3x1x64+1)x64x4 = 49408                                   |


