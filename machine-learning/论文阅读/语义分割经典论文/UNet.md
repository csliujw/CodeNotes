# UNet

## 博客

**受FCN的启发，U型结构。分为编码解码。**

**简化的示意图**

<img src="../../pics/CV/UNet/UNet_simplify.png" style="float:left">

**完整的示意图**

<img src="../../pics/CV/UNet/UNet_structures.png">

**详解：**

> <span style="color:green">**注意**</span>

- 蓝色箭头代表3x3的卷积操作，并且stride是1，padding策略是vaild，因此，每个该操作以后，featuremap的大小会减2。

- 红色箭头代表2x2的maxpooling操作，需要注意的是，此时的padding策略也是vaild（same 策略会在边缘填充0，保证featuremap的每个值都会被取到，vaild会忽略掉不能进行下去的pooling操作，而不是进行填充），这就会导致如果pooling之前featuremap的大小是奇数，那么就会损失一些信息 。

- 绿色箭头代表2x2的反卷积操作，这个只要理解了反卷积操作，就没什么问题，操作会将featuremap的大小乘2。

- 灰色箭头表示复制和剪切操作，可以发现，在同一层左边的最后一层要比右边的第一层要大一些，这就导致了，想要利用浅层的feature，就要进行一些剪切，也导致了最终的输出是输入的中心某个区域。

- 输出的最后一层，使用了1x1的卷积层做了分类。

----

> <span style="color:green">**具体说明**</span>

- 输入 $572*572$，经过$3*3$的卷积和ReLU激活后，变为$570*570$了。$570*570$，经过$3*3$的卷积和ReLU激活后，变为$568*568$了。
- $568*568$经过最大池化后，图片大小缩小一半，变为$284*284$ 后面依次类推。不再赘述。
- 32-->30-->28,28经过$2*2$的上采样，然后和（64裁剪成56）的进行特征融合（重叠通道），后面依次类推。
- 输出的最后一层用$1*1$的卷积做了分类的操作（像素级别的分类）

----

> <span style="color:green">**其他**</span>

U-net网络比较简单，前半部分也就是图中左边部分的作用是**特征提取**，后半部分也就是图中的右边部分是上采样。

在一些文献中也把这样的结构叫做编码器-解码器结构。由于此网络整体结构类似于大写的英文字母U，故得名U-net。

U-net与其他常见的分割网络有一点非常不同的地方：**U-net采用了完全不同的特征融合方式：拼接，U-net采用将特征在channel维度拼接在一起，形成更厚的特征。而FCN融合时使用的对应点相加，并不形成更厚的特征。**

总结一下：U-net建立在FCN的网络架构上，作者修改并扩大了这个网络框架，使其能够使用很少的训练图像就得到很精确的分割结果。添加上采样阶段，并且添加了很多的特征通道，允许更多的原图像纹理的信息在高分辨率的layers中进行传播。U-net没有FC层，且全程使用valid来进行卷积，这样的话可以保证分割的结果都是基于没有缺失的上下文特征得到的，因此输入输出的图像尺寸不太一样

## 学习目标

- 为什么要提出UNet？

- UNet的基本结构与FCN的区别，有何改进，提升如何？
- 适用场景，特点，缺陷。
- UNet准确读的衡量指标。

论文的细节可能受数据集的影响/作者的个人爱好 ect.

## 论文泛读

读Abstract、小标题、图标

Abstract感觉理解有误。

**效果自测**

- 论文要解决什么问题：少量数据的有效分割，无缝分割任意大图像？
- 采用了什么方法：
  - 对称的网络结构
  - 三次卷积一次池化，且卷积核大小统一。
  - 特征融合对称？
- 达到了什么效果：2015年ISBI细胞跟踪挑战赛的冠军。

原文：that relies on the strong use of data augmentation（数据增强） to use the available annotated samples more efficiently.

原文：We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. 【在少量数据上的效果也很好】

原文：Moreover, the network is fast. 【速度很快】



## 论文精读

知识不足阶段，建议整篇都好好读一下。后期知识充足了，选出文章中感兴趣的部分。仔细阅读即可。

**目标及效果自测**：

- 所读段落是否详细掌握。

###  Introduction

#### 分析不足

- 介绍了卷积的发展史，视觉的任务和现有不足，引出分割中的不足
- 他人策略的不足，Obviously, the strategy in Ciresan et al. [1] has two drawbacks. 
  - 速度慢，patch直接重叠，重复计算【以前的思路，根据像素点周围的像素来对该像素进行分类-->滑动窗的神经网络，这种网络以像素为单位对图片进行分类（计算量太大，冗余性太高，每个patch之间都有相关的像素），以图片为整体来进行图片分类分割相比起来计算量就小一些】
  - 位置的准确率和视野无法兼得
    - Larger patches范围大，准确率低
    - small patches范围又太小不合适
    - 两者理应兼得

#### 本文网络特点

- elegant   architecture，优雅的结构
- 修改了网络结构，只需很少的训练图像就能工作，并产生更精准的分割。
- 应用取得成功的经典网络，pooling operators被替换为upsampling operators.【与FCN类似】
- 原文：In order to localize, high resolution features from the contracting path are combined with the upsampled output. 为了解决定位问题，将收缩路径中的高分辨率特征与上采样的输出结合在一起，用这些信息进行组装，得到精确的输出。【U型结构中 统一水平线的进行融合】
- 原文：One important modification in our architecture is that in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers. 低层的特征信息会被一级一级传递上去进行融合，上下文的信息可以传播到更高的分辨率层。
- 无全连接层
- 推断缺失上下文【the missing context is extrapolated by mirroring the input image（怎么推断？重叠拼贴策略（请参见图2）。 预测边界区域中的像素）】
- 切片策略进行分割，避开GPU的限制（医疗图像的分辨率好像特别大）【<span style="color:green">网络的小技巧？</span>】

----

#### 挑战

- 任务中可用的数据集很小，所以没办法，过渡使用了【data augmentation by applying elastic deformations】这在细胞中很重要，因为细胞的变形是很常见的。
- 许多细胞分割任务中的另一个挑战是分离同一类别的接触对象
  - To this end, we propose the use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function。【设置权重，以便分离】为此，我们建议使用加权损失，其中触摸单元之间的分离背景标签在损失函数中获得较大的权重。

#### 结果

FCN【DL 语义分割的鼻祖？】

2015年ISBI细胞追踪挑战赛冠军。【细胞分割的鼻祖？】

### Network Architecture

与博客中的描述一致，网络结构比较简单。

都是用$3*3$的卷积核，padding=0，$2*2$的max-pooling，无全连接。

存在的问题：输入图像和输出图像大小并不一致。 由于采用nopadding的卷积层，每次卷积图像都会小一圈，所以downsample和upsample所还原的像素并不一致。(一般会采用加上padding方式来使图像size不变)

<img src="../../pics/CV/UNet/UNet_structures.png">

<span style="color:green">**注意：对输入图片的尺寸有要求！**</span>

### Training

介绍如何训练网络的【包含一些训练技巧】

用输入图像和他对应的语义分割图（maps不知道咋翻译比较好）进行模型训练

> <span style="color:green">**英语学习**</span>

**corresponding segmentation map ==对应的语义分割图**

**the output image is smaller than the input by a constant border width == 输出图像比输入图像小一个恒定的边框宽度。**

**To minimize the overhead == 最小化开销**

**denotes == 表示**

**deviation == 偏差**

**compensate == 补偿**

> <span style="color:green">**训练技巧**</span>

- 使用的SGD 随机梯度下降
- 一次训练一张图
- SGD的momentum设置的很高 达到了 0.99
- 为了平衡不同像素类别之间的差距，为不同类别的像素引入了不同的权值

> <span style="color:green">**解释公式**</span>

energy function 能量函数 ， 在某种意义上和损失函数类似。

能量函数是通过交叉熵损失函数和最终特征图上的像素soft-max 来计算的。（我翻译的好垃圾）

解释了下交叉熵公式中各个值的意思。

**soft-max函数的定义如下：**

$P_{k}(x) = \frac{\exp(a_{k}(X)) }{(\sum_{k^`}^{K}\exp(a_{k^`}(X))}$

$a_{k}(x)$表示 第k个feature channel在像素x处激活。k代表分类数 $P_k(X)$是approximated maximum-function

交叉熵在每个位置上计算其偏差，使用以下的公式进行计算【不懂】

**$E=\sum_{x\in\Omega}\omega(x)log(P_{l(x)}(x))$**

$l$是每个像素的真实label，w是权值，我们通过权值让一些label更加重要【忽略背景，抓住重点分割的对象】

ground truth 正确打标签的训练数据。

预先计算正确打标签的训练数据的权重，用这个补偿训练中不同类别像素之间频率不同的情况，【<span style="color:green">并迫使网络学习我们在接触细胞之间引入的小的分离边界，</span>**<span style="color:red">这个不是很理解</span>**】

> <span style="color:green">**细胞边界的处理**</span>

分离细胞边界采用的是 **morphological operations**【形态学操作】

权重图（weight map）的计算公式如下：

**$\omega(x) = \omega_{c}(x) + \omega_{0}*\exp(- \frac{(d_{1}(x)+d_{2}(x))^2}{2\sigma^2})$**

$d_1$表示到第一个最近细胞边界的距离

$d_2$表示到第二个最近细胞边界的距离

在实验中，我们设置$\omega_0 = 10$，$\sigma≈5$ 【<span style="color:red">**训练技巧**</span>】

为了防止网络过度激活或部分网络无法激活，我们需要初始化一个合适的权值。在UNet网络架构中，我们可以从标准差为$\sqrt{2/N}$的高斯分布中提取初始化权重【<span style="color:red">**实验得出的？？？**</span>】

> <span style="color:green">**3.1 数据增强**</span>

说明数据增强的必要性，数据少的话，最好进行数据增强，扩大数据的规模。

原文：Especially random elastic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images.【<span style="color:red">**不懂**</span>】

**数据增强的方式：**

- 仿射变换
- 颜色抖动
- 水平/垂直翻转
- 随机crop(裁剪)

**仿射变换：**可用OpenCV

- 旋转

- 平移

- 错切（shear）

- 尺度变化

- <span style="color:green">特点：经过仿射变化仍是直线</span>

  <img src="../../pics/CV/UNet/affine_transformation.jpg" style="float:left">

​	特别是训练样本的随机弹性变形似乎是训练一个带有少量图像注释的语义分割网络是一个关键概念。

​	在粗糙的3×3网格上使用随机位移矢量生成平滑变形。位移从具有10个像素标准偏差的高斯分布中采样。然后使用双三次插值计算每像素位移。收缩路径末端的退出层执行进一步的隐式数据增加。【<span style="color:red">**这部分真的没看太明白**</span>】

###  Experiments

效果明显比滑动卷积好。

实验结果对比，UNet效果特别好，没了。

### Conclusion

> ****

## 总结

总览全文，归纳总结。总结文中创新点，关键点，启发点等重要信息。