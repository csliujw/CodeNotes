# Mask R-CNN

目标检测+语义分割 ≈ 实例分割

其他博客

<a href="https://blog.csdn.net/soaring_casia/article/details/110677745">UNet镜像</a>

<a href="https://blog.csdn.net/weixin_43198141/article/details/90178512">Fast RCNN</a>

<a href="https://blog.csdn.net/crazyice521/article/details/65448935?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control">Mask RCNN导读</a>

<a href="https://blog.csdn.net/xiaqunfeng123/article/details/78716136?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control">Mask RCNN翻译</a>

<a href="https://blog.csdn.net/myGFZ/article/details/79136610">Mask RCNN完整翻译</a>

<a href="https://zhuanlan.zhihu.com/p/145842317">Faster RCNN解读</a>

# 开心的背单词时刻

## Abstract中的

夸了下自己的网络有多好，速度有多快，方便应用到其他地方。

**present**

- adj. 现在的；出席的；（事物）存在的；正在考虑的；现在时（态）的
- n. 礼物；现在，目前；（动词）现在式；（举枪的）瞄准
- v. 颁发；赠送；

**conceptually** 

- adv. 概念上

例句：We present a conceptually simple, flexible, and general framework for object instance segmentation.

我们提出了一种概念简单，灵活和通用的实例对象分割框架。

**simutaneously**

- adv. 同时地

例句：Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance.

**overhead**

- 开销
- computational overhead 计算开销

例句：Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps.

**serve as a solid baseline**

- 作为一个坚实的基线

## Introduction中的

**over a short period of time**

- 在很短的时间内

例句：The vision community has rapidly improved object detection and semantic segmentation results over a short period of time.

视觉社区在短时间内迅速改进了目标检测和语义分割结果。

**respectively**

- adv. 分别地；各自地，独自地

----

**intuitive**

- adj. 直觉的；凭直觉获知的

**inference**

- n. 推理；推论；推断

例句：These methods are conceptually intuitive and offer flexibility and robustness, together with fast training and inference time. 

这些方法概念直观，提供灵活性和鲁棒性，以及快速训练和推理时间。

----

**facilitates**

- v. 促进；帮助；使…容易（facilitate的第三人称单数形式）

例句：which facilitates a wide range of flexible architecture designs.

促进了广泛灵活的架构设计。

---

**感觉不错的句子**

In principle Mask R-CNN is an intuitive extension（直接扩展） of Faster R-CNN, yet constructing the mask branch properly is critical for good results（构建mask分支对取得良好的结果至关重要）

----

**de facto**：事实

**coarse**：粗糙

**spatial **：空间的（adj）

**quantization**：量化

the de facto core operation for attending to instances, performs coarse spatial quantization for feature extraction.

----

- decouple ：解耦

Second, we found it essential to decouple mask and class prediction

我们发现解耦掩码和分类至关重要。

----

- couples：夫妻，一对

---

- without bells and whistles：没有花里胡哨的东西。 

---

- ablation：消融
- In ablation experiments：在消融实验中

----

- showcase：展示

----

- detect instance-specific poses 检测特定姿势
  - instance-specific：特定于

## Related Work

- **resorted：**
  - n. 凭借，手段；度假胜地；常去之地
  - vi. 求助，诉诸；常去；采取某手段或方法

----

- **simultaneously**
  - adv. 同时地

## Mask R-CNN

- **finerspatial layout：**精细的布局

# 第一篇博客

<a href="https://blog.csdn.net/chao_shine/article/details/85917280?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control">**来源**</a>

## 问题背景

实例分割需要目标检测+语义分割。

Mask R-CNN在Faster R-CNN的基础上，对每个RoI增加一个掩膜预测分支。这个分支用的是简单的FCN，可以实现像素到像素产生分割掩膜。

为了解决像素不对称问题（实际就是因为截断造成的），提出RoIAlign，效果优于RoIPooling和RoIWarp，可以提高10%-50%的性能。解耦掩膜和类别预测，并对每个感兴趣区域生成$K$个二值掩膜，$K$为类别的个数，每个掩膜dui。这相当于做了多个二分类，机器学习中，用多个二分类处理比一个多分类效果要好。

**啥是掩膜？**

掩模是由0和1组成的一个二进制图像。当在某一功能中应用掩模时，1值区域被处理，被屏蔽的0值区域不被包括在计算中。通过指定的数据值、数据范围、有限或无限值、感兴趣区和注释文件来定义图像掩模，也可以应用上述选项的任意组合作为输入来建立掩模。

**啥是RoIPooling?**

RoIPooling = Region of interest pooling

对感兴趣的区域进行特征提取。具体操作请看下图~

画出感兴趣的区域，我们希望它输出$2*2$的feature  map，所以，把它分成四份（可以不均匀），然后分别进行最大池化（也可以其他池化）。

<img src="../../../pics/CV_blog/instance segment/ROI_Pooling.webp" style="float:left">

**啥是RoIWarp？**

相当于将目标尺寸放缩成指定尺寸的特征图。弃用了？

**啥是RoIAlign?**

使生成的候选框region proposal（候选区域）映射产生固定大小的feature map时提出的。

<img src="../../../pics/CV_blog/instance segment/explain_roialign.png" style="float:left">

针对上图，有着类似的映射

- Conv layers使用的是VGG16，feat_stride=32(即表示，经过网络层后图片缩小为原图的1/32),原图$800*800$,最后一层特征图feature map大小:$25*25$
- 假定原图中有一region proposal，大小为$665*665$，这样，映射到特征图中的大小：665/32=20.78,即$20.78*20.78$，此时，没有像RoiPooling那样就行取整操作，保留浮点数
- 假定pooled_w=7,pooled_h=7,即pooling后固定成$7*7$大小的特征图，所以，将在 feature map上映射的$20.78*20.78$的region proposal 划分成49个同等大小的小区域（$7*7$的特征图需要49个值，所以要pooling49次），每个小区域的大小20.78/7=2.97,即$2.97*2.97$
- 假定采样点数为4，即表示，对于每个$2.97*2.97$的小区域，平分四份，每一份取其中心点位置，而中心点位置的像素，采用双线性插值法进行计算，这样，就会得到四个点的像素值。
- 最后，取四个像素值中最大值作为这个小区域(即：$2.97*2.97$大小的区域)的像素值，如此类推，同样是49个小区域得到49个像素值，组成$7*7$大小的feature map

----

## 研究内容

### Mask R-CNN的特点

- 训练收敛速度快，分割效果优；

- 不外加任何trick，多个技术的融合，例如RoIAlign、Faster R-CNN、FPN；

- 同时完成检测、分割和人体关键点检测任务，并取得start-of-art效果；

- 基础网络强势：ResNeXt-101+FPN；

### 与Faster R-CNN的异同

<img src="../../../pics/CV_blog/instance segment/compare_with_mask rcnn.png" style="float:left">

将有截断的RoIPooling改成RoIAlign

### 区域推荐网络

RPN，region proposal network。<span style="color:red">**这块不懂~**</span>

### FPN特征金字塔网络

Feature Pyramid Networks（特征金字塔网络）

结合不同层的信息，低层位置信息，高层语义信息，提高检测精度。

### Mask branch掩码分支

**目标**：生成掩膜，解耦物体框和掩膜掩膜的关系。

因为COCO提供80类分割的实例，所以最后的输出的通道数为80。

<img src="../../../pics/CV_blog/instance segment/mask_branch.png" style="float:left">

因为完全基于检测的分割，受限于检测的精度。对于未能检测到的小部分，分割效果自然不好。Mask R-CNN让网络自己选择，选择最好尺度的框用于分割，大尺度下的区域可以的分割操作肯定比紧凑的不完整信息要好。直观的影响便是，出现检测重叠部位，出现“块效应”。

## 损失函数

在训练中，Mask R-CNN将每个RoI上的多任务损失函数定义为：

$L=L_{cls}+L_{box}+L_{mask}$

其中，$L_{cls}$和$L_{box}$和在Fast R-CNN中提出的一样，为分类误差和物体框误差。$L_{mask}$仅对分类分支得到的类进行计算损失，即只关注某个类别的分割效果，对其他的类也没法求啊，实际位置都没有，求了也应该是0。

## 疑问点总结

**全连接层的原理以及作用**

全连接层之前的作用是提取特征，全连接层的作用是分类。

**什么是ROI呢**

ROI是Region of Interest的简写，指的是在“特征图上的框”；

​    1）在Fast RCNN中， RoI是指Selective Search完成后得到的“候选框”在特征图上的映射。

​    2）在Faster RCNN中，候选框是经过RPN产生的，然后再把各个“候选框”映射到特征图上，得到RoIs。

**什么是RPN**

替代Selective Search得到感兴趣的区域，并且一个重要的意义是算法的所有步骤都被包含在一个完整的框架中，实现了端到端的训练。

## 常用英语表达

| 英语表达                                           | 中文翻译             |
| -------------------------------------------------- | -------------------- |
| Object instance segmentation                       | 目标实例分割         |
| Segmentation mask                                  | 分割掩模             |
| Object mask                                        | 目标掩模             |
| Bounding-box object detection                      | 边框目标检测         |
| Person keypoint detection                          | 人体关键点检测       |
| Single-model entry                                 | 单一模型实体         |
| Semantic segmentation                              | 语义分割             |
| Fully convolutional network(*FCN*)                 | 全卷积网络           |
| R-CNN                                              | 区域卷积神经网络     |
| Region of Interest(*ROI*)                          | 感兴趣区域           |
| Feature extraction                                 | 特征提取             |
| Quantization-free layer                            | 量化无关层           |
| *Decouple mask*                                    | 解耦掩模             |
| Ablation experiment                                | 对比实验             |
| COCO key-point dataset                             | COCO关键点数据集     |
| One-hot binary mask                                | One-hot 二进制掩模   |
| Instance-level recognition                         | 实例级识别           |
| Region Proposal Network (*RPN*)                    | 区域建议网络         |
| Attention mechanism                                | 注意机制             |
| Segment proposal                                   | 分割建议             |
| Segment candidate                                  | 分割候选区           |
| Bounding box                                       | 边框                 |
| Bounding box proposal                              | 建议边框             |
| Bounding box offset                                | 边框偏移量           |
| Bounding box regression                            | 边框回归             |
| Class label                                        | 类标签               |
| Fully convolutional instance segmentation (*FCIS*) | 全卷积实例分割       |
| pixel-to-pixel alignment                           | 像素到像素对齐       |
| Per-pixel/pixel-level                              | 像素级               |
| Average binary cross-entropy                       | 平均二进制交叉熵损失 |
| Multinomial cross-entropy loss                     | 多项式交叉熵损失     |
| Fully-connected (*fc*) layer                       | 全连接层             |
| Pixel-to-pixel correspondence                      | 像素到像素匹配       |
| Bilinear interpolation                             | 双线性插值           |
| Bilinear resampling                                | 双线性重采样         |
| Network-depth-feature                              | 网络深层特征         |
| Feature Pyramid Network (*FPN*)                    | 特征金字塔网络       |
| Lateral connection                                 | 横向连接             |
| Non-maximum suppression (*NMS*)                    | 非极大值抑制         |
| receptive field (RF)                               | 感受野               |

# 第二篇博客

## 文章思想

把Faster-RCNN进行拓展，添加一个分支进行语义分割。

## 简介

先检测出目标，再对目标进行分割。这样，目标检测效果的好坏会直接影响到分割效果的好坏。

<img src="../../../pics/CV_blog/instance segment/mask_rcnn_framework.png" style="float:left">

## 主要思想

对图片进行检测，找出图像中的ROI，对ROI使用ROIAlign进行像素校正，然后对每一个ROI使用设计的FCN框架进行预测不同的实例所属分类，最终得到图像实例分割结果。

损失函数 = 分类误差+检测误差+分割误差

$L=L_{cls}+L_{box}+L_{mask}$；我的问题是：损失函数的反向传播咋反向的？

分类误差：

检测误差：

分割误差：分类有3类（猫，狗，人），检测得到当前ROI属于“人”这一类，那么所使用的Lmask为“人”这一分支的mask。这样的定义使得我们的网络不需要去区分每一个像素属于哪一类，只需要去区别在这个类当中的不同分别小类。（变成二分类问题吗？）

----

**Faster R-CNN：**包含两个部分，提出RPN区域，找到目标框，对ROI进行分类。核心思想就是把图片区域内容送给深度网络，然后提取出深度网络某层的特征，并用这个特征来判断是什么物体（文章把背景也当成一种类别，故如果是判断是不是20个物体时，实际上在实现是判断21个类。），最后再对是物体的区域进行微微调整。

**Mask Representation**（表示）：mask 对输入目标的空间布局进行编码。使用m*m的矩阵对每一个ROI进行预测而不是使用向量去进行预测，这样可以保证ROI当中的空间信息不会损失。

**ROIAlign：**RoI Pooling就是实现从原图区域映射到卷积区域最后pooling到固定大小的功能，把该区域的尺寸归一化成卷积网络输入的尺寸。在归一化的过程当中，会存在ROI和提取的特征不重合现象出现，作者就提出了这个概念ROIAlign，使用ROIAlign层对提取的特征和输入之间进行校准。 

**改变：**我们避免对每一个ROI边界或者块进行数字化。使用双线性内插法计算在ROI 块当中固定的四个采样位置得到的输入特征值并对结果进行融合。

**Network Architecture：** 分成三个部分进行介绍，第一个是<span style="color:red">主干网络用来进行特征提取</span>，第二个是<span style="color:red">头结构用来做边界框识别（分类和回归）</span>，第三个就是<span style="color:red">mask预测用来对每一个ROI进行区分</span>。

# 第三篇博客

## 摘要

Mask R-CNN：简单，灵活和通用的目标分割框架。

通过添加一个与现有目标检测框回归并行的（原文的确是分割与检测并行），用于预测目标掩码的分支来扩展Faster R-CNN。Mask R-CNN训练简单，相对于Faster R-CNN，只需增加一个较小的开销，运行速度可达5 FPS。

# 原文

## Abstract

## Introduction

提出了Mask R-CNN，继承自Faster R-CNN，增加了一个预测分割的并行分支（对每个RoI进行分割）。

mask branch用的是FCN，我觉得我们测评的改成UNet比较好。

他说，Faster R-CNN改成Mask R-CNN很简单，我们不如再Faster R-CNN的基础上进行修改。（需要什么样的标签数据？？）

Mask R-CNN中提出了一个简单的，量化无关的层，称为RoIAlign，可以保留精确的空间位置。

Mask R-CNN是为每个类独立地预测二进制掩码，这样不会跨类别竞争。并且依赖于网络的RoI分类分支来预测类别。相比之下，FCN通常执行每像素多类分类，分割和分类同时进行，基于我们的实验，对于目标分割效果不佳。【<span style="color:red">为每个类独立的预测二进制掩码，不会引起跨类别竞争。机器学习P63页，多分类和二分类的适用场景和对比</span>】

## Related Work

介绍像素到像素对齐，这是Fast/Faster R-CNN的主要缺失。

别人要么是先分割，再识别。

要么是从候选框中预测候选分割，然后进行分类（that predicts segment proposals from bounding-box proposals， followed by classification）

我们的方法并行进行掩码和类标签的预测，更简单也更灵活。

## Mask R-CNN

Faster R-CNN的每个候选目标都有两个输出，一个类别标签和一个边框偏移量。为此，我们添加了一个输出目标掩模的第三条分支。（掩模的分支是并行的）

介绍Mask R-CNN的关键特点，包括像素到像素的对齐。

# 我的总结