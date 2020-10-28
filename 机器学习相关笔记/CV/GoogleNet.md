# GoogleNet

优点

- 重复性大，可结构化设计
- 

缺点

- 知识太少，不好描述

具备模块化的性质。

inception具备模块化的性质。

<img src="..\..\pics\pytorch\GoogleNet.jpg" style="float:left">

----

# Naive Inception结构

## 图示

<img src="..\..\pics\pytorch\naive_inception.png" style="float:left">

##  说明

**四部分**

- 1×1的卷积层、
- 3×3的卷积层、
- 5×5的卷积层
- 3×3的最大池化层

最后再将各层的计算结果汇聚至合并层，完成合并后输出。合并的方式是：将各部分输出的特征图相加。

假设图中Naive Inception单元的前一层**输入的数据是一个32×32×256的特征图**，该特征图先被复制成4份并分别被传至接下来的4个部分。我们**假设这4个部分对应的滑动窗口的步长均为1**，其中，

- 1×1卷积层的Padding为0，滑动窗口维度为1×1×256，要求输出的特征图深度为128;
- 3×3卷积层的Padding为1，滑动窗口维度为3×3×256，要求输出的特征图深度为192;
- 5×5卷积层的Padding为2，滑动窗口维度为5×5×256，要求输出的特征图深度为96;
- 3×3最大池化层的Padding为1，滑动窗口维度为3×3×256；

这里对每个卷积层要求输出的特征图深度没有特殊意义，之后通过计算，分别得到这4部分输出的特征图为32×32×128、32×32×192、32×32×96和32×32×256，最后在合并层进行合并，得到32×32×672的特征图，合并的方法是将各个部分输出的特征图相加，最后这个Naive Inception单元输出的特征图维度就是32×32×672。【==只增加了通道数，没有改变图像的大小==】

**存在的问题**

- 所有卷积层直接和前一层输入的数据对接，所以卷积层中的计算量会很大；
- 其次，在这个单元中使用的最大池化层保留了输入数据的特征图的深度，所以在最后进行合并时，总的输出的特征图的深度只会增加，这样增加了该单元之后的网络结构的计算量。

GoogleNet模型对Naive Inception进行了改进！

# GoogleNet inception

<img src="..\..\pics\pytorch\Google_inception.png" style="float:left">

##  1 x 1 卷积的作用

<a href="https://blog.csdn.net/Guo_Yaohua/article/details/106294659">参考博客</a>

==一句话概况：整合通道信息，对通道进行降维或者升维==

**卷积的输出输入都只是一个平面，那么1x1卷积核并没有什么意义，它是完全不考虑像素与周边其他像素关系。 但卷积的输出输入是长方体，所以1x1卷积实际上是对每个像素点，在不同的channels上进行线性组合（信息整合），且保留了图片的原有平面结构，调控depth，从而完成升维或降维的功能。**

**如下图所示，如果选择2个filters的1x1卷积层，那么数据就从原本的depth 3 降到了2。若用4个filters，则起到了升维的作用。**

<img src="..\..\pics\pytorch\1x1_kernel.png" style="float:left">

## GoogleNet  start

增加1x1的卷积层，减少特征图的深度。

<img src="..\..\pics\pytorch\GoogleNet_start.jpg"  style="float:left">

INPUT是整个GoogleNet模型最开始的数据输入层；

Conv层对应在模型中使用的卷积层；

MaxPooL层对应在模型中使用的最大池化层；

Local Response Normalization是在模型中使用的**局部响应归一化层**。

每个层后面的数字表示滑动窗口的高度和宽度及步长，比如第1个卷积层中的数字是7×7+1(S),7×7就是滑动窗口的高度和宽度，1就是滑动窗口的步长。大写的S是Stride的缩写，这个起始部分的输出结果作为Inception单元堆叠部分的输入。

## GoogleNet end

<img src="..\..\pics\pytorch\GoogleNet_end.jpg" style="float:left">

- 最后分类输出部分的输入数据来自Inception单元堆叠部分最后一个Inception单元的合并输出

- AveragePool层对应模型中的平均池化层（Average pooling）

- FC层对应模型中的全连接层

- Softmax对应模型最后进行分类使用的Softmax激活函数【==Softmax避免梯度消失？？==】

























