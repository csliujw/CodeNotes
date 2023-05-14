# Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth


# 简要描述

Abstract
- 当前最先进实例分割方法实时性不高，不适用于自动驾驶这些实时应用。虽然精度高但是fps太低，且生成的mask分辨率低。
- 提出了一种新的高精度的速度快的实例分割模型，生成的mask分辨率高，在200w像素的图像上速度超过10帧每秒。


Introduction
- Figure1 聚类算法恢复实例
- 当前方法基于检测+分割。使用边界框检测方法检测对象，然后为每个对象生成二进制掩码。
- 迄今为止，Mask RCNN依旧是最常用，且效果出色的框架。MaskRCNN在精度方面提供了良好的结果，但是生成的低分辨率mask不总是让人满意。生成mask速度也慢，不适用于自动驾驶这种实时应用。

- 实例分割有一个流行的分支==> proposal-free methods

2.标记像素，然后聚类

对图像的每个像素进行分类标记。接下来是使用聚类算法将像素分组到对象实例中。下图显示了一般框架。



该方法受益于语义分割，可以预测高分辨率的对象掩模。与分割检测跟踪技术相比，标签像素跟踪聚类方法在经常使用的基准上精度较低。由于像素标记需要密集的计算，通常需要更多的计算能力。
https://blog.csdn.net/Yong_Qi2015/article/details/107777080

[不错的总结](https://blog.csdn.net/qq_41562445/article/details/109208873)



总结一下,整体的框架如上图所示，其实是一个很简单的语义分割 Encoder-Decoder 结构。输入为一张 3 × H × W 3\times H\times W3×H×W图像，输出包含三个部分的，a )   σ a)\ \sigmaa) σ图（1 × H × W 1\times H\times W1×H×W） b ) b)b) 像素偏移向量（2 × H × W 2\times H\times W2×H×W） c ) c)c) 类别特定的种子图 （C × H × W C\times H\times WC×H×W）
网络的核心是为了学习像素的偏移,即o i o_io 
i
	
 ,这个偏移量是通过最大化IoU得Lovasz-hinge来优化的.但在inference过程中仍无法确定实例中心,因此网络通过每一类学习一个种子图,来确定中心点和σ \sigmaσ值, 为了平滑σ \sigmaσ值和实例大小的关系,网络预测σ \sigmaσ图.最后在inference的时候,结合种子图和sigma图的值得到高斯分布配合像素嵌入得到实例分割结果.
————————————————
版权声明：本文为CSDN博主「ZhenhangHuang」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_41562445/article/details/109208873

# 详细翻译

## Abstract

目前，最先进的实例分割方法并不适用于像自动驾驶这样的实时应用程序，这些实时应用程序需要快速的执行速度和很高的精度。虽然目前 proposal-based（基于提案）的方法具有很高的精度，但是它们生成 mask 的速度慢且生成的 mask 结果分辨率低。相比之下，proposal-free 方法可以快速生成高分辨率的 mask，但是精度不如 proposal-based。In this work 我们为proposal-free 实例分割提出了一种新的聚类损失函数。这个损失函数将属于同一实例像素的 spatial embeddings（嵌入空间）拉到了一起，共同学习特定实例的 clustering bandwidth（聚类的带宽），以便最大化实例结果mask的iou。当和一个快速的结构相结合时，网络可以实时的进行实例分割且保持一个较高的精度。我们在具有挑战性的Cityscapes基准上评估了我们的方法，并在200万像素的图像上以超过10fps/s的速度获得了最佳的效果。（超过Mask RCNN 5%）

## Introduction

语义实例分割的任务是定位图片中的所有目标对象，并为每个对象赋值一个特殊的类别并为每个对象生成a pixel-perfect mask，完美勾勒出他们的形状。这个标准的边框目标检测形成鲜明的对比，边框目标检测中每个对象由粗糙的矩形框表示。在许多应用中，每个对象都需要一个二进制mask，从自动驾驶和机器人应用到照片编辑/分析，实例分割仍是一个重要的研究课题。

目前，实例分割的主要方法是基于检测和分割，其中，使用bounding-box检测方法对物体进行检测，然后为每个对象生成一个二进制mask。尽管过去进行了许多尝试，MaskRCNN框架是第一个在许多benchmarks中取得出色结果的框架，至今仍是使用最多的实例分割方法。尽管Mask RCNN这种方法在精度上取得了很好的结果，但是它生成的底分辨率的masks不总是可取的（如照片编辑）且以较低的帧率运行【实时性不好】，这使得注入自动驾驶一类的实时应用无法实现。

另一种流行的实例分割分支方法是proposal-free方法，这种方法是<span style="color:green">基于embeding损失函数或像素关联性学习。</span>由于这些方法通常依赖于密集预测网络（dense-prediction networks）。他们可以生成高分辨率的mask。除此之外，proposal-free方法通常意味着比proposal-based更快。虽然这些方法很有希望（promising），然而，他们的性能且不如Mask R-CNN。

在本文中，我们为proposal-free实例分割设计了一种新的损失函数，结合了两者的优点：精准，mask分辨率高结合实时性能（combined with real-time performance）。<span style="color:green">我们的方法是基于该种原则：像素可以被关联，通过指向一个物体的中心【像素与对象的关联可通过指向该对象中心来实现】（Our method is based on the principle that pixels can be associated with an object by pointing to that object’s center.）。我们并未像之前的工作一样，对所有像素使用标准的回归损失（regression loss），迫使他们直接指向对象的中心不同，我们引入了新的损失函数，该函数优化了每个对象mask的iou值。因此我们的损失函数将间接的迫使对象的像素指向对象的中心。</span><span style="color:red">对于大对象，网络将学会让这个区域变大，减少远离物体中心像素的损失。（For big objects, the network will
learn to make this region bigger, relaxing the loss on pix-
els which are further away from the object’s center.）</span> <span style="color:red">在推理时，通过在每个对象中心进行聚类学习，特定于对象的区域进行聚类来恢复的。（At inference time, instances are recovered by clustering around each object’s center with the learned, object-specific region）</span>see figure 1

<img src="../../pics/CV\ISG\Embeddings and Clustering Bandwidth\image-20210311160610722.png">

Figure 1. <span style="color:red">我们的损失函数鼓励像素指向对象中心区域的一个最佳位置，对象中心周围的特定区域，最大化每个实例对象mask的iou（Our loss function encourages pixels to point into an optimal, object-specifific region around the object’s center, maximizing the intersection-over-union of each object’s mask.）</span>对于大对象，这个区域会更大，以减少这些边缘像素的损失！左下角显示了用颜色编码学习到的偏移向量。右下角显示了移位的像素和学习道德偏移向量。<span style="color:green">通过在每个中心周围使用学习到的最佳聚类区域进行聚类来恢复实例。（ Instances are recovered by clustering around each center with the learned, optimal clustering region.）</span>

我们在富有挑战性的cityscapes数据集上测试了我们的方法，并且我们的方法取得了最佳效果，以27.6与26.2的得分超过了Mask R-CNN，且平均速度达到了每秒10fps。我们还观察到，我买的方法在车辆和行人上的效果很好。在cityscapes和coco上达到了与Mask R-CNN相似的分数。在cityscapes数据集上我们是第一个实时运行且精度高的方法。

总结：（1）<span style="color:green">提出了一个新的损失函数，该函数通过将像素拉入最佳的，特定于对象的聚类区域来直接优化每个实例的Iou。（ propose a new loss function which directly optimizes the intersection-over-union of each instance by pulling pixels into an optimal, object-specifific clustering region）</span>（2）在cityscapes数据集上取得了最好的实时效果。



## Related Work

<span style="color:green">当前最好的实例分割方法是基于候选的（proposal-based）</span>，它依赖于Faster R-CNN对象检测框架，<span style="color:red">Faster R-CNN在当前目标检测领域处于领导地位，是benchmarks。</span>先前的实例分割方法依赖于先获得对象候选输出，在转化为实例mask。Mask R-CNN和它的衍生网络PANet 通过为Faster R-CNN网络增加了一个用于对象mask预测的分支改善和简化了this pipeline。虽然它们是流行基准（benchmarks）上最好的评分方法，但是他们生成的实例mask像素都是低分辨率的，实际上不适用于实时应用。

实例分割方法的另一个分支依赖于密集预测（dense-prediction），以输入分辨率生成实例mask的分割网络。【输入200x200，生成的mask也是200x200？】这些方法大多基于embedding损失函数，它们强迫属于同一实例像素的特征向量相互相似并且与属于其他对象的像素的特征向量完全不同【属于同一实例的像素的特征向量要相似，不同实例的要完全不同。】近期的工作表明，<span style="color:red">全卷积神经网络的空间不变性</span>对embedding方法并不理想，并建议合并坐标映射或使用所谓的半卷积来缓解这一问题。然而，这些方法仍未能达到与proposal-based一样的性能。

有鉴于此，Kendall等人提出了一个更有前景和简单的方法。他们提出<span style="color:green">通过指向对象的中心来为对象分配像素</span>。这样，他们通过学习位置的相对偏移向量来避免上述的空间不变性问题。我们的方法也是基于相同的概念，通过指向对象中心来为对象分配像素，<span style="color:red">但是我们将后处理的聚类步骤集成到了损失函数中，用这种方式直接优化每个mask的iou值（ Our method is based on the same concept, but integrates the post-processing clustering step directly into the loss function and optimizes the intersection-over-union of each object’s mask directly）</span>Novotny等人与我们的想法类似，但是用的损失函数不同，且他们仍然是检测优先。【我们是proposal-free】【通过指向对象中心向对象分配像素，我们将后处理聚类步骤直接集成到损失函数中。】

通过学习an optimal clustering margin，我们的损失松弛显示出一些相似之处，它们将无谓不确定性整合到损失函数中。 与这些作品相反，我们在测试时直接使用learned margin。 



# Method

把实例分割看成像素赋值问题。希望像素与正确的对象相关联。为此，我们学习了每个像素的offset vector，offset vector指向对象的中心。与我们在3.1中进一步解释的标准回归方法不同，<span style="color:green">我们还为每个对象学习了一个最佳的聚类区域，这样我们就可以放宽远离中心的像素的损失。这将在3.2中说明。</span>为了定位对象的中心，我们为每个语义类学习了一个seed map，如3.5中所述。 该管道如图2所示。 

<img src="../../pics/CV\ISG\Embeddings and Clustering Bandwidth\image-20210314162212430.png">



- 底部分支是预测网络。
- 每个像素的sigma值，直接将其转换为每个对象的聚类裕度。 较大的物体偏蓝，意味着较大的边缘，较小的物体偏黄，意味着较小的边缘 

- 每个像素的偏移矢量，指向吸引力的中心，并使用颜色编码显示，其中颜色表示矢量的角度
- 顶部分支为每个语义类预测一个种子映射。 高值表示该像素的偏移矢量直接指向对象中心，因此请注意，边界值较低，因为它们很难知道要指向哪个中心。 pixel embeddings and margins 是由predicted sigma计算出来的。

## Regression to the instance centroid

实例分割的目的是聚集这些像素$X=\{ {x_0,x_1,x_2,...,x_N} \}$.  x是二维坐标向量。

$S = \{ S_0,S_1,S_2,...,S_k \}$ 代表实例

常用的方法是把像素分配给对应实例的centroid（质心）

$C_k=\frac{1}{N} * \sum_{x∈S_k}X$  

这是通过学习每个像素xi的偏移矢量oi来实现的，因此所得的（空间）嵌入$e_i = x_i + o_i$指向其对应的实例质心。 通常，在直接监督下使用回归损失函数来学习偏移向量：

为了解决这些问题，以前的方法依靠基于密度的聚类算法来首先定位一组质心$C = {C_0，C_1，...，C_K}$，然后基于到目标的最小距离将像素分配给特定实例。 质心公式： 

![image-20210314164300504](..\..\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210314164300504.png)

但是，上述方法在推理时存在两个问题。 

- 首先，必须确定实例质心的位置，
- 其次，对于每个像素需要知道==属于哪个实例==

之前的方法一般是通过基于密度的聚类方法获得实例的中心点，然后按照距离中心最小原则来分配像素。由于此后处理步骤没有集成在损失函数中，因此无法对网络进行端到端的优化，从而导致效果不佳。

作者采用铰链loss迫使同实例像素去贴近中心点的值的某个固定邻域δ,为什么要这样设置呢?因为所有像素值只要贴近中心点都是可以被接受的,比如图二中的(3,5),选择周围的(3,6),(3,4),(2,5),(4,5)都是可以被接受的,不必要限制中心到像素级.因此,铰链损失(hinge loss)如下:

![image-20210314202932140](..\..\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210314202932140.png)

## Learnable margin

大物体会产生大量偏离中心邻域的像素点,从而导致loss增大.最直观的想法就是对于每个实例产生一个特定的δ值.既然一是需要满足δ随着实例越大而越大,而是需要越偏离中心越loss越大. 那么很直观的想法就是生成一个以实例中心为原点的高斯概率图,概率值越高表示像素越靠近中心.


讨论了边界的设定。小对象的边界应该设置的小一些，大对象的边界应该设置的大一些。

- 小物体的边界设置的小，才有利于区分靠的近的两个小物体。
- 大物体的边界设置的大，才有利于提高大物体的分割精度，减少物体边界区域的损失。

我们对每个实例$S_k$使用一个高斯核函数$φk$，它将嵌入（空间）像素ei = xi + oi与实例质心Ck之间的距离转换为属于该实例的概率：

![image-20210314174449780](..\..\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210314174449780.png)

如果$φk（ei）> 0.5$，则将在位置xi处的那个像素分配给实例k。

且可以通过调整sigma参数，控制margin。

![image-20210314174635086](..\..\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210314174635086.png)

大的sigma，margin会更大；小的sigma，margin会更小

我们将σk定义为属于实例k的所有σi的平均值

![image-20210314174745877](..\..\pics\CV\ISG\Embeddings and Clustering Bandwidth\image-20210314174745877.png)

using the Lovasz-hinge loss，它直接优化了每个实例的交集。 因此，我们不需要考虑前景和背景之间的类不平衡。

请注意，对网络的sigma和偏移向量输出没有直接监督（与标准回归损失一样）。 取而代之的是，它们被联合优化以最大化每个实例蒙版的交集，通过反向传播通过Lovasz-hinge损失函数和高斯函数接收梯度。 

## Loss extensions

### Elliptical margin

使用一个值认为Sigma，它只会产生出圆形的，不如分别预测出x，y的Sigma，让他可以产生出椭圆形的,有利于分割细长的物体。

<img src="https://pic2.zhimg.com/v2-550367b11f9531b1729fe25166e46b3d_r.jpg">

### **Learnable Center of Attraction** 可学习的吸引力中心

当前，我们将高斯放置在每个实例的质心Ck中。 这样，像素嵌入被拉向实例质心。 但是，我们也可以让网络学习更理想的吸引力中心。 这可以通过将中心定义为实例k的嵌入上的均值来完成。 这样，网络可以通过更改嵌入的位置来影响吸引力中心的位置： 

使用预测出来的点e（预测出的质心）去训练效果更好

<img src="https://pic4.zhimg.com/80/v2-41dfad0ae011fc36cd76b1e22647d103_720w.jpg">

实验中证明了。

## **Seed map**

在推论的时候，我们需要围绕每个对象的中心进行聚类。我们对良好的像素嵌入进行采样。离实例中心近的像素得分高，远的得分低。我们取得分高的像素嵌入。

Pixel embeddings who lay very close to their instance center will get a high score in the seed map, pixel embeddings which are far away from the instance center will get a low score in the seed map.

实际上，像素嵌入的种子度分数应等于高斯函数的输出，因为它会将嵌入与实例中心之间的距离转换为紧密度分数。嵌入越靠近中心放置，输出越接近1。 

因此，我们用回归损失函数训练种子图。背景像素回归为零，前景像素回归为高斯的输出。我们为每个语义类训练一个种子图，并具有以下损失函数：

<img src="https://pic4.zhimg.com/80/v2-993f5ece74dc8fede174046fb4535823_720w.png">

$s_i$表示像素i的seed输出。计算时将φk（ei）视为标量：仅针对si计算梯度

## **Postprocessing 后处理**

透过上面Seed map的部分，我们知道可以从分数高的Embedding考虑中心点，在Inference的时候，成为各个像素归类到各个物体，而判断的标准就是那个Pixel的机率> 0.5为同一个物体，反之为背景。

# 结论

在这项工作中，我们提出了一个新的聚类损失函数用于实例分割。通过使用高斯函数将像素嵌入转换为前景/背景概率，我们可以直接优化每个对象的遮罩的交过并，并学习最佳的、特定于对象的聚类余量。我们表明，当应用于实时、密集预测网络时，我们在城市景观基准上取得了超过10帧/秒的最高结果，使我们的方法成为第一个提出的高精度实时实例分割方法。

