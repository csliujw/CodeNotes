# 博客解读

<a href="https://zhuanlan.zhihu.com/p/149252931?from_voters_page=true">其他参考博客</a>

## 创新点

将目标检测任务转化为一个序列预测（set prediction）的任务，使用transformer编码-解码器结构和双边匹配的方法，由输入图像直接得到预测结果序列。和SOTA的检测方法不同，没有proposal（Faster R-CNN），没有anchor（YOLO），没有center(CenterNet)，也没有繁琐的NMS，直接预测检测框和类别，利用二分图匹配的匈牙利算法，将CNN和transformer巧妙的结合，实现目标检测的任务。

![](https://pic1.zhimg.com/v2-477a4e2a04b4913e1d8dd4b67e4df0f0_r.jpg)

在本文的检测框架中，有两个至关重要的因素：

①使预测框和ground truth之间一对一匹配的序列预测loss；

②预测一组目标序列，并对它们之间关系进行建模的网络结构。接下来依次介绍这两个因素的设计方法。

## 模型的整体结构

![](https://pic1.zhimg.com/80/v2-aae1329060cd9d50df17c4e7a421e09c_720w.jpg)

Backbone + transformer + Prediction

CNN + encoder+decoder + FFN

> Backbone

利用传统的CNN网络，将输入的图像 ![[公式]](https://www.zhihu.com/equation?tex=3+%5Ctimes+W_%7B0%7D+%5Ctimes+H_%7B0%7D) 变成尺度为 ![[公式]](https://www.zhihu.com/equation?tex=2048+%5Ctimes++%5Cfrac%7BW_%7B0%7D%7D%7B32%7D+%5Ctimes+%5Cfrac%7BH_%7B0%7D%7D%7B32%7D) 的特征图

> transformer

![](https://pic2.zhimg.com/v2-1be61511d53dca07f1c83697eb23a87d_r.jpg)

**Transformer encoder**部分首先将输入的特征图降维并flatten，然后送入下图左半部分所示的结构中，和空间位置编码一起并行经过多个自注意力分支、正则化和FFN，得到一组长度为N的预测目标序列。其中，每个自注意力分支的工作原理为可参考[刘岩：详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)，也可以参照论文：[https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

接着，将Transformer encoder得到的预测目标序列经过上图右半部分所示的Transformer decoder，并行的解码得到输出序列（而不是像机器翻译那样逐个元素输出）。和传统的autogreesive机制不同，每个层可以解码N个目标，由于解码器的位置不变性，即调换输入顺序结果不变，除了每个像素本身的信息，位置信息也很重要，所以这N个输入嵌入必须不同以产生不同的结果，所以学习NLP里面的方法，加入positional encoding并且每层都加，==作者非常用力的在处理position的问题，在使用 transformer 处理图片类的输入的时候，一定要注意position的问题。==

>预测头部（FFN）

使用共享参数的FFNs（由一个具有ReLU激活函数和d维隐藏层的3层感知器和一个线性投影层构成）独立解码为包含类别得分和预测框坐标的最终检测结果（N个），FFN预测框的标准化中心坐标，高度和宽度w.r.t. 输入图像，然后线性层使用softmax函数预测类标签。

## 模型损失函数

基于序列预测的思想，作者将网络的预测结果看作一个长度为N的固定顺序序列 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D%3D%7B%5Ctilde%7By%7D_%7Bi%7D%7D%2C%5C+i%5Cepsilon%281%2CN%29) ,（其中N值固定，且远大于图中ground truth目标的数量） ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Ctilde%7By%7D_%7Bi%7D%7D%3D%28%5Ctilde%7Bc_%7Bi%7D%7D%2C%5Ctilde%7Bb%7D_%7Bi%7D%29) ，同时将ground truth也看作一个序列 ![[公式]](https://www.zhihu.com/equation?tex=y%3Ay_%7Bi%7D%3D%28c_%7Bi%7D%2Cb_%7Bi%7D%29) （长度一定不足N，所以用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) （表示无对象）对该序列进行填充，可理解为背景类别，使其长度等于N），其中 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bi%7D) 表示该目标所属真实类别， ![[公式]](https://www.zhihu.com/equation?tex=b_%7Bi%7D) 表示为一个四元组（含目标框的中心点坐标和宽高，且均为相对图像的比例坐标）。

那么预测任务就可以看作是 ![[公式]](https://www.zhihu.com/equation?tex=y%E4%B8%8E%5Ctilde%7By%7D) 之间的二分图匹配问题，采用匈牙利算法[[1\]](https://zhuanlan.zhihu.com/p/144974069#ref_1)作为二分匹配算法的求解方法，定义最小匹配的策略如下：

![img](https://pic1.zhimg.com/80/v2-8e3856e0d4bf2f3feb44e032bab5f7e0_720w.jpg)

求出最小损失时的匹配策略 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Csigma%7D) ，对于 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmatch%7D) 同时考虑了类别预测损失即真实框之间的相似度预测。

对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28i%29) , ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bi%7D) 的预测类别置信度为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BP%7D_%7B%5Csigma%28i%29%7D%28c_%7Bi%7D%29) ,边界框预测为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bb%7D_%7B%5Csigma%28i%29%7D) ,对于非空的匹配，定于 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmatch%7D) 为：

![img](https://pic4.zhimg.com/80/v2-67ba49e2b04107cf3507090c062374ef_720w.png)

进而得出整体的损失：

![img](https://pic1.zhimg.com/80/v2-fbf5d4eff6d770f23f2bbc46278ec890_720w.png)

考虑到尺度的问题，将L1损失和iou损失线性组合，得出![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D)如下所示：

![img](https://pic3.zhimg.com/80/v2-e3105edd50e92ae3e7cb9adb3ad69926_720w.png)

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D) 采用的是Generalized intersection over union论文提出的GIOU[[2\]](https://zhuanlan.zhihu.com/p/144974069#ref_2),关于GIOU后面会大致介绍。

为了展示DETR的扩展应用能力，作者还简单设计了一个基于DETR的全景分割框架，结构如下：

![preview](https://pic1.zhimg.com/v2-2b9fad8f3430b22f47251fd62394f108_r.jpg)

## 结果

其中DETR对于大目标的检测效果有所提升，但在小目标的检测中表现较差。

# 摘要

提出了一种新的方法，将目标检测视为直接集预测问题（将目标检测作为预测一个集合的问题？）。我们的方法简化（streamlines）了检测的流程，有效地消除了对许多手动设计组件的需求，如非最大抑制过程或锚生成，这些组件需要我们对预测任务有一定的常识性经验（先验知识）。名为`DEtection TRansformer`或者叫`DETR`的新框架组成部分（ingredients）是一种基于集合的全局损失，它通过`二分匹配`和`transformer encoder-decoder`结构来强制进行唯一的预测。`【好像是用到了匈牙利算法】`。给定一个固定的小的学习对象查询集合，`DETR`对对象和全局图像上下文的关系进行推理，以直接并行输出最终的预测集。与许多其他现代探测器不同，新模型在概念上很简单，不需要专门的库。在coco目标检测数据集上，DETR展现了与Faster R-CNN baseline相当的准确性和高效性。此外，DETR可以很容易地推广，以统一的方式产生全景分割。我们证实了，它明显优于竞争基线。

----

Abstract. We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr.

# 引言

对象检测的目的是为每个感兴趣的对象预测出一系列的bounding boxes和category labels。现代的检测器通过间接的方式处理这一系列预测任务，如在大量的候选区域、锚点或窗口中心上定义代理回归和分类问题。它们的性能受到以下处理步骤的影响，通过锚点集的设计和将目标框分配给锚点的启发式方法来瓦解近乎重复的预测。`为了简化这些方法，我们提出了一钟直接设置预测方法从而绕过代理任务（surrogate tasks）。`这种端到端的哲学已经在复杂的结构化预测任务中（如机器翻译或语音识别）取得了重大进展（significant advances），但是在目标检测上尚未取得进展。先前的尝试要么添加了其他形式的先验知识，要么没有证明在具有挑战的基准上与之前强大的基线显比，更具竞争力。本文旨在弥补这一差距。

我们将对象检测视为直接预测集合的问题，从而简化了训练方法。我们采用基于transformers的编码解码结构，这是一种用于序列预测的流行结构。`transformers的注意力机制明确地对序列中元素之间的所有成对交互进行建模，这使得这些架构特别适用于集合预测的特殊约束，如删除重复预测。`

我们的DETR(DEtection transformer)结构可一次预测所有对象，并通过设置损失函数进行端到端训练，该函数执行预测对象与ground-truth对象之间的二分图匹配。DETR通过删除多个手工设计的编码组件，简化了检测流程的先验知识，如空间锚点、非最大抑制。与大多数现有的检测方法不同，DETR不需要任何自定义层，因此可以很容易地在包含标准CNN和transformer类地框架中复制。

与之前大多数关于直接集预测的工作相比，DETR的主要特征是将二部匹配损失（bipartite matching loss）和带有并行解码器的transformers相结合。相反，以前的工作集中在rnn的自回归解码器上。我们的匹配损失函数将唯一的预测赋值给ground truth对象，并且对预测对象的排列时不变的，因此我们可以并行的处理它们（so we can emit them in parallel）。

我们在最流行的目标检测数据集coco上评估了DETR，且与Faster R-CNN相比表现出了具有竞争力的baseline。Faster R-CNN经过多次迭代，自发布以来，其性能取得了极大的提高。我们的实验表明，我们的新模型性能与它们相当。更确切地说，DETR在大型镀锡上表现出的性能更好，这个结果可能是由于transfomer的非本地计算导致的。但是，它在小物体上的性能较低。我们希望未来的工作将改善这方面，以同样的方式开发FPN[22]更快的R-CNN。

DETR的训练设置在许多方面与标准物体检测器不同。新模型需要超长的训练，并受益于transformer中的辅助译码损失。我们将彻底探索那些组件对演示的性能（证明的性能）至关重要。

DETR的设计精神很容易扩展到更复杂的任务。在我们的实验中，我们表明，在预训练的DETR之上训练的简单分割头优于全景分割上的竞争基准，这是一项具有挑战性的像素级识别任务，最近获得了流行。

The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by defining surrogate regression and classification problems on a large set of proposals, anchors, or window centers. Their performances are significantly influenced by postprocessing steps to collapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors. `To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks.` This end-to-end philosophy has led to significant advances in complex structured prediction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on challenging benchmarks. This paper aims to bridge this gap.

We streamline the training pipeline by viewing object detection as a direct set prediction problem. We adopt an encoder-decoder architecture based on transformers, a popular architecture for sequence prediction. `The self-attention mechanisms of transformers, which explicitly model all pairwise interactions between elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions.`

# 相关工作

我们的工作建立在多个领域的先验工作之上：集合预测的二分匹配损失，基于transformer的编码解码结构，并行编码和对象检测方法。

## Set Prediction

没有规范的深度学习模型可以直接预测集合。基本的集合预测任务是在计算机视觉的上下文中对参考进行多标签分类），基线方法“一对多”进行 不适用于诸如元素之间存在基础结构的检测之类的问题。这些任务的第一个困难是避免重复。大多数当前的检测器使用诸如非最大抑制之类的后处理来解决此问题，`但是直接设置预测是无后处理的`。`他们需要对所有预测元素之间的交互进行建模的全局推理方案，以避免冗余。对于恒定大小的集预测，密集的全连接网络就足够了，但代价很高。一种通用方法是使用自回归序列模型，例如循环神经网络（recurrent neural networks）。在所有情况下，损失函数都应通过预测的排列而不变。通常的解决方案是基于匈牙利算法[20]设计损失，以找到地面真相与预测之间的二分匹配。这将强制执行排列不变性，并确保每个目标元素都具有唯一的匹配项。我们遵循二分匹配损失法。但是，与大多数先前的工作相反，我们不使用自回归模型，而是使用具有并行解码的转换器，我们将在下面进行介绍。` 

There is no canonical deep learning model to directly predict sets. The basic set prediction task is multilabel classification for references in the context of computer vision) for which the baseline approach, one-vs-rest, does not apply to problems such as detection where there is an underlying structure between elements. The first difficulty in these tasks is to avoid near-duplicates. Most current detectors use postprocessings such as non-maximal suppression to address this issue, but direct set prediction are postprocessing-free. They need global inference schemes that model interactions between all predicted elements to avoid redundancy. For constant-size set prediction, dense fully connected networks are sufficient but costly. A general approach is to use auto-regressive sequence models such as recurrent neural networks. In all cases, the loss function should be invariant by a permutation of the predictions. The usual solution is to design a loss based on the Hungarian algorithm [20], to find a bipartite matching between ground-truth and prediction. This enforces permutation-invariance, and guarantees that each target element has a unique match. We follow the bipartite matching loss approach. In contrast to most prior work however, we step away from autoregressive models and use transformers with parallel decoding, which we describe below.

## Transformers and Parallel Decoding

transformers是Vaswani等人引入的一种新的基于注意力的机器翻译构件。`注意力机制`是神经网络层，`它从整个输入序列中聚合信息`。transformers引入了self-attention层，类似于非局部神经网络，它扫描序列的每个元素，并通过聚合整个序列的信息来更新它。基于注意力的模型的主要优点之一是它的全局计算能力和完美的记忆（memory），这使得它比神经网络更适合处理长序列。在自然语言处理、语音处理和计算机视觉等领域，transformers正在取代rnn。transformers首先被用于自回归模型，然后是早期的序列-序列模型，一个接一个地产生输出标记。然而，过高的推理成本(与输出长度成比例，而且难以批处理)导致了在音频，机器翻译，单词表示学习以及最近的语音识别领域中并行序列生成的发展。我们还将transformer和parallel decoding结合在一起，以便在计算成本和执行集预测所需的全局计算能力之间进行适当的权衡。

Transformers were introduced by Vaswani et al. as a new attention-based building block for machine translation. Attention mechanisms are neural network layers that aggregate information from the entire input sequence. Transformers introduced self-attention layers, which, similarly to Non-Local Neural Networks, scan through each element of a sequence and update it by aggregating information from the whole sequence. One of the main advantages of attention-based models is their global computations and perfect memory, which makes them more suitable than RNNs on long sequences. Transformers are now replacing RNNs in many problems in natural language processing, speech processing and computer vision. Transformers were first used in auto-regressive models, following early sequenceto-sequence models, generating output tokens one by one. However, the prohibitive inference cost (proportional to output length, and hard to batch) lead to the development of parallel sequence generation, in the domains of audio, machine translation, word representation learning, and more recently speech recognition. We also combine transformers and parallel decoding for their suitable trade-off between computational cost and the ability to perform the global computations required for set prediction.

## Object detection

w.r.t.( with respect to ) 关于 谈到 涉及

大多数现代物体检测方法都是相对于一些初始的猜测进行预测。两阶段的检测框预测涉及到候选区域，单阶段的方法预测涉及到锚点或可能的对象中心的网格。近期的工作表明，这些系统的最终性能在很大程度上取决于设置这些初始猜测的方式。`在我们的模型中，我们能够通过对输入图像（而不是锚点）进行绝对框预测来直接预测检测集，从而消除了手工制作的过程并简化了检测过程。`

Most modern object detection methods make predictions relative to some initial guesses. Two-stage detectors predict boxes w.r.t. proposals, whereas single-stage methods make predictions w.r.t. anchors or a grid of possible object centers. Recent work demonstrate that the final performance of these systems heavily depends on the exact way these initial guesses are set. `In our model we are able to remove this hand-crafted process and streamline the detection process by directly predicting the set of detections with absolute box prediction with respect to the input image rather than an anchor.` 

**Set-based loss：**几个物体检测器使用了二分匹配损失。但是，在这些早期的深度学习模型中，仅使用卷积层或全连接层对不同预测之间的关系进行建模，而手工设计的NMS后处理可以提高其性能。较新的探测器将地面真实情况和预测之间的非唯一分配规则与NMS一起使用。

可学习的NMS方法和关系网络会特别注意地明确建模不同预测之间的关系。使用直接设定损失，它们不需要任何后处理步骤。但是，这些方法采用其他手工制作的上下文特征（例如提案框坐标）来有效地对检测之间的关系进行建模，而我们正在寻找能够减少模型中编码的先验知识的解决方案。

Set-based loss. Several object detectors used the bipartite matching loss. However, in these early deep learning models, the relation between different prediction was modeled with convolutional or full connected layers only and a hand-designed NMS post-processing can improve their performance. More recent detectors use non unique assignment rules between ground truth and predictions together with an NMS. 

Learnable NMS methods and relation networks explicitly model relations between different predictions with attention. Using direct set losses, they do not require any post-processing steps. However, these methods employ additional hand-crafted context features like proposal box coordinates to model relations between detections efficiently, while we look for solutions that reduce the prior knowledge encoded in the model.

**Recurrent detectors：**`与我们的方法最接近的是`用于对象检测和实例分割的端到端集预测。与我们类似，他们将双向匹配损失与基于CNN激活的编码器-解码器体系结构一起使用，以直接生成一组边界框。但是，这些方法仅在小型数据集上进行了评估，而不是根据现代基准进行评估。特别是，它们基于自回归模型（更确切地说是RNN），因此它们没用利用并行解码的transformers。

Recurrent detectors. Closest to our approach are end-to-end set predictions for object detection and instance segmentation. Similarly to us, they use bipartite-matching losses with encoder-decoder architectures based on CNN activations to directly produce a set of bounding boxes. These approaches, however, were only evaluated on small datasets and not against modern baselines. In particular, they are based on autoregressive models (more precisely RNNs), so they do not leverage the recent transformers with parallel decoding.

# DETR模型

直接集预测中有两个要素是必不可少的：

（1）一组预测损失，迫使预测的和真实的boxes进行唯一匹配。

（2）一种架构来预测一组对象并为它们之间的关系建模。我们在图二中详细描述我们的架构。

![image-20210417160102874](..\..\pics\CV\transformer\DETR.png)

## Object detection set prediction loss

对象检测集损失。

DETR通过解码器一次推断出固定大小的N个预测集，其中N被设置为显着大于图像中对象的典型数量。 训练的主要困难之一是根据ground truth对预测的对象（类，位置，大小）进行评分。 我们的损失会在预测的和ground true之间产生最佳的二分匹配，然后优化特定于对象的（边界框）损失。 

DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image. One of the main difficulties of training is to score predicted objects (class, position, size) with respect to the ground truth. Our loss produces an optimal bipartite matching between predicted and ground truth objects, and then optimize object-specific (bounding box) losses.

我们用y表示真实对象的集合，$\hat y = \{\hat y_i\}^N_{i=1}$表示N个预测的集合。假设N大于图像中的对象数目，y是为大小为N的集合，其中并不都是要检测的对象，也包含无需检测出的对象。为了找到这两个集合之间的二分匹配，我们搜索具有最低成本的N个元素σ属于$S_n$的排列

> 找集合的二分匹配，使这个匹配最合适

![image-20210417160648998](..\..\pics\CV\transformer\(1))

其中$L_{match}(y_i,\hat y_{σ(i)})$是ground truth $y_i$ 和索引$σ(i)$的预测之间的成对匹配成本。使用匈牙利算法可以有效地计算出最佳分配 

![image-20210417161413428](..\..\pics\CV\transformer\(1)_word)

匹配成本同时考虑了类别预测及groudn truth和预测值的相似性。每个groudn truth集合的元素i可以看作是$y_i=(c_i,b_i)$,$c_i$是类别标签（可能是空对象），$b_i∈[0,1]^4$是一个向量，`这个向量被定义为groudn truth box的盒中心坐标及相对于图像大小的高度和宽度。`对于预测的$σ(i)$我们可以看作是类别$c_i$ as $\hat p_{σ(i)}(c_i)$和预测框$\hat b_{σ(i)}$. $L_{match}(y_i,\hat y_{σ(i)})$定义如下：

> 二分匹配的计算方式

![image-20210417162426808](..\..\pics\CV\transformer\L_match)

这种寻找匹配的过程与现代探测器中用于匹配区域或锚定的启发式分配规则具有相同的作用。`主要的区别是，我们需要为没有重复的直接集预测找到一对一的匹配。`

第二步是计算损失函数，即上一步中匹配的所有配对的匈牙利损失。 我们将损失定义为类似于常见对象检测器的损失，即用于类预测的`负对数似然`和稍后定义的`框损失`的`线性组合`

> 用二分匹配的最佳结果进行损失计算。

![image-20210417162952354](..\..\pics\CV\transformer\(2).png)

$\hat σ$是上一步计算出的最佳匹配。实际上，当$c_i= \empty$时，`我们将log-probability项的权重降低了10倍，以解决类别不平衡问题。`这类似于Faster R-CNN的训练过程如何通过分段抽样平衡positive/negative proposals。`注意，对象与∅的匹配代价不依赖于预测，在这种情况下代价为常数。`在匹配成本中，我们使用概率$\hat p_{\hatσ(i)}(ci)$代替对数概率(log-probabilities)。 这使得类预测项可与$L_box(.,.)$相称（如下所述），并且我们观察到了更好的经验性能 

**Bounding box loss：**匹配成本和匈牙利损失的第二部分是$L_{box}(.)$,bounding box的scores. `不像多数检测的box预测那样，先做一些初始化的猜测，我们是直接做的box预测。`尽管这种方法简化了实现，但是却带来了损失相对缩放的问题（relative scaling of the loss）。即使大小相对误差相似，最常见的$l1$损失对大盒子、小盒子会有不同的刻度（scales）。 为了减轻这个问题，我们使用了$l1$损失和广义的IOU损失$L_{iou}(.,.)$的线性组合，该比例不变（应该是系数都是1的意思？看看代码怎么写的！）总的来说，我们的box损失是$L_{box}(b_i,\hat b_{σ(i)})$定义为：

![image-20210417170336776](..\..\pics\CV\transformer\L_box.png)

$入_{iou}, 入_{L1}∈R$是超参数。这两个损失通过批处理中对象的数量进行归一化。

![image-20210417160235019](..\..\pics\CV\transformer\panoptic head)

## DETR结构

整个DETR的结构十分简单，如图二所示。它包含三个主要组件。

- CNN backbone提取compact特征表示
- encoder-decoder transformer
- 一个进行最终检测的简单前馈网络（FFN）

![image-20210417160102874](..\..\pics\CV\transformer\DETR.png)

与许多的现代检测不同，DETR可以在任何深度学习框架中实施，该框架可提供通用的CNN主干和仅几百行的变压体系结构。DETR的推理代码可以在PyTorch中用少于50行实现！我们希望我们的方法的简便性将吸引新的研究人员进入检测领域。

> **Backbone**

初始图像大小$x_{img}∈ \mathbb{R}^{C*H*W}$（三通道的彩色图片），CNN主干网络为其删除低分辨率的activation map$f∈\mathbb{R}^{C*H*W}$。我们使用的典型值是$C=2048$ and $H,W = \frac{H_0}{32},\frac{W_0}{32}$

> **transformer encoder**

首先，$1*1$的卷积将high-level activation map f从C个通道减少到更小的维数d，从而生成一个新的特征图$z_0∈\mathbb{R}^{d*H*W}$,编码器需要一个序列作为输入，因此我们将$z_0$的空间尺寸折叠为一维的，从而生成$d*HW$的特征图。每个编码层都有一个标准的体系结构，由一个multi-head self-attention模块和一个前馈网络（FFN）组成。因为transormer是`特征之间没有空间位置关系`，因此我们提供了一个固定位置编码来进行补充，该编码被添加到每个attention层中。我们遵循补充材料中的体系结构的详细定义，它遵循了中所描述的定义。

> **transformer decoder**

解码器遵循标准的transformer结构，使用multi-headed self- and encoder-decoder attention机制。`与原始的转换器的不同之处在于，我们的模型在每个解码器并行解码N个对象`，而Vaswani等人则是，使用自回归模型，该模型一次预测一个元素的输出序列（NLP由于前后的相关性好像只能一次预测一个？）我们把这些补充材料推荐给不熟悉这些概念的读者。由于解码器也是`特征之间没有空间位置关系`，因此N个输入嵌入必须不同才能产生不同的结果。这些输入嵌入是学习的位置编码，我们将其称为对象查询，并且类似于编码器，我们将它们添加到每个关注层的输入中。 N个对象查询由解码器转换为嵌入的输出。 然后，它们通过前馈网络（在下一个小节中描述）独立地解码为框坐标和类标签，从而得出N个最终预测。 通过对这些嵌入的自编码器和解码器注意，模型可以使用它们之间的成对关系全局地将所有对象归为一类，同时能够将整个图像用作上下文。 

> **FFNs**Prediction feed-forward networks 

最终的预测是通过一个带有ReLU激活函数和隐藏维数d的3层感知器和一个线性投影层来计算的。FFN预测归一化的中心坐标、输入图像框的高度和宽度，线性层使用softmax函数预测类标签。由于我们预测一个固定大小的N个包围框，其中N通常比图像中实际感兴趣对象的数量大得多，因此使用额外的特殊类标签∅表示槽内没有检测到对象。这个类在标准的对象检测方法中扮演类似于“background”类的角色。

> **Auxiliary decoding losses**

我们发现在训练期间在解码器中使用辅助损耗[1]很有帮助，特别是有助于模型输出正确数量的每个类的对象。 我们在每个解码器层之后添加预测FFN和匈牙利损失。 所有预测FFN均共享其参数。 我们使用附加的共享层范数来标准化来自不同解码器层的预测FFN的输入。 

