# Swin Unet

<a href="https://www.cnblogs.com/shuimuqingyang/p/14776524.html">别人的博客就是清楚</a>

# Abstract

在过去的几年里，卷积神经网络在医学图像分析中取得了里程碑式的进展。特别是基于U型结构和跳跃连接的深度神经网络，已经广泛应用于各种医学图像任务中。然而，CNN虽然取得了优异的性能，但由于==卷积运算的局部性，无法很好地学习全局和长程语义信息交互==。本文提出了==Swin-Unet，一种用于医学图像分割的类Unet纯transformer。==token（符号化）化的图像补丁被馈送到基于转换器的U型编码器-解码器架构中，==该架构具有用于局部全局语义特征学习的跳过连接==。具体来说，我们使用具有移位窗口的分层Swin transformer作为编码器来提取上下文特征。设计了一个对称的基于Swin transformer的带补片扩展层的解码器来执行上采样操作，以恢复特征图的空间分辨率。在输入和输出直接下采样和上采样4倍的情况下，多器官和心脏分割任务的实验表明，纯基于变压器的U形编解码网络优于全卷积或变压器和卷积组合的方法。代码和经过培训的模型将在https://github.com/HuCaoFighting/Swin-Unet.公开发布

----

> 我的总结

- CNN的缺点：卷积运算是局部的，无法很好的学习全局和长距离语义信息。
- transformer的优点：Swin Unet采用的纯transformer的结构，将图片进行切分，每个patch被送入transfomer中。标记化的图像块被输入到基于 Transformer 的 U 形编码器-解码器架构中，并带有用于局部全局语义特征学习的跳过连接。 
  - 局部全局语义特征什么意思呢？用Swin transformer的结构进行解释

> 作者创新

- 基于Swin Transformer块，构建了一个具有跳跃连接的对称编码器-解码器体系结构。在编码器中实现了从局部到全局的自注意;在解码器中，将全局特征上采样到输入分辨率，进行相应的像素级分割预测。
- 开发了patch扩展层，无需卷积或插值操作即可实现上采样和特征维数的增加。
- 实验发现跳跃连接对transformer也是有效的，因此最终构建了一个纯基于transformer的u型编解码结构，具有跳跃连接，命名为swin-unet。

# Introduce

现有的医学图像分割方法主要依赖于U型结构的全卷积神经网络(FCNN)[3，4，5]。典型的U形网络，U形网络[3]，由带有跳跃连接的对称编码器-解码器组成。在编码器中，使用一系列卷积层和连续下采样层来提取具有大感受野的深层特征。然后，解码器将提取的深度特征上采样到输入分辨率进行像素级语义预测，来自编码器的不同尺度的高分辨率特征通过跳跃连接进行融合，以减轻下采样造成的空间信息损失。凭借如此优雅的结构设计，U-Net在各种医学成像应用中取得了巨大的成功。遵循这一技术路线，已经开发了许多算法，例如3D  U-Net [6]、Res-UNet [7]、U-Net++ [8]和UNet3+  [9]，用于各种医学成像模式的图像和体积分割。这些基于FCNN的方法在心脏分割、器官分割和病变分割中的优异性能证明了CNN具有很强的学习辨别特征的能力。

目前，虽然基于CNN的方法在医学图像分割领域取得了优异的性能，但仍然不能完全满足医学应用对分割精度的严格要求。在医学图像分析中，图像分割仍然是一项具有挑战性的任务。由于卷积运算的内在局部性，基于CNN的方法很难学习到明确的全局和远程语义信息交互[2]。一些研究试图通过使用atrous卷积层[10，11]，自我注意机制[12，13]和图像金字塔[14]来解决这个问题。然而，这些方法在建模长期依赖关系时仍然有局限性。最近，受Transformer在自然语言处理(NLP)领域的巨大成功[15]的启发，研究人员试图将Transformer引入视觉领域[16]。在[17]中，提出了视觉转换器(ViT)来执行图像识别任务。以具有位置嵌入的2D图像块为输入，在大数据集上进行预训练，该方法取得了与基于CNN的方法相当的性能。此外，文献[18]中提出了数据有效的图像变换器，这表明变换器可以在中等规模的数据集上训练，并且通过将其与蒸馏方法相结合可以获得更鲁棒的变换器。在[19]中，开发了一个分层的Swin变换器。以Swin  Transformer为视觉中枢，[19]的作者在图像分类、对象检测和语义分割方面取得了最先进的性能。ViT、DeiT和Swin  Transformer在图像识别任务中的成功展示了Transformer在视觉领域应用的潜力。

最近，受Transformer在自然语言处理(NLP)领域的巨大成功[15]的启发，研究人员试图将Transformer引入视觉领域[16]。在[17]中，提出了视觉转换器(ViT)来执行图像识别任务。以具有位置嵌入的2D图像块为输入，在大数据集上进行预训练，该方法取得了与基于CNN的方法相当的性能。此外，文献[18]中提出了数据有效的图像变换器，这表明变换器可以在中等规模的数据集上训练，并且通过将其与蒸馏方法相结合可以获得更鲁棒的变换器。

在[19]中，开发了一个分层的Swin变换器。以Swin  Transformer为视觉中枢，[19]的作者在图像分类、对象检测和语义分割方面取得了最先进的性能。ViT、DeiT和Swin  Transformer在图像识别任务中的成功展示了Transformer在视觉领域应用的潜力。每个补丁都被视为一个令牌，并被输入到基于转换器的编码器中，以学习深层特征表示。提取的上下文特征由具有片扩展层的解码器上采样，并通过跳跃连接与来自编码器的多尺度特征融合，从而恢复特征图的空间分辨率并进一步执行分割预测。在多器官和心脏分割数据集上的大量实验表明，该方法具有良好的分割精度和稳健的泛化能力。具体来说，我们的贡献可以概括为:(1)基于Swin变换块，我们构建了一个具有跳跃连接的对称编解码体系结构。在编码器中，实现了从局部到全局的自关注；在解码器中，全局特征被上采样到输入分辨率，用于相应的像素级分割预测。(2)开发了一个面片扩展层，实现了不使用卷积或插值运算的上采样和特征维数增加。(3)在实验中发现跳跃连接对于变压器也是有效的，因此最终构建了一个基于变压器的带跳跃连接的纯U型编解码架构，命名为Swin-Unet。

> 我的总结

现有的医学图像分割方法主要依赖于U型结构。由带有跳跃连接的对称编码器-解码器组成。==使用一系列卷积层和连续下采样层来提取具有大感受野的深层特征==。然后，==解码器将提取的深度特征上采样到输入分辨率进行像素级语义预测，来自编码器的不同尺度的高分辨率特征通过跳跃连接进行融合，以减轻下采样造成的空间信息损失。==

transformer在CV中大放异彩，解释ViT，引出Swin Transformer。

CNN及其变种在建模长期依赖关系时仍然具有局限性，

# Related Network

**CNN-based methods : **早期的医学图像分割方法主要是基于轮廓的和传统的基于机器学习的算法[20，21]。随着深度CNN的发展，U-Net在[3]中被提出用于医学图像分割。由于U型结构的简单性和优越性能，各种类似Unet的方法不断涌现，如Res-UNet  [7]、Dense-UNet [22]、U-Net++ [8]和UNet3+ [9]。并且也引入到三维医学图像分割领域，如3D-Unet [6]和V-Net  [23]。目前，基于CNN的方法由于其强大的表示能力，在医学图像分割领域取得了巨大的成功。

**Vision transformers :**transformer最初是在[15]中为机器翻译任务提出的。在自然语言处理领域，基于transformer的方法在各种任务中取得了最先进的性能[24]。在transformer的成功推动下，研究人员在[17]中引入了一种开创性的视觉transformer(ViT)，它在图像识别任务中实现了令人印象深刻的速度-精度权衡。与基于CNN的方法相比，ViT的缺点是需要在自己的大数据集上进行预训练。为了减轻训练ViT的困难，Deit  [18]描述了几种训练策略，使ViT能够在ImageNet上进行良好的训练。最近，在ViT  [25，26，19]上完成了几个优秀的作品。值得一提的是，在[19]中提出了一种高效且有效的分层视觉转换器，称为Swin  Transformer，作为视觉通用backbone。==基于移动窗口机制，Swin  Transformer在包括图像分类、目标检测和语义分割在内的各种视觉任务上取得了最先进的性能在这项工作中==，我们试图以Swin  Transformer块为基本单元，构建一个具有跳跃连接的U型编解码架构，用于医学图像分割，从而为Transformer在医学图像领域的发展提供一个基准比较。

**Self-attention/Transformer to complement CNNs :**近年来，研究人员试图将自我注意机制引入CNN，以提高网络的性能[13]。在参考文献[12]中，具有附加注意门的跳跃连接被集成在U形结构中以执行医学图像分割。然而，这仍然是基于CNN的方法。目前，正在努力将CNN和Transformer结合起来，以打破CNN在医学图像分割中的主导地位[2，27，1]。在文献[2]中，作者将Transformer和CNN相结合，构成了一个用于2D医学图像分割的强编码器。类似于[2]，[27]和[28]利用Transformer和CNN的互补性来提高模型的分割能力。==目前，Transformer和CNN的各种组合被应用于多模态脑肿瘤分割[29]和3D医学图像分割[1]\[30]。==不同于上述方法，==我们尝试探索纯Transformer在医学图像分割中的应用潜力。==

# Method

## Architecture overview

> 提出的**Swin-Unet**的整体架构如图1所示。

<img src="..\..\pics\CV\transformer\swin-unet-transformer.png">

Swin-Unet由编码器、瓶颈、解码器和跳转连接组成。Swin Unet的基本单元是Swin transformer block。

对于编码器来说，为了将输入转换成序列嵌入，医学图像被分割成不重叠的块，每块大小为4 × 4。通过这种划分方法，每个面片的特征维数变为4 × 4 × 3 = 48。此外，线性嵌入层用于将特征维度投影到任意维度C（**我感觉就是把pach展平送入Swin Transformer block模块**）。 The transformed patch tokens pass through several Swin Transformer blocks and patch merging layers to generate the hierarchical feature representations.（转换后的patch通过几个Swin Transformer块和patch merging层来生成分层特征表示。）具体来说，patch merging layers层负责降采样和增维，Swin Transformer块负责特征表示学习。受U-Net  [3]的启发，我们设计了一个transformer-based decoder。解码器由Swin  Transformer块和==patch merging==组成。提取的上下文特征通过跳跃连接与来自编码器的多尺度特征融合，以弥补下采样造成的空间信息损失。与patch merging层不同，==patch expanding层是专门设计来执行上采样的==patch expanding层将相邻维度的特征图重新变形为分辨率为2倍的大特征图。最后，使用最后一个patch expanding层执行4×上采样，以将特征映射的分辨率恢复到输入分辨率(W  ×H)，然后对这些上采样的特征进行Linear Projection，以输出像素级分割预测。我们将在下面详细说明每个模块

 ## Swin Transformer block

Swin Transformer block是基于 shifted windows构建的。

<img src="..\..\pics\CV\transformer\swin-transformer-block.png">

图二，连续两个swin transformer block。

Swin Transformer通过shifted windows替换MSA（标准多头注意力机制），Swin Transformer模块由一个基于shifted windows的MSA模块组成，后面是一个两层MLP，中间是GELU非线性。在W-MSA和MLP中间会有一个LN（LayerNorm），在每个模块之后应用一个残差连接。【LN和BN的区别，此处为什么用LN不用BN？】

> MLP：fc --> gelu --> fc -->drop，drop 默认为0

swin transformer块可以被公式化为:

<img src="..\..\pics\CV\transformer\swin-unet-math.png">

其中Q，K，$V∈R^{M2×d}$ 表示查询矩阵、键矩阵和值矩阵。$M^2$和d分别代表一个窗口中patch的数量和查询或键的维数。B中的值取自偏差矩阵$\hat B∈R^{(2M-1)×(2M+1)}$

## Encoder★

在编码器中，分辨率为$\frac{H}{4}*\frac{W}{4}$的C维标记化输入被馈送到两个连续的Swin transformer block中，以执行representation 学习，其中特征维数和分辨率保持不变。同时，patch merging层将减少标记数量(2倍下采样)，并将特征维数增加到原始维数的2倍。该程序将在编码器中重复三次。

Patch merging layer：输入patch被分为4部分，把这四部分通过patch merging layer连接在一起。通过这样的处理，特征分辨率将被下采样2倍。并且由于连接操作导致特征尺寸增加4倍，所以在连接的特征上应用线性层将特征尺寸统一到2倍的原始尺寸。

## Bottleneck

由于Transformer太深而无法收敛[33]，因此仅使用两个连续的Swin  Transformer块来构建学习深层特征表示的Bottleneck。在Bottleneck中，特征尺寸和分辨率保持不变。

## Decoder

与编码器相对应，对称解码器基于Swin transformer block构建。为此，与编码器中使用的patch merging layer不同，我们使用解码器中的patch expanding layer对提取的深层特征进行上采样。patch expanding layer将相邻维度的特征映射重新成形为更高分辨率的特征映射(2倍上采样)，并相应地将特征维度减少到原始维度的一半。

**Patch expanding layer ：**以第一个patch expanding layer为例，在上采样前，在输入特征$(\frac{W}{32}×\frac{H}{32}×8C)$上应用一个线性层，将特征维数增加到$(\frac{W}{32}×\frac{H}{32}×16C)$，即原始维数的2倍。然后，我们使用重排操作将输入特征的分辨率扩展到输入分辨率的2倍，并将特征维数减少到输入维数的四分之一$(\frac{W}{32}×\frac{H}{32}×16C ->\frac{W}{32}×\frac{H}{32}×4C)$[这里应该是跳跃连接的原因，缩小到1/4？]。我们将在第4.5节讨论使用补丁扩展层执行向上采样的影响。

## Skip connection

类似于U-Net，跳跃连接用于将来自编码器的多尺度特征与上采样特征融合。我们将浅层特征和深层特征连接在一起，以减少由下采样引起的空间信息损失。接下来是一个线性图层，串联要素的尺寸保持与上采样要素的尺寸相同。在4.5节中，我们将详细讨论跳过连接的数量对模型性能的影响。

# Experiments

## Datasets

Synapse multi-organ segmentation dataset  多器官分割数据集。

数据集包括30个病例，3779张轴向腹部临床CT图像。请看参考文献2，34

18个样本分为训练集，12个样本分为测试集。以平均Dice-Similarity coefficient（DSC）和平均Hausdorff Distance距离作为评价指标，对8个腹部器官(主动脉、胆囊、脾脏、左肾、右肾、肝脏、胰腺、脾脏、胃)进行评价。

自动心脏诊断挑战数据集(ACDC):ACDC数据集是使用磁共振扫描仪从不同患者收集的。对于每个患者的磁共振图像，标记左心室、右心室和心肌(MYO)。数据集分为70个训练样本、10个验证样本和20个测试样本。类似于[2]，只有平均DSC用于评估我们在该数据集上的方法。

## Implementation details

Swin-Unet是基于Python 3.6和Pytorch  1.7.0实现的。对于所有训练案例，数据扩充(如翻转和旋转)用于增加数据多样性。输入图像大小和补片大小分别设置为224×224和4。我们在32GB内存的Nvidia  V100  GPU上训练我们的模型。在ImageNet上预先训练的权重用于初始化模型参数。在训练期间，批量大小为24，SGD优化器动量为0.9，权重衰减为1e-4，用于优化我们的反向传播模型。

## Experiment results on Synapse dataset

在Synapse多器官CT扫描数据集上，Swin Unet与之前方法的比较如表一所示。

<img src="..\..\pics\CV\transformer\swin-unet-exp.png">

是用的自己实现的U-Net和TransUnet进行的训练。

如图三所示：基于CNN的方法存在过分割的问题，可能是卷积运算的局部性造成的。在这项工作中，我们证明了通过将变压器与具有跳跃连接的U形架构相集成，没有卷积的纯Transformer可以更好地学习全局和长期语义信息交互，从而产生更好地分割效果。

<img src="..\..\pics\CV\transformer\swin-unet-seg.png">



## Experiment results on ACDC dataset

与Synapse数据集相似，所提出的Swin-Unet在ACDC数据集上训练以执行医学图像分割。实验结果总结在表2中，通过使用磁共振模式的图像数据作为输入，Swin Unet仍然能够以90.00%的准确率获得优异的性能，这表明我们的方法具有良好的泛化能力和鲁棒性。

<img src="..\..\pics\CV\transformer\swin-unet-exp2.png">

## Ablation study

主要是对上采样、跳跃连接的数量、输入图片的大小和模型比例进行消融实验。

### 上采样

设计了一个专门的上采样模块，patch expanding layer。为了验证patch expanding layer效果的确是更好，采用了双线性插值、转置卷积、patch expanding layer进行实验。结果如表三所示：的确是patch expanding layer效果更好。

<img src="..\..\pics\CV\transformer\image-20210615212159669.png">

### 跳跃连接

探讨了不同跳跃连接数对分割精度的影响。跳跃连接是被加在了$\frac{1}{4} \space \frac{1}{8} \space \frac{1}{16} $分辨率等级的位置。

<img src="..\..\pics\CV\transformer\image-20210615212906276.png">

### 输入图片尺寸

测试了$224*224 和 384*384$，patch大小不变仍为4，实验发现，Transformer的输入token越大，模型分割精度越高。但是网络的计算量也显著增加。为了效率，选取的224大小的。

<img src="..\..\pics\CV\transformer\image-20210615213924224.png">

### 加深网络

几乎没有提升，所以采用的tiny模型。

<img src="..\..\pics\CV\transformer\image-20210615214129324.png">

## Discussion

transformer-based model受预训练模型额严重影响（**那TransUnet是不是也很受预训练权重的影响？？**）。直接使用了ImageNet上Swin transformer地预训练权重。我们使用的是2D图像，而大多数医疗图像是3D的，我们将探索3D图像的分割。

# Conclusion

大量实验表明，我们的Swin-Unet性能好，泛化能力强。