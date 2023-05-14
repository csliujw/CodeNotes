# 阅读顺序

- 标题
- 摘要
- 结论
- 然后就知道这篇论文讲的什么了。
- 跳到一些实验，看看关键的图表和方法的图表，看看它在干什么。大概就知道这篇文章讲什么，质量怎么样了。是不是适合自己，要不要再读。

---

- 从头读到尾
- 不要太关注细节，公式、证明这些很细节的先 pass，那些重要的图表知道它每一个字在做什么。
    - 方法的流程图
    - 算法的图
    - 实验图的含义，和别人方法的对比，差距。
- 对论文有个大概的了解（圈出一些重要文献，再看看要不要继续精度论文）
- 太难读不懂，就看它引用的文章，再回过来读。

---

- 自己带入，如何就解决问题，如何实现。换我我会怎么做。
- 知道每一句话作者再做什么，脑补作者在做什么。

# Quick Start

先搞清楚这篇论文需要解决的问题是什么。

## 标题

标题为：对<span style="color:green">少样本</span>的无监督领域自适应的点向跨域自监督学习。

- 方法：跨域自监督
- 问题：少样本无监督领域自适应（医疗影像样本少，且需要领域自适应）

## 摘要

无监督域自适应目的是将有标签的源域转移到无标签的目标域。但是部分场景，源域的标签也很难得到。

先前有方法通过 “实例级别的跨域自监督学习+微调” 运用在少样本的无监督领域自适应上，但是这种方法仅学习和对齐低级别的特征。

<span style="color:red">本论文提出：FUDA，端到端。跨域低级特征对齐+在跨域共享的嵌入空间中对语义结构（高层语义信息）进行编码和对齐。</span>

方法的核心思想是：（捕获的是高层的语义信息）

- 通过在域原型上的对比学习，捕获数据的类别语义结构；
- 通过跨域原型自监督来进行特征对齐；

## 结论

本文研究了少样本无监督领域自适应（源域样本少，目标域无标签），提出了一个新的原型跨域自监督学习（（PCS）框架，<span style="color:red">该框架同时执行域内和跨域原型自监督学习以及自适应原型分类器学习。</span>

我们在多个基准数据集上做了大量实验，PCS 优于先前的 SOTA。

# 泛读

## Introduction

传统深度学习训练的缺陷，一旦两个数据集域偏移差距大，A数据上训练的模型就很难运用到B数据上。无监督领域自适应是将在有标签的source上训练，然后到无标签。许多无监督方法在含有丰富label的source上训练，可以在目标域上得到很高的精度，但是事实上，想要获得大量的带有标记的 source 也是很困难的，不切实际。 

本文考虑了一种少样本的无监督领域自适应，只有极少数的有label的source。

> 先前的无监督领域自适应的方法

- 通过最小化数据分布距离和最小化source中的监督损失来对齐source和target的特征（映射到统一空间）然而我们带label的source很少，很难识别出它们的特征，更别说target的特征了。

最近的一些论文利用自监督学习，在当个域上表现出了很好的特征学习效果；并进一步的扩展跨两个域的自监督学习，以获得更好的域自适应性能（将SSL用到了域自适应上【文献39所有的方法】）

文献【39】的方法性能很好，但是缺陷也很明显。

- 首先，数据的语义结构不是由学习到的结构编码的。因为只要两个实例来自于不同的样本，就将它们视为负对，不会考虑它们之间的语义是否相似。（有什么缺点不理解）
- 异常样本对结果的影响很大【难收敛】

![image-20211008181235987](.\Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211008181235987-16336879589411.png)

- 【39】中用的是 two-stage pipeline，很复杂，且在不同数据集上，最佳的 DA 方法不同（通用性不强）

本文提出的：原型跨域自监督学习，是一个新颖的 single-stage 的用于小样本领域自适应学习的框架。我们提出的方法 PCS 主要由三部分组成。

- 首先，PCS 在域原型中执行自我监督，将数据的语义结构隐式编码到嵌入空间中（灵感来源于文献41，并对41的方法进行了改进，进一步利用任务的已知语义信息并在每个域中学习更好的语义结构【学习两中特征嘛？一种特有的，一种共有的】）
- PCS 执行跨域实例到原型匹配，以更稳健的方式将知识从源传输到目标。代替实例到实例匹配，将样本与原型匹配。（对文献39的改进，文献39是采用的实例到实例的匹配，会忽略语义信息。改进后收敛更快，鲁棒性更强）
- 将原型学习和余弦分类器相结合，并根据源和目标原型自适应地更新余弦分类器。为了进一步减轻跨域不匹配的影响，我们执行熵最大化以获得更多样化的输出。【猜想。本来信息就少，为了减少非必要特征的误判，用熵最大化，进行最随机的判断，】<span style="color:green">（最大熵原理的实质就是，在已知部分知识的前提下，关于未知分布最合理的推断就是符合已知知识最不确定或最随机的推断，这是我们可以作出的不偏不倚的选择，任何其它的选择都意味着我们增加了其它的约束和假设，这些约束和假设根据我们掌握的信息无法作出。）</span>

## Related Work

### Domain Adaptation

介绍域自适应的常见方式。

- 特征分布对齐
    - 基于差异的方法显式计算source和target的最大平均差异以对齐两个域。
    - 1）联合最大平均差异对齐分布
    - 2）对齐source和target的二阶统计特征
    - 在特征空间使用对抗学习进行域对齐
    - 通过执行像素对齐来改进域适应。

这些方法比较适用于含label的source较多的情况，但是我们考虑的是样本较少的情况。

### Self-supevised Learning

自监督学习：无监督学习的子集。自监督中的监督是指从数据中自动生成任务进行学习，如：故意丢弃一部分数据，然后预测丢失的数据。

目前，对比学习在学习特征方面效果很好，但大多数对比学习都是实例化（instance-wise）的，旨在学习一个嵌入空间，让来自同一个实例的样本拉的更近。最近基于原型的对比学习在特征学习上的效果很不错【文献 41，2，7，19】。<span style="color:red">本文采用了文献中所说的，基于原型的对比学习来提取特征。</span>

### Self-supervised Learning for Domain Adaptation

基于自监督的领域自适应方法将 SSL 损失加入到了原始的任务网络。Reconstruction was first utilized as self-supervised task in some early works [20, 21], in which source and target share the same encoder to extract domain-invariant features.【在一些早期工作中，重建首先被用作自监督任务 [20, 21]，其中source和target共享相同的编码器以提取域不变特征。】<span style="color:red">为了捕捉到特定于域的特征和共享的特征，将图像的特征提取到了两个空间，一个是域独有的特征，一个是 source 和 target 共有的特征</span>（SSL 在域自适应中用到的方式）

## Approach

域内原型对比学习（提取特征）+跨域原型自监督（特征对齐）

我们限制了带有 label 的 source 的数量。source 一共分为两类。

- $D_s=\{ (X_i^s,y_i^s)_{i=1}^{N_s} \}$ 有标签的 source
- $D_{su}=\{ (X_i^{su})_{i=1}^{N_{su}} \}$ 无标签的 source

target

- $D_{tu}=\{ (X_i^{tu})_{i=1}^{N_{tu}} \}$ 

目的是 在 $D_s、D_{su}、D_{tu}$ 上训练模型，然后再 $D_{tu}$ 上进行评估验证。

基本的模型结构：编码器F + l2 归一化层 ，得到的输出是一个归一化的特征向量 f ∈ $R^d$ 和一个基于余弦相似度的分类器。

### 域内原型对比学习

学习了一个共享特征编码器 F ，用来提取两个域中的判别特征（discriminative features）之前的方法使用实例判别器学习判别特征。这种实例级的对比学习方法产生一种嵌入空间，在此空间中所有实例都能被很好的分离 → **然而实例判别方法有一个缺陷：数据的语义结构不是由学习到的特征编码的**（即语义结构与数据的分类无关），这是因为两个数据被设为负对时，只要两个数据来自不同样本，就可以被设为负对，而不考虑其语义结构（即，本方法没有用到语义结构） → **ProtoNCE通过迭代聚类和表示学习来学习单个域中数据的语义结构，使得同一簇中的特征变得更加聚合，而不同簇中的特征变得更远** → 而本方法在域适配应用下出现问题：来自不同域的不同类的图像可能会被错误地聚合到同一个簇中，而来自不同域的同一类的图像可能会映射到相距很远的簇中 → 本文分别在Ds∪Dsu和Dtu中执行原型对比学习，防止跨域图像的错误聚类和不加区分的特征学习。

# 论文的想法来源

- SSL，用在少样本领域自适应上
- 对比学习在学习特征方面的效果很好，但是多数对比学习用的是 instance-to-instance的缺点很明显；基于原型对比学习的特征提取效果更优。
- 提出通过联合学习多个自我监督任务来执行适应 【文献64】：source 和 target 共享特征编码器，将提取到的特征输入到不同的自监督任务头中。
- 文献39 提出了一种 based on instance discrimination，提出了一种仅有少量有label标签source的跨域自监督学习

## 域内原型对比学习

- 原方法只学习和对齐低级判别特征，没有考虑语义结构。典型的方法是：ProtoNCE通过迭代聚类和表示学习来学习单个域中数据的语义结构，使得同一簇中的特征变得更加聚合，而不同簇中的特征变得更远。它可能会出现误分类（不同域的实例分为一个簇，同域的实例分为不同簇）
- 本方法通过再 source 和 target 中执行原型对比学习，可以防止跨域图像的错误聚类和不加区分的特征学习。

对于 target 和 source 都会执行以下操作：

![image-20211009112549823](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009112549823-16337499512331.png)

其中vi是xi的存储特征向量，用fi初始化并在每个batch之后用一个动量m更新

![image-20211009115523945](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009115523945.png)

- $f_i$ 是训练得到的特征
- 动量 m 最开始从配置文件中读入，默认0.9，训练过程中未修改。

为了进行域内原型对比学习，在 Vs 和 Vt 上执行 kmeans 聚类以获得源聚类 $C^s 和相似聚类 C^t$，$C^s$和$C^t$分别包含已归一化的源原型${μ(s,j)}(k,j=1)$和目标原型${μ(t,j)}(k,j=1)$. 

原型的对比学习计算如下

- 先通过编码器 F 计算特征向量，然后求特征向量和原型的相似性分布向量。

![image-20211009121856106](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009121856106.png)

![image-20211009121915640](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009121915640.png)是热度

域内原型对比损失为

![image-20211009123555059](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009123555059.png)

$C_s(.)$ 返回的是聚簇的索引。考虑到聚类的随机性，一共做了 m 次聚类，且由于数据的类别是已知的，设置的类别数 = 聚类次数。

![image-20211009125950130](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009125950130.png)

##  Cross-domain Instance-Prototype SSL

为了域对齐和学习到更有辨别力的特征，执行跨域实例原型自监督学习。

先前的方法采用：实例到实例的对齐；性能差，训练不稳定；考虑的分布匹配，没有考虑语义相似性。

本文的方法：发现了实例和原型之间的匹配关系。通过在特征之间执行相似性分布向量熵最小化，找到对应的匹配关系。

如 $f_i^s 和 另一个域的质心 \{ μ_j^t \}^k_{j=1}$

给定 source 的一个特征向量 $f_i^s$ 和 target 的质心 $\{ μ_j^t \}^k_{j=1}$，计算它们的相似分布向量

![image-20211009132722762](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009132722762.png)

最小化熵

![image-20211009132738427](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009132738427.png)

然后以同样的方式计算 $H(P_i^{t->s})$，最终的 SSL 损失为：

![image-20211009132927459](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009132927459.png)

## Adaptive Prototypical Classifier Learning

自适应原型分类器学习。

这部分的目标是：

- 学习一个更好的域对齐，特征判别编码器和一个在 target 上的高精度的余弦分类器。

余弦分类器由权重向量组成 W = [ w1, w2,..., wnc ], nc 是总共的类别数。

![image-20211009145514783](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009145514783.png)

```python
class CosineClassifier(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
    """
    feat_lbd = self.model(images_lbd)
    feat_lbd = F.normalize(feat_lbd, dim=1)
    out_lbd = self.cls_head(feat_lbd)
    """
	# x 是对提取到的特征进行归一化后的结果
    def forward(self, x, reverse=False, eta=0.1):
        self.normalize_fc()

        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x)
        x_out = x_out / self.temp

        return x_out

    def normalize_fc(self):
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, eps=1e-12, dim=1)

    @torch.no_grad()
    def compute_discrepancy(self):
        self.normalize_fc()
        W = self.fc.weight.data
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()
```

![image-20211009150555784](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009150555784.png)

```python
# classification on few-shot
if ls == "cls-so" and domain_name == "source":
    # 分类损失
    loss_part = self.criterion(out_lbd, labels_lbd)
```

Ds 中的 label 很少，仅靠它是很难得到一个在 target 上的高精度分类器的。

<span style="color:red">因此提出了一个 自适应原型分类更新（Adaptive Prototype-Classifier Update (APCU)）</span>

权重向量wi的方向若能代表相应类别 i 的特征（即wi的含义与类 i 的理想簇原型一致），C就可以对样本进行正确分类 → 基于此思考，本文提出使用”对理想簇原型的估计来更新wi“的方法 → 计算出的质心$\{ μ_j^s \}$无法直接用来达到目的，原因有两方面：

- 1）权重向量$\{wi\}$和质心$\{μj\}$之间的对应关系未知；
- 2）kmeans结果包含不纯的簇，导致原型不具有代表性

**本文使用少量标记数据以及具有高可信度预测的样本来估计每个类的原型**

符号定义如下：

$D_s^{(i)} = \{X|(X,y)∈ D_s,y=i \}$，$D_{su}^{i} 和 D_{su}^{i}$ 表示 source 和 target 中高置信度的标签 i 的样本集合。

$p(x) = [p(x)_1，...，p(x)_{nc}]$

$D_{su}^{(i)} = \{ X|X ∈ D_{su},p(x)_i>t \}$，t 表示置信度阈值。$D_{su}^{i}$的定义也是类似的。

可以通过下面的式子来估计 $w_i$

![image-20211009153520557](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009153520557.png)

$V^s(x)$ 返回对应 x 中存储库中的表示

由于源域中只有很少的标记样本，因此很难学习跨域共享的代表性原型。本文提出以域自适应的方式更新$w_i$，而非使用类别 i 的全局原型。在早期的训练中使用$\hat w_{i}^{s}$（在早期训练更鲁棒） 在后期的训练中使用 $\hat w_{i}^{t}$（对目标域根据代表性），在后期可以得到更好的适配性。我们用高置信度的样本集合 $|D_{tu}^{(i)}|$ 来确定 $\hat w_i^s$更鲁棒。

![image-20211009162607841](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009162607841.png)

unit(·) 对输入的向量进行归一化，tw 是阈值超参数

**互信息最大化**

为了使上述统一的原型分类器学习范式能够正常工作，网络需要有足够的置信度预测，以获得鲁棒性好的权重向量。

- 为了保证输出的多样化，最大化了网络预测的熵 $H(E_{x∈D}[p(y|x;θ)])$，θ表示 编码器 F 和分类器 C 中的可学习的参数。D = Ds∪Dsu∪Dtu
- 为了对每个样本进行高置信度的预测，利用网络输出熵最小化，在标签少的常见非常有效。

上述两步等同于最大化输入和输出之间的互信息

![image-20211009172006208](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009172006208.png)

先验分布 $P_0$ 由 $E_{x∈D}[p(y|x;θ)]$ 给出。最终需要优化的目标为：

![image-20211009172238872](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009172238872.png)

## PCS Learning for FUDA

PCS 学习框架执行域内原型对比学习、跨域实例原型自监督学习和统一自适应原型分类器学习，总体学习目标为

![image-20211009172414608](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009172414608.png)

# Experiments

四个数据集上进行了实验。

- Office（真实世界的数据集，包含3个domain Amazon DSLR Webcam）和31种类别，每个类别分别取 1-shot 和 3-shot 源标签进行训练。
- Office-Home（4个 domain，65个类别）每类具有 3% 和 6% 标记源图像的设置，这意味着每个类平均有 2 到 4 个标记图像
- VisDA-2017（模拟真实世界的数据集）12 个类别的超过 28 万张图像。带标签的 source 分别取 0.1% 1%
- DomainNet（使用了其中四个 domain Clipart, Real, Painting, Sketch）126个类别。1-shot 和 3-shots 源标签进行训练

使用 ImageNet上预训练的 ResNet-101（DomainNet） 和 ResNet-50（其他数据集） 作为backbone。效果比 baseline 要好出不少。

![image-20211009182724137](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009182724137.png)

尝试将：source 上的熵最小化 (ENT) 添加到以前的 DA 方法中以获得更好的baseline性能

**消融实验**

![image-20211009183442182](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009183442182.png)

![image-20211009183540504](Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation.assets\image-20211009183540504.png)

