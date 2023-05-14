[论文链接](https://arxiv.org/abs/1808.00157)，[官方代码](https://github.com/Engineering-Course/CIHP_PGN)，[参考文章](https://l1aoxingyu.github.io/2018/12/21/instance-level-human-parsing论文笔记/)

这篇文章的主要有两个贡献，首先论文发表了一个新的`crowd human instance-level human parsing` 的数据集，其次论文提出了一个detection-free的unified结构PGN去解决多人的human parsing问题。

### 介绍

`human parsing`主要做的事情是识别人的每个部分的语义信息，是很多任务的基础。目前对人的human parsing主要有两种方法，一种是“parsing by detection”，即先做检测，再对每一个detection的person做human parsing。这样需要两个网络来通过若干独立的目标和阶段来训练，是的训练过程复杂，同时两个网络的目标函数粒度不同，detection是粗粒度的bounding box，而 parsing 是细粒度的pixel-level part segementation,如果用同一个backbone，会出现一些反常的效果。比如segmentation会因为detection的原因错误的在bbox的外面预测part semantic。

在本了论文中，作者对这个问题reformulate，用一个unified的网络对两个连续的分割任务进行分组，一个是part-level pixel grouping，一个是instance -level part grouping。首先part-level 的分割可以解决人体的各个部分的语义，然后通过instance-level的分割在将不同的人分开，这个一致的网络叫做Part Grouping Network（PGN）。

### 提出的方法

![1567649415448](https://wenliangsun.github.io/img/PGN/fig1.png)

#### Semantic part segmentation branch

在semantic的分支上，直接对pyramid pooling之后的第一个结果接一个1×11×1 的Conv，然后输出一个K个通道的map，对应于·目标的类别数目，包括背景类。

#### Instance-aware edge detection branch

在edge detection这个分支上，会对最后三个阶段的特征图做一个atrous spatial pyramid pooling，得到三个输出的特征图，然后做一个二分类任务，即1为边界0为背景。同时也像上面一个分支一样，将 feature map concatenate 在一起，同时用一个 pyramid pooling 得到不同位置的增强特征，然后再通过一个 1x1 的卷积得到一个 edge 的 feature map，同样做一个二分类。

#### Refinement branch

除了上面的两个分支单独来做之外，还有另外一个分支联合了segmentation 和edge的结果来做refinement。用一个1×11×1 的Conv 把segmentation score map 和edge score map映射到高维的embedding空间，然后Concat在一起分别经过两个pyramid pooling 再次得到segmentation 和 edge的score map。

#### Instance partition process

组合instance-aware edge和 semantic part segmentation，得到最终的instance-aware human parsing。

同时扫描 part segmentation maps 和 edge maps 得到水平和竖直的线。对于水平的线，从左往右，背景的 segmentation 直接被跳过，只有扫描到前景才算，如果击中了边界，这条线就终止，直到又扫描到前景，又开始一条新的线，每条线都有独立的编号，竖直方向和水平方向做法类似。

第二部分我们把竖直的线和水平的线看成一个连通图，同一条直线上的点是连通的，然后我们用 BFS 去遍历整个图，这样就可以得到一个不同的 instance，最后再组合 part segmentation 的结果，就得到了 instance part segmentation 的结果了，非常的直观。

![1567651031880](https://wenliangsun.github.io/img/PGN/fig2.png)

### 实验结果

![1567651087750](https://wenliangsun.github.io/img/PGN/fig3.png)

![1567651119555](https://wenliangsun.github.io/img/PGN/fig4.png)

