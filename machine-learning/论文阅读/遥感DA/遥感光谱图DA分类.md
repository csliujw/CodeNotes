# Cross-Scene Hyperspectral Image Classification With Discriminative Cooperative Alignment

我怎么读取遥感数据呢？

## Abstract

多数方法采用统计偏移（statistical shift）学习**不变子空间**（invariant subspace）。如果分布差异大，则很难学习到这个空间；且对于分类来说在原来的空间中保存每个 domain 差异信息也很重要（<span style="color:red">为什么重要？是如果只保存了不变的信息，两个域不同类别的图片相似，误分类吗？</span>）。本文提出了 geometrical and statistical alignments ，在保留 discrimination（差异） 信息的情况下去学习两个域的子空间。（学习的子空间会保留各自空间数据的一些特点）

同时加入了 reconstruction constraint 以增强算法的鲁棒性。

## Introduction

直接在 source 中训练模型，然后在 target 中预测，如果两个数据集分布不一样，预测效果不会太好。

> 遥感数据的特点：

即便是同样的物体，也会有不同的分布。

- 用的不同的传感器拍摄的同一地方
- 拍摄区域的图片大小不一样（分辨率）
- 不同时间，在同一地点拍摄（区域内可能会发生一些变化）

这些都会使即便 S 和 T 来自同一个 ground Object，它们的数据分布和 spectral reflectance（光谱折射率）也会不同。

**光谱折射率：**地表物体反射、发射的电磁波经过与大气等的相互作用以后到达传感器，之后被各种传感器接受并记录下来。不同的地物目标在不同环境条件下反射、发射的电磁辐射的强度和性质是不同的，传感器记录下的这些信息可以作为地物目标判别、自然现象的识别等的依据。

[1.2遥感数据 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/54222756) 遥感数数据解释。

目的：减小光谱偏移（spectral shift）

> 传统领域自适应

- 基于分布对齐（减小 Source 和 Target 的分布差异），有三种常见的对齐方式，边缘分布对齐，条件分布对齐，联合分布对齐。
    - 
- 特征选择（假定它们有共同的特征，The feature selection-based methods assume that both SD and TD contain part of common features, where the data distribution of SD and TD are consistent）；用机器学习算法提取 share feature，然后用这些特征构建模型
    - 如果没有共同特征，这个方法就失效了
- 子空间学习（本文学习子空间）通常会假设 S 和 T 在子空间中分布类似。可以根据特征的转换形式，分为两类：
    - 基于统计特征变换的统计特征对齐方法
    - 基于流形变换的流形学习方法（manifold learning methods based on manifold transformation）
        - 流形学习是把一组在高维空间中的数据在低维空间中重新表示，在减维的过程中，保持流形上点的某些几何性质特征。
        - [流形学习(Manifold Learning)_chenaiyanmie的博客-CSDN博客_流形学习](https://blog.csdn.net/chenaiyanmie/article/details/80167649)

