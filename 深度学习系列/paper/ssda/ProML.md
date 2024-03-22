# 2023 CVPR ProML

## 概述

多级原型学习的 SSDA 框架。

- 域内对齐：target strong aug target 对齐 target prototype
  - 对齐标记和未标记样本在目标域内的数据分布
  - 拉进弱增强目标样本和目标原型的距离
  - 同时也拉进强增强样本和目标原型的距离
  - <b>损失计算</b>
    - 弱增强的目标样本使用由目标原型计算的最优迁移计划生成伪标签，并与强增强的样本计算一致性损失
    - <span style="color:red">最优迁移计划生成伪标签什么意思？</span>
- 域间对齐：source labeled feature 对齐 target prototype
  - 将源样本与目标原型的同一类别进行交叉对齐
  - <b>损失计算</b>
    - 计算源样本与相应类别的目标原型之间的相似性损失，以实现跨域知识迁移
- 批量对齐：batch 之间，target weak aug feature 对齐 target strong aug feature
  - 考虑一个小批样本来计算不同增强的预测之间的类相关矩阵，增加了同一类的相关性，减少了不同类的相关性。
  - <b>损失计算</b>
    - 从线性分类器和基于原型的分类器的视角考虑每个小批量中两个增强视图的双重一致性损失

## 核心做法

### Intra-domain Pseudo-label Aggregation

对应上文提及的“域内对齐”。



