# Universal Domain Adaptation for Remote Sensing Image Scene Classifification

DA 通常假设 source domain 有标签，target 无或仅有少量标签，而本文探讨的是更为实际的遥感 DA 场景：由于隐私或保密问题，源数据通常无法访问。

文章提出了一种不需要 source label sets（利用提供的 pretrained 估计 source 的分布，生成合成的 source data）的 DA 方法。方法分为两个阶段

- source data generation stage（估计 source 的 distribution，生成 source 的合成数据）
- model adaptation stage

提及了自然图像的 source-free 方法，但是这不适用于遥感，遥感的 source 有时你根本无法访问，只能获取到 pretrained model。

- 如何利用 pretrained model 合成 source domain data。
- 为什么不用 gans？因为 gans 是生成数据，而 UniDA 的目的则是恢复 source domain 的 distribution。
- 生成数据==>区分 target sample 是 shared label sets 还是 private label sets.