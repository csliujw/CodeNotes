![image-20210412210336781](D:\Code\note\CV-Study\pics\CV\ISG\Cell-DERT结构\image-20210412210336781.png)

- 主干网络：CNN编码器，提取图像特征。
    - 用的四个类似于`ResNet`的64、128、256、256的四个卷积块。每个块后用$2*2$的平均池化（average pooling）进行下采样。`Cell-DETR`采用不同的激活函数核卷积。

```shell

```

- transformer编码-解码器：确定图像特征之间的关注度。
    - The transformer decoder predicts the attention regions for each of the N = 20 object queries.
    - transformer解码器为每20个对象查询预测一次他们的注意力区域。
    - 我们把transformer encoder blocks 减少到了三个，decoder blocks减少到了两个，每个前馈神经网络中有512个hidden features.
    - 128的backbone feature在输入transformer前会被展平`与最初的DETR相反`，我们`使用了学习过的位置编码`
    - 包围盒和分类的预测头都是前馈神经网络。他们将transformer encoder-decoder输出映射到bounding box和classification prediction。这些`FFNN（前馈神经网络）`并行处理每个查询，并共享查询中的参数。（除了细胞和trap类，classification head还可以预测无对象类∅）

- segmentation head（分割头）：由multi-head注意力机制 和 CNN 解码器组成，用于预测每个对象实例的分割。
    - 在transformer encoder and decoder features上我们使用的是二维的multi-head注意力机制。
    - 注意力机制图的结果按通道连接到图像特征上，然后输入到CNN解码器。（The resulting attention maps are concatenated channel-wise onto the image features and fed into the CNN decoder.）
    - 三个类似`ResNet`的解码器块减小了特征通道的大小，同时增加了空间维度。
    - 在`CNN encoder` 和 `CNN decoder block` 输出间使用了跳跃连接。
        - 这些特征在`Cell-DETR A`模型中通过元素相加进行融合
        - 在`Cell-DETR B`中通过像素自适应卷积进行融合
        - 第四个卷积块将查询合并特征维度中，并为每个查询返回原始输入空间维度。所有查询的`softmax`保证了不重叠的分割

## 模型训练

- 联合损失函数（combined loss function）。[ 多任务的网络 ]

- direct set prediction

    - the set prediction
    - $\hat{y} = \{\hat{y}_i = \{ \hat{p}_i, \hat{b}_i, \hat{s}_i \}\}^{N=20}$
    - 包括对类别概率的相应预测
        - $\hat{p}_i∈R^k$  ；$\hat{p}_i$ 属于类别k的概率；k=3，因为这里仅三类（no-object，trap，cell）
        - $\hat{b}_i∈R^4$；包围盒的概率
        - $\hat{s}_i∈R^{128*128}$；

- 用匈牙利算法将每个实例集标签$y_{σ(i)}$分配给相应的查询集预测，索引σ识别标签的最佳匹配排列

- 联合损失函数的公式

    ![image-20210413185013503](D:\Code\note\CV-Study\pics\CV\ISG\Cell-DERT结构\image-20210413185013503.png)

    - $L_p$ 分类损失。
    - $L_b$ 包围盒损失，只算非空对象的。
    - $L_s$ 分割损失，只算非空对象的。

- 我们使用分类加权交叉熵来计算分类损失，权重为β=[ 0.5, 0.5, 1.5 ] K=3时（no-object，trap，cell classes）

    ![image-20210413185118585](D:\Code\note\CV-Study\pics\CV\ISG\Cell-DERT结构\image-20210413185118585.png)

- 包围盒损失本身由两个加权损失项组成。【文献33】

    ![image-20210413190232224](D:\Code\note\CV-Study\pics\CV\ISG\Cell-DERT结构\image-20210413190232224.png)

    - $λ_J$= 0.4
    - $λ_{L1}$= 0.6
    - These are a generalised intersection-over-union $L_J$ 

- 分割损失是焦点损失（focal loss）和Sorensen Dice loss的加权和。【文献6 8】

    ![image-20210413190206025](D:\Code\note\CV-Study\pics\CV\ISG\Cell-DERT结构\image-20210413190206025.png)

    - $λ_F$= 0.05 and $λ_D$= 1；focusing parameter γ = 2 and![image-20210419153301914](..\..\pics\CV\ISG\Cell-DERT结构\image-20210419153301914.png)= 1 for numerical stability