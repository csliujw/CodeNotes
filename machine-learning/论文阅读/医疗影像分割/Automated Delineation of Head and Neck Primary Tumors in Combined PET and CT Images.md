# Automated Delineation of Head and Neck Primary Tumors in Combined PET and CT
Images
# Abstract

模型使用的是带残差的 UNet 结构，补充了Squeezeand-Excitation Normalization（一种归一化方式）

# Introduction

放射组学工作流程的分割步骤是最耗时的瓶颈，通常半自动分割方法中的可变性会显著影响提取的特征，尤其是在手动分割的情况下，手动分割受到观察者之间和观察者内部最大幅度的可变性的影响。在这些情况下，完全自动化的分割对于整个过程的自动化和促进其临床常规使用是非常理想的。（不看）

用 Dice score(DSC), precision and recall. 进行评估。

# Materials & Methods

## SE Normalization

模型的关键在于 SE Normalization layers（该篇论文中提出的）

SE Normalization layers 类似于Instance Normalization，对于输入$X  = (x_1，x_2，...，x_N）$对于N个通道，SE范数层首先使用均值和标准差对一批中每个示例的所有通道进行归一化：
$$
x^{\prime}_i = \frac{1}{\sigma_i}(x_i - \mu _i) ----- (1)
$$
$\mu_i=E[x_i]$,  $\sigma_i=\sqrt{Var[x_i] + \epsilon}$,  $\epsilon$是一个非0的小常量，为了防止除0异常。**之后，将这对参数$\gamma_i, \beta_i $对每个通道进行==缩放和移动==  归一化值。**（**这个的计算方式得看看代码里怎么写的。**）
$$
y_i = \gamma_i x^{\prime}_i+\beta_i----(2)
$$
Instance Normalization是在训练过程中拟合两个参数$\gamma_i, \beta_i $，推理时保持固定，且与输入X独立。

而我们则建议通过Squeeze-and-Excitation (SE) blocks 将参数$\gamma_i, \beta_i $建模为输入X的函数，即
$$
\gamma = f_\gamma(X) ---- (3)
\\
\beta = f_\beta(X)----(4)
$$
$\gamma = (\gamma_1,\gamma_2,\gamma_3,....,\gamma_N) \ \ \ \beta=(\beta_1,\beta_2,\beta_3,....,\beta_N);$ - the scale and shift parameters for all channels.

$f_\gamma$ - the original SE block with the sigmoid

$f_\beta$ - is modeled as the SE block with the tanh activation function to enable the negative shift 

这两个SE块首先应用全局平均池(GAP)将每个通道压缩到一个描述符中，然后，两个全连接捕获非线性跨通道依赖关系。第一个FC层以缩减率r实现，形成一个bottleneck以便控制模型复杂度。 -- The first FC layer is implemented with the reduction ratio r to form a bottleneck for controlling model complexity.

在本文中，我们应用了具有固定缩减率r  = 2的SE范数层。

> **SE Normalization结构图**

<img src="..\..\pics\CV\Medical\image-20210629114004656.png">

----

<img src="..\..\pics\CV\Medical\image-20210629114154905.png">

## Network Architecture

模型以UNet为基础，增加了SE Norm layers。

> 解码器部分

模型解码的卷积块时$3*3*3$的卷积，激活函数用的ReLU，ReLU后用SE Norm layers。编码器中的残差块由具有快捷连接的卷积块组成 。如果残差块中的输入/输出通道的数量不同，会用1  × 1 × 1卷积块添加到shortcut来执行非线性投影，以便匹配维度。

> 编码器部分

下采样使用$2*2*2$的最大池化。

为了在解码器中线性上采样特征图，使用3×3×3转置卷积。

此外，我们通过应用1×1×1卷积块来减少信道数量，从而用三个上采样路径来补充解码器以在模型中进一步传递低分辨率特征，以在模型中进一步传递低分辨率特征。放置在输入之后的第一个残差块以7 × 7 × 7的核大小实现，以增加模型的感受野，而没有显著的计算开销。sigmoid函数用于目标类的输出概率。

- 输入是2个图，图大小为$144*144*144$
- encoder采用 Fig (b)中的结构，如果通道不一样就用Fig（c）
- decoder 转置卷积 + 结合低分辨率特征 + 残差连接

<img src="..\..\pics\CV\Medical\3DUNet_SE.png">

## Data Preprocessing & Sampling

PET 和 CT 图像首先通过三线性插值重新采样到 1 × 1 × 1 mm3 的共同分辨率。 每个训练示例都是从整个 PET/CT 图像中随机提取的 144 × 144 × 144 大小的图，而验证示例来自组织者提供的边界框。 提取训练补丁以包含概率为 0.9 的肿瘤类别，以促进模型训练。（所有的图片变成通用的大小，在随机取$144*144*144$大小的图，每个像素有score，score>0.9才是肿瘤。）

CT强度在[1024，1024]  Hounsfield单位范围内限幅，然后映射到[1，1]。正电子发射断层扫描图像通过使用在每一片上执行的Z分数归一化独立地被转换。

## Training Procedure

Adam优化器

Batch Size = 2

The cosine annealing schedule was applied to reduce the learning rate from $10^{−3}$ to $10^{−6}$within every 25 epochs.

## Loss Function

使用Sotf Dice Loss 和 Focal Loss，未加权。
$$
L_{Dice}(y,\hat y) = 1 - \frac{2\sum_{i}^{N}y_i \hat y_i+1}{\sum_i^Ny_i+\sum_i^N \hat y_i + 1} ----(5)
$$

$$
L_{Focal}(y,\hat y) = -\frac{1}{N}\sum_i^Ny_i(1-\hat y_i)^{\gamma}*ln(\hat y_i)----(6)
$$

$y_i \in \{0,1\}-- the \ label \ for \ the \ i-the \ voxel$

$\hat y_i \in \{0,1\}-- the \ predicted \ probability \ for \ the \ i-the \ voxel$

$N \ - \ the \ total \ numbers \ of \ voxels$

$L_{Dice}$分子分母+1是为了避免训练中不存在肿瘤类别的情况，避免除0异常。

$The \ parameter \ γ \ in \ the \ Focal \ Loss \ is \ set \ at \ 2.$   $\gamma$设置的值是2

