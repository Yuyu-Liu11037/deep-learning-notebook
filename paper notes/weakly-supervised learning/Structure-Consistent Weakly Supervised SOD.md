[Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence](https://arxiv.org/abs/2012.04404)

# Motivation
![pic](../images/StructureConsis1.png "首页右上角插图")
对于显著性物体检测(salient object detection, SOD)任务，由于逐像素标记的成本太高，使用稀疏标注的弱监督学习成为新的考虑。然而，以往的弱监督网络都比较复杂，比如需要预/后处理、额外的边缘检测。

因此，本文提出了一个只需一个阶段、端到端的弱监督SOD网络。

# Details
![pic](../images/StructureConsis2.png "Model")

## Aggregation Module (AGGM)
![pic](../images/StructureConsis3.png "AGGM")
每个AGGM模块都分别加权聚合来自编码器的低层、高层、全局上下文信息：
$$f_{out}=\frac{w_h f_h+w_g f_g+w_l f_l}{w_h+w_g+w_l}$$
权重是通过网络自学习得到的，即 Figure 3 中的3x3卷积层和全局平均池化层。

## Local Saliency Coherence Loss ($L_{lsc}$)
弱监督学习中，大量未标注像素使得物体的边缘难以准确定位。因此，作者设计了局部显著一致性损失(LSC loss):
$$L_{lsc}=\sum_i\sum_{j\in K_i}F(i,j)D(i,j)$$
- i,j是两个像素点，$K_i$是以i为中心、$k\times k$的区域
- $F(i,j)=\frac{1}{w}exp(-\frac{\|P(i)-P(j)\|_2}{2\sigma^2_P}-\frac{\|I(i)-I(j)\|_2}{2\sigma^2_I})$, $P(i), I(i)$分别是像素i的位置和RGB颜色
- $D(i,j)=|S_i-S_j|$, $S_i$是i的显著性分数预测值。

这样的设计可以使具有相似特征/相近距离的像素有类似的显著性得分。

## Self-consistent Mechanism ($L_{ssc}$)
![pic](../images/StructureConsis4.png "Self-consistent")
一个好的SOD模型应当对同一幅图、不同尺寸输入的预测结果保持一致，即：记参数为 $\theta$ 的网络为 $f_\theta(\cdot)$, 变换 (transformation) 为 $T(\cdot)$, 输入图像为 $x$，则一个理想的网络应该满足：$f_\theta(T(x))=T(f_\theta(x))$。然而如 Figure 4 所示，弱监督网络很难做到这一点。

为了解决上述问题，作者把上式考虑成一种正则化方式，设计了针对不同输入尺寸图片预测结果的结构一致性损失(structure consistency loss)：
![pic](../images/StructureConsis5.png "Self-consistent")

## Objective Function
$$L_{total}=L_{dom}+\sum_{q=1}^3\lambda_q L_{aux}^q,\quad q\in\{1,2,3\}$$
$$L_{dom}=L_{ce}+L_{ssc}+\beta L_{lsc}$$
$$L_{aux}^q=L_{ce}+\beta L_{lsc}$$
$\lambda_q,\beta$为超参数 ($\beta$ 由后面两个公式共享)。

# Experiments
