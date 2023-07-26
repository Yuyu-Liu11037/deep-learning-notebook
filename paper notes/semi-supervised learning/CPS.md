[作者知乎解读](https://zhuanlan.zhihu.com/p/378120529)


[Consistency Regularization小结](https://zhuanlan.zhihu.com/p/46893709)

# Motivation

语义分割数据集的标注通常很昂贵，而半监督学习有希望解决这一问题。

半监督学习通常分为以下几类:
- **Pseudo-labeling methods**: self-training, mutual training
- **Consistency regulation**: mean teacher, CPC, PseudoSeg
- **Adversarial methods**
- **Contrastive learning**
- **Hybrid methods**

其中，一致性正则化 (consistency regulation) 鼓励网络学习一个紧致的特征编码；自训练(self-training)可以通过伪标签扩充数据集。以往的方法一般关注一致性正则化而忽视了自训练. 作者发现自训练在数据量不那么小的时候，性能非常的好。于是便提出了两者(思想上)的结合：CPS (cross pseudo supervision)。

# Details
![Fig1](../images/CPS1.png "Fig")

Figure 1(a)是该论文提出的方法。它由2个结构相同但参数初始化不同 (使用kaiming_normal进行两次随机初始化) 的网络构成。输入 $X$经过同样的增强 (即 CutMix augmentation) 后输入这两个网络，softmax层的输出为 $P_1,P_2$. 接下来把 $P_1, P_2$转化为独热标签 (one-hot label map) $Y_1,Y_2$.

训练过程包含两种损失函数:
- 监督损失 $\mathrm{L}_s$: 仅在有标签数据上的标准交叉熵损失
- 交叉伪标签损失(cross pseudo supervision loss) $\mathrm{L}_{cps}$: 双向，且作用在有标签和无标签的数据上， $Y_1$ 监督 $P_2$, $Y_2$ 监督 $P_1$ .

总的损失函数: $\mathrm{L}=\mathrm{L}_s+\lambda\mathrm{L}_{cps}$.

# Experiments
Datasets: PASCAL VOC 2012, Cityscapes

Evaluation metric: mean Intersection-over-Union (mIoU)

Implementation: DeepLabv3+ (segmentation head) with ResNet-50 or ResNet-101

## Improvements over baselines
![Fig1](../images/CPS2.png "Fig")
在Cityscapes上，使用不同的分割策略，以及是否加CutMix augmentation，对比论文方法与有监督的 baseline. 所有方法都基于 DeepLabv3+  with ResNet-50 / ResNet-101.

## Comparison with SOTA
![Fig1](../images/CPS3.png "Fig")
在Cityscapes和PASCAL VOC 2012上，使用不同的分割策略，以及是否加CutMix augmentation，对比论文方法与semi-supervised segmentation methods. 所有方法都based on DeepLabv3+  with ResNet-50 or ResNet-101.

## Improving Full- and Few-Supervision
### Full-supervision
![Fig1](../images/CPS4.png "Fig")
baseline models使用整个Cityscapes train set训练；论文方法使用整个Cityscapes train set和另外3000张unlabeled image (from Cityscapes coarse set)训练。

### Few-supervision
![Fig1](../images/CPS5.png "Fig")
在使用少量labeled images和大量unlabeled images来训练的情况下，对比其他几个方法和论文方法。

## Empirical Study
![Fig1](../images/CPS6.png "Fig")
### Cross pseudo supervision
研究了cps loss作用于 labeled set/unlabeled set/both 带来的影响，见Table 4第3、4、5行。
<!-- cps loss作用于labeled set，是指完全代替ce loss吗? -->

### Comparison with cross probability consistency (CPC)
比较CPS loss与CPC loss (即Figure 1(b))，见Table 4最后两行。

![Fig1](../images/CPS7.png "Fig")
### The trade-off weight $\lambda$
研究了不同$\lambda$的值的影响. ($\lambda$用于平衡监督损失和CPS损失)

![Fig1](../images/CPS8.png "Fig")
### investigate the inﬂuence of different $\lambda$ that is used to balance the supervision loss and CPS loss
对比CPS和single network pseudo supervision. (single-network pseudo supervision with the CutMix augmentation类似于FixMatch应用于语义分割，就像PseudoSeg. 作者认为这个结果说明了论文方法优于PseudoSeg.)

![Fig1](../images/CPS9.png "Fig")
![Fig1](../images/CPS10.png "Fig")
### Combination/comparison with self-training
Table 7: 对比仅使用CPS、仅使用自训练、两者结合训练的网络。

Figure 5: 对比使用自训练和增加了额外epoch的CPS训练的网络。(这是因为由于自训练网络是multi-stage的，它需要更多epoch)

## Qualitative Results
![Fig1](../images/CPS11.png "Fig")
分割结果的可视化。