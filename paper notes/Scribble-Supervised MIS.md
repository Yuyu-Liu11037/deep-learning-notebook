[Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision](https://arxiv.org/abs/2203.02106)

---
# Motivation

医学图像分割所需数据集的逐像素标注通常很昂贵。使用稀疏标注(sparse annotation)的弱监督学习(weakly-supervised learning)有望解决这一问题。

伪标签(Pseudo-labeling)方法被广泛用于半监督学习(semi-supervised learning)中。不过也有[相关工作](https://arxiv.org/abs/2006.12890)表明，对于弱监督学习，伪标签方法也能改善效果。

受以上两种方法的启发，作者提出了一个结合了稀疏标注和伪标签方法的分割网络。


---
# Details
![Overall Architecture](../images/Scribble-SupervisedMIS.jpg "Architecture")
下面是对该网络的说明:
- 一个共享的编码器和两个解码器. 这两个解码器区别在于输入的特征编码有没有加 dropout. 这样的设计类似于半监督学习里的consistency training / knowledge distillation, 作用是抑制 [inherent weakness of pseudolabel in the single branch network](https://zhuanlan.zhihu.com/p/604063439) (remembering itself predictions without updating).
- 直接用涂鸦标签训练网络: 计算部分交叉熵损失(partial cross-entropy loss, pCE)
   $$L_{pCE}(y,s)=-\Sigma_{c}\Sigma_{i\in\omega_s}\log y_i^c$$
   其中 $s$: 独热涂鸦标注(one-hot scribble annotations). $y_i^c$: 观测样本$i$ 属于类别 $c$ 的预测概率. $\omega_s$: $s$中有标注的像素.
- 通过伪标签训练网络：利用两个解码器的输出 $y_1,y_2$ 生成伪标签 $PL$:
  $$PL=argmax[\alpha\times y_1 + (1.0-\alpha)\times y_2],\;\alpha=random(0,1)$$
  其中 $\alpha$ 在每个epoch中随机取值。再计算伪标签损失 $L_{PLS}$：

  $$L_{PLS}(PL, y_1, y_2)=0.5\times (L_{Dice}(PL, y_1)+L_{Dice}(PL,y_2))$$

  其中 $L_{Dice}\coloneqq$ dice loss.
- 最后，总的损失函数为：
  $$L_{total}=0.5\times(L_{pCE}(y_1,s)+L_{pCE}(y_2,s)) + \lambda \times L_{PLS}(PL,y_1,y_2)$$
  其中 $\lambda$ 是可调整的权重参数。


---
# Experiments
Dataset: [ACDC](https://paperswithcode.com/sota/medical-image-segmentation-on-automatic) via [five-fold cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). 每张扫描有三个结构的密集注释，包括右心室（RV）、心肌（Myo）和左心室（LV）。

Evaluation metrics:

Backbone network: UNet

## Quantitative Analysis
![Table1](../images/Scribble-SupervisedTable1.png "Result1")
"Mean" 代表 average results
1. WSL: 与7个弱监督学习方法比较.
2. SSL: 与4个半监督学习方法比较 (similar annotation budget).
3. FSL: 与逐像素标注的全监督学习方法比较.

## Sensitivity Analysis of $\lambda$
![Figure](../images/Scribble-SupervisedFig.png "Analysis")
分析不同权重的伪标签损失的作用。

## Ablation Study
![Figure](../images/Scribble-SupervisedTable2.png "Analysis")
探究使用不同监督策略对网络表现的影响：
1. Consistency Regularization (CR) 
2. Cross Pseudo Supervision (CPS) 
3. The proposed approach 

除此之外，这篇文章还探究了，生成伪标签过程中，用于调整两个解码器输出占比的 $\alpha$ 在固定值0.5和随机值时，网络的表现。