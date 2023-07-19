# Evaluation Metrics

## General
**No.**|**Name**|**Usage**
:- |:- |:-
01 | [Cross Entropy Loss](https://zhuanlan.zhihu.com/p/54066141) | 量化两个概率分布之间差异的损失函数（多用于分类问题）
02 | [Area Under the Curve (AUC)](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) |评估二分类模型性能的一种常见指标
02'| mean Area Under the Curve(mAUC) |衡量多类别分类模型性能的指标
03 | Pearson correlation coefficien | 衡量两个变量之间线性相关程度的统计量. Pearson相关系数的取值范围在-1到1之间, 当相关系数为1时，表示两个变量之间存在完全正向线性关系.

## Image Segmentation
**No.**|**Name**|**Usage**
:- |:- |:-
01|[Dice similarity coefﬁcient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)|衡量两个集合的相似度, 最常用于评估图像分割算法的性能
01'|Normalized Surface Dice|Dice相似系数的一种变体，评估图像分割算法在边界准确性方面的性能
02 |Intersection-over-Union (mIoU) |交叉比：预测分割结果中与真实标注中属于该类别的像素数量与两者的并集像素数量的比. 语义分割任务中常用的一个性能指标，范围在0到1之间，值越接近1表示模型预测的分割结果和真实标注越吻合，性能越好。
02' | mean Intersection-over-Union (mIoU) |所有类别IoU的平均值，表示模型在所有类别上的平均分割准确度

## Medicine
**No.**|**Name**|**Usage**
:- |:- |:-
01 |[Sensitivity and Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)|**Sensitivity** (true positive rate) is the probability of a positive test result, conditioned on the individual truly being positive, 反映了模型对于正例的检测能力. **Specificity** (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative, 反应了模型的误诊率.