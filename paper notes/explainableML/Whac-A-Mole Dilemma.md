[A Whac-A-Mole Dilemma : Shortcuts Come in Multiples Where Mitigating One Amplifies Others](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_A_Whac-a-Mole_Dilemma_Shortcuts_Come_in_Multiples_Where_Mitigating_One_CVPR_2023_paper.pdf)

# Motivation
"Whac-A-Mole Dilemma": 当存在多个shortcuts时，减轻一个shortcut的影响可能会导致模型更加依赖其他shortcuts.

# Method
1. 为multi-shortcut问题提出了2个dataset
2. 全面评估现有的shortcut mitigation方法 (通过与ERM结果作对比的方式说明这些方法存在Whac-A-Mole问题)
3. 作者提出的解决multi-shortcut的方法: LLE (?怎么、为什么能解决这个问题)

# Related Work
## "Use Shortcut Labels for Mitigation"
- [Simple data balancing achieves competitive worst-group-accuracy](https://arxiv.org/abs/2110.14503#)
- :ballot_box_with_check:[Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)
  - existing work has identiﬁed many distributional shifts that can be expressed with pre-speciﬁed groups, e.g. image artifact ([Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging](https://arxiv.org/abs/1909.12475)), patient demographics in medicine ([Deep learning predicts hip fracture using confounding patient and healthcare variables](https://www.nature.com/articles/s41746-019-0105-1))
- *[Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation](https://arxiv.org/abs/1911.11834#)

## With only knowledge of the shortcut type
### "use architectural inductive biases"
- *[Learning Robust Representations by Projecting Superficial Statistics Out](https://arxiv.org/abs/1903.06256)
- *[Learning De-biased Representations with Biased Representations](https://arxiv.org/abs/1910.02806)

### "use augmentation"
- *[ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
- *[Characterizing and Improving the Robustness of SelfSupervised Learning through Background Augmentations](https://arxiv.org/abs/2103.12719)
- [Noise or Signal: The Role of Image Backgrounds in Object Recognition](https://arxiv.org/abs/2006.09994)

### "re-trains the last layer for mitigation"
- *[On Feature Learning in the Presence of Spurious Correlations](https://arxiv.org/abs/2210.11369)
- *[Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations](https://arxiv.org/pdf/2204.02937.pdf)
  
## Without knowledge of shortcut types
### "infer pseudo shortcut labels"
This approach is theoretically impossible: [ZIN: When and How to Learn Invariance Without Environment Partition?](https://arxiv.org/abs/2203.05818)
- 3,15,54,59,61,79,84,96

## improve OOD robustness
### self-supervised pretraining
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- *[Learning Debiased Classifier with Biased Committee](https://arxiv.org/abs/2206.10843)
### foundation models
- [Foundation Models大型综述](https://www.zhihu.com/question/498275802)
- 10,28,28,41,67,91,92
