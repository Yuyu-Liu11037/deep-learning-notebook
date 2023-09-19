# 半监督/弱监督论文整理
- 任务类别: 图像分割、视频理解
- 会议/期刊: CVPR, ICCV, ECCV, MICCAI, MIA, TPAMI

## Survey
**Year** |**Pub.** |**Link** |**Contribution**
:-: | :-: | :- | :-
2022 | IET | [Medical image segmentation using deep learning: A survey](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12419) | 医学图像分割综述
2023 | arXiv | [A Survey on Semi-Supervised Semantic Segmentation](https://blog.csdn.net/CV_Autobot/article/details/129234235) | 半监督语义分割综述

## 半监督
### 2023
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
MIA | [Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841523001408) | 提出了一类target selection的策略来改进MT模型
TMI | [Anti-Interference From Noisy Labels: MeanTeacher-Assisted Conﬁdent Learning for Medical Image Segmentation](https://arxiv.org/abs/2106.01860) | 使用置信学习(confident learning)的方法改善MT架构中teacher的预测质量
CVPR | [MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf) | 双流网络做半监督，labeled data的差异区域额外使用MSE loss, unlabled data使用类似co-training的方法


### 2022
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
JBHI |[All-Around Real Label Supervision: Cyclic Prototype Consistency Learning for Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/2109.13930) | 为半监督训练提出了以真实标签为中心的循环原型一致性学习（CPCL）框架

### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
MICCAI | [Noisy Labels are Treasure: Mean-Teacher-assisted Confident Learning for Hepatic Vessel Segmentation](https://arxiv.org/abs/2106.01860) | 利用定点学习技术改进加权平均MT模型，更好地利用低质量数据
AAAI | [Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence](https://arxiv.org/abs/2012.04404) | 弱监督(涂鸦标签)SOD网络
CVPR | [CPS: Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://zhuanlan.zhihu.com/p/378120529) | 提出了新的半监督语义分割算法

### 2020
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### 2019
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### Previous
**Year**|**Pub.** |**Link** | **Brief Intro**
:-:|:-: | :-: | :-
2017|NIPS | [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780) | 提出mean teacher(MT)架构, 将 $\Pi$-model中对预测的EMA改成模型参数的EMA, 在每个step之后更新

## 弱监督
### 2023
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :- 
TPAMI | [Uncertainty-aware Dual-evidential Learning for Weakly-supervised Temporal Action Localization](https://ieeexplore.ieee.org/abstract/document/10230884/) | 
AAAI | [Weakly-Supervised Camouﬂaged Object Detection with Scribble Annotations](https://arxiv.org/abs/2207.14083) | 发布了第一个基于涂鸦标注的COD数据集(S-COD), 同时提出了第一个基于涂鸦标注的端到端COD网络

### 2022
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
MICCAI | [Scribble2D5: Weakly-Supervised Volumetric Image Segmentation via Scribble Annotations](https://arxiv.org/abs/2205.06779) | 3D涂鸦监督网络
MICCAI | [Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision](https://arxiv.org/abs/2203.02106) | 使用scribble supervised learning替代supervised learning, 用于医学图像分割

### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### 2020
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### 2019
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### Previous
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-