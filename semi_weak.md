# 半监督/弱监督论文整理
- 任务类别: 语义分割、视频理解、图像分类、目标检测
- 会议/期刊:
  - CVPR: 2023, 2022, 2021 
  - ICCV: 2023, 2021
  - MICCAI: 2022
  - ECCV, NIPS, AAAI, ICLR, MIA, TPAMI, TMI, IJCAI
- 其它链接: 

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
CVPR | [Pseudo-label Guided Contrastive Learning for Semi-supervised Medical Image Segmentation](https://paperswithcode.com/paper/pseudo-label-guided-contrastive-learning-for) | 
CVPR | [MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery](https://arxiv.org/abs/2212.14310)


### 2022
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
JBHI |[All-Around Real Label Supervision: Cyclic Prototype Consistency Learning for Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/2109.13930) | 为半监督训练提出了以真实标签为中心的循环原型一致性学习（CPCL）框架
CVPR | [ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation](https://mp.weixin.qq.com/s/knSnlebdtEnmrkChGM_0CA) | 
CVPR | [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels](https://mp.weixin.qq.com/s/-08olqE7np8A1XQzt6HAgQ) | 
CVPR | [Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation](https://arxiv.org/pdf/2111.12903.pdf) | 
CVPR | [BoostMIS: Boosting Medical Image Semi-supervised Learning with Adaptive Pseudo Labeling and Informative Active Annotation](https://arxiv.org/abs/2203.02533) | 
CVPR | [Anti-curriculum Pseudo-labelling for Semi-supervised Medical Image Classification](https://arxiv.org/abs/2111.12918) | 
MICCAI | [ACT: Semi-supervised Domain-adaptive Medical Image Segmentation with Asymmetric Co-Training](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_7) | 
MICCAI | [Addressing Class Imbalance in Semi-supervised Image Segmentation: A Study on Cardiac MRI]() | 
MICCAI | [Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation]() | 
MICCAI [Censor-aware Semi-supervised Learning for Survival Time Prediction from Medical Images]() | 
MICCAI | [Clinical-realistic Annotation for Histopathology Images with Probabilistic Semi-supervision: A Worst-case Study]() | 
MICCAI | [Consistency-based Semi-supervised Evidential Active Learning for Diagnostic Radiograph Classification]() | 
MICCAI | [Dynamic Bank Learning for Semi-supervised Federated Image Diagnosis with Class Imbalance]() | 
MICCAI | [FUSSNet: Fusing Two Sources of Uncertainty for Semi-Supervised Medical Image Segmentation]() | 
MICCAI | [Leveraging Labeling Representations in Uncertainty-based Semi-supervised Segmentation]() | 
MICCAI | [Momentum Contrastive Voxel-wise Representation Learning for Semi-supervised Volumetric Medical Image Segmentation]() | 
MICCAI | [Reliability-aware Contrastive Self-ensembling for Semi-supervised Medical Image Classification]() | 
MICCAI | [S5CL: Unifying Fully-Supervised, Self-Supervised, and Semi-Supervised Learning Through Hierarchical Contrastive Learning]() | 
MICCAI | [SD-LayerNet: Semi-supervised retinal layer segmentation in OCT using disentangled representation with anatomical priors]() | 
MICCAI | [Semi-supervised Learning for Nerve Segmentation in Corneal Confocal Microscope Photography]() |
MICCAI | [Semi-Supervised Medical Image Classification with Temporal Knowledge-Aware Regularization]() | 
MICCAI | [Semi-Supervised Medical Image Segmentation Using Cross-Model Pseudo-Supervision with Shape Awareness and Local Context Constraints]() | 




### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
MICCAI | [Noisy Labels are Treasure: Mean-Teacher-assisted Confident Learning for Hepatic Vessel Segmentation](https://arxiv.org/abs/2106.01860) | 利用定点学习技术改进加权平均MT模型，更好地利用低质量数据
AAAI | [Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence](https://arxiv.org/abs/2012.04404) | 弱监督(涂鸦标签)SOD网络
CVPR | [CPS: Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://zhuanlan.zhihu.com/p/378120529) | 提出了新的半监督语义分割算法
CVPR | [Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation](https://arxiv.org/abs/2103.04705) | 
CVPR | [Semi-Supervised Semantic Segmentation With Directional Context-Aware Consistency](https://openaccess.thecvf.com/content/CVPR2021/html/Lai_Semi-Supervised_Semantic_Segmentation_With_Directional_Context-Aware_Consistency_CVPR_2021_paper.html) | 
CVPR | [Semantic Segmentation With Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Semantic_Segmentation_With_Generative_Models_Semi-Supervised_Learning_and_Strong_Out-of-Domain_CVPR_2021_paper.html) | 
CVPR | [Three Ways To Improve Semantic Segmentation With Self-Supervised Depth Estimation](https://openaccess.thecvf.com/content/CVPR2021/html/Hoyer_Three_Ways_To_Improve_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_CVPR_2021_paper.html) | 
ICCV | [Spatial Uncertainty-Aware-Semi-Supervised-Crowd-Counting](https://arxiv.org/abs/2107.13271) | 
ICCV | [Trash to Treasure: Harvesting OOD Data with Cross-Modal Matching for Open-Set Semi-Supervised Learning](https://arxiv.org/abs/2108.05617) | 
ICCV | [Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments With Support Samples](https://arxiv.org/abs/2104.13963) | 
ICCV | [Semi-Supervised Active Learning for Semi-Supervised Models: Exploit Adversarial Examples With Graph-Based Virtual Labels](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Semi-Supervised_Active_Learning_for_Semi-Supervised_Models_Exploit_Adversarial_Examples_With_ICCV_2021_paper.pdf) | 
ICCV | [CoMatch: Semi-Supervised Learning With Contrastive Graph Regularization](https://arxiv.org/abs/2011.11183) | 
ICCV | [Multiview Pseudo-Labeling for Semi-supervised Learning from Video](https://arxiv.org/abs/2104.00682) | 
ICCV | [Graph-BAS3Net: Boundary-Aware Semi-Supervised Segmentation Network With Bilateral Graph Convolution](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Graph-BAS3Net_Boundary-Aware_Semi-Supervised_Segmentation_Network_With_Bilateral_Graph_Convolution_ICCV_2021_paper.pdf) | 
ICCV | [Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-Supervised Polyp Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_and_Adversarial_Learning_of_Focused_and_Dispersive_Representations_for_ICCV_2021_paper.pdf) | 
ICCV | [Semi-Supervised Active Learning with Temporal Output Discrepancy](https://arxiv.org/abs/2107.14153) | 
ICCV | [Warp-Refine Propagation: Semi-Supervised Auto-labeling via Cycle-consistency](https://arxiv.org/abs/2109.13432) | 
ICCV | [Semi-Supervised Semantic Segmentation With Pixel-Level Contrastive Learning From a Class-Wise Memory Bank](https://arxiv.org/abs/2104.13415) | 

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
CVPR | [Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2203.00962) | 
CVPR | [Multi-class Token Transformer for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02891) | 
CVPR | [Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers](https://arxiv.org/abs/2203.02664) | 
CVPR | [CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02668) | 
CVPR | [CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation](https://arxiv.org/abs/2203.13505)
CVPR | [FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation](https://arxiv.org/abs/2204.01587) | 
CVPR | [Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.09653) | 
CVPR | [Scribble-Supervised LiDAR Semantic Segmentation](https://arxiv.org/abs/2203.08537) | 
MICCAI | [Anatomy-Guided Weakly-Supervised Abnormality Localization in Chest X-rays](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_22) | 
MICCAI | [Point Beyond Class: A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays]() | 
MICCAI | [ShapePU: A New PU Learning Framework Regularized by Global Consistency for Scribble Supervised Cardiac Segmentation]() | 
MICCAI | [Transformer based multiple instance learning for weakly supervised histopathology image segmentation]() | 
MICCAI | [Uncertainty Aware Sampling Framework of Weak-Label Learning for Histology Image Classification]() | 
MICCAI | [Weakly Supervised MR-TRUS Image Synthesis for Brachytherapy of Prostate Cancer]() | 
MICCAI | [Weakly Supervised Online Action Detection for Infant General Movements]() | 
MICCAI | [Weakly Supervised Segmentation by Tensor Graph Learning for Whole Slide Images]() | 
MICCAI | [Weakly-supervised Biomechanically-constrained CT/MRI Registration of the Spine]()



### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
CVPR | [Railroad Is Not a Train: Saliency As Pseudo-Pixel Supervision for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.html) | 
CVPR | [Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2104.00905) | 
CVPR | [Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2103.14581) | 
CVPR | [Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Embedded_Discriminative_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2021_paper.html) | 
CVPR | [BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation](https://arxiv.org/abs/2103.08907)
ICCV | [Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping](https://arxiv.org/abs/2108.06816) | 
ICCV | [Weakly Supervised Representation Learning With Coarse Labels](https://arxiv.org/abs/2005.09681) | 
ICCV | [Normalization Matters in Weakly Supervised Object Localization](https://arxiv.org/abs/2107.13221) | 目标定位
ICCV | [Online Refinement of Low-Level Feature Based Activation Map for Weakly Supervised Object Localization](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Online_Refinement_of_Low-Level_Feature_Based_Activation_Map_for_Weakly_ICCV_2021_paper.pdf) |目标定位 
ICCV | [Foreground Activation Maps for Weakly Supervised Object Localization](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_Foreground_Activation_Maps_for_Weakly_Supervised_Object_Localization_ICCV_2021_paper.pdf) |目标定位 
ICCV | [Boosting Weakly Supervised Object Detection via Learning Bounding Box Adjusters](https://arxiv.org/abs/2108.01499) | 目标检测
ICCV | [CaT: Weakly Supervised Object Detection With Category Transfer](https://arxiv.org/abs/2108.07487) | 目标检测
ICCV | [Detector-Free Weakly Supervised Grounding by Separation](https://arxiv.org/abs/2104.09829) | 目标检测
ICCV | [Robust Trust Region for Weakly Supervised Segmentation](https://arxiv.org/abs/2104.01948) | 
ICCV | [Weakly Supervised Segmentation of Small Buildings With Point Labels](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Weakly_Supervised_Segmentation_of_Small_Buildings_With_Point_Labels_ICCV_2021_paper.pdf) | 
ICCV | [Scribble-Supervised Semantic Segmentation by Uncertainty Reduction on Neural Representation and Self-Supervision on Neural Eigenspace](https://arxiv.org/abs/2102.09896) | 
ICCV | [Scribble-Supervised Semantic Segmentation Inference](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Scribble-Supervised_Semantic_Segmentation_Inference_ICCV_2021_paper.pdf) | 
ICCV | [Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2107.11787) | 
ICCV | [Complementary Patch for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2108.03852) | 
ICCV | [ECS-Net: Improving Weakly Supervised Semantic Segmentation by Using Connections Between Class Activation Maps](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_ECS-Net_Improving_Weakly_Supervised_Semantic_Segmentation_by_Using_Connections_Between_ICCV_2021_paper.pdf) | 
ICCV | [Unlocking the Potential of Ordinary Classifier: Class-Specific Adversarial Erasing Framework for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf) | 
ICCV | [Context Decoupling Augmentation for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2103.01795) | 
ICCV | [Seminar Learning for Click-Level Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2108.13393) | 
ICCV | [A Weakly Supervised Amodal Segmenter with Boundary Uncertainty Estimation](https://arxiv.org/abs/2108.09897) | 实例分割
ICCV | [DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence From Box Supervision](https://arxiv.org/abs/2105.06464) | 实例分割
ICCV | [Parallel Detection-and-Segmentation Learning for Weakly Supervised Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Shen_Parallel_Detection-and-Segmentation_Learning_for_Weakly_Supervised_Instance_Segmentation_ICCV_2021_paper.pdf) | 实例分割
ICCV | [Weak Adaptation Learning: Addressing Cross-Domain Data Insufficiency With Weak Annotator](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Weak_Adaptation_Learning_Addressing_Cross-Domain_Data_Insufficiency_With_Weak_Annotator_ICCV_2021_paper.pdf) | 域适应



### 2020
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### 2019
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### Previous
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

## 语义分割
### 2023
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :- 
CVPR | [Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos](https://arxiv.org/abs/2303.07224) | 
CVPR | [FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding](https://arxiv.org/abs/2304.02135)

### 2022
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
CVPR | [Novel Class Discovery in Semantic Segmentation](https://arxiv.org/abs/2112.01900) | 
CVPR | [Deep Hierarchical Semantic Segmentation](https://arxiv.org/abs/2203.14335) | 
CVPR | [Rethinking Semantic Segmentation: A Prototype View](https://arxiv.org/abs/2203.15102)
MICCAI | [Link](https://conferences.miccai.org/2022/papers/categories/#Image%20Segmentation)

### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
CVPR | [HyperSeg: Patch-wise Hypernetwork for Real-time Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Nirkin_HyperSeg_Patch-Wise_Hypernetwork_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf) | 
CVPR | [Rethinking BiSeNet For Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188) | 
CVPR | [Progressive Semantic Segmentation](https://arxiv.org/abs/2104.03778) | 
CVPR | [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840) | 
CVPR | [Capturing Omni-Range Context for Omnidirectional Segmentation](https://arxiv.org/abs/2103.05687) | 
CVPR | [Learning Statistical Texture for Semantic Segmentation](https://arxiv.org/abs/2103.04133)|
CVPR | [InverseForm: A Loss Function for Structured Boundary-Aware Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Borse_InverseForm_A_Loss_Function_for_Structured_Boundary-Aware_Segmentation_CVPR_2021_paper.html) | 
CVPR | [DCNAS: Densely Connected Neural Architecture Search for Semantic Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_DCNAS_Densely_Connected_Neural_Architecture_Search_for_Semantic_Image_Segmentation_CVPR_2021_paper.html) | 

### 2020
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### 2019
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

### Previous
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
 

## 其它
**Year** |**Pub.** |**Link** |**Contribution**
:-: | :-: | :- | :-
2023 | CVPR | [Masked Image Training for Generalizable Deep Image Denoising](https://arxiv.org/abs/2303.13132) | 图像去噪
2023 | ICCV | [BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification](https://arxiv.org/abs/2203.01937) | 医学图像分类
2022 | CVPR | [DiRA: Discriminative, Restorative, and Adversarial Learning for Self-supervised Medical Image Analysis](https://arxiv.org/abs/2204.10437) | 自监督医学图像分割
2022 | MICCAI | [Frequency-Aware Inverse-Consistent Deep Learning for OCT-Angiogram Super-Resolution]() | 
2022 | MICCAI | [CRISP - Reliable Uncertainty Estimation for Medical Image Segmentation]() | Uncertainty
2022 | MICCAI | [Efficient Bayesian Uncertainty Estimation for nnU-Net]() | Uncertainty
2022 | MICCAI | [Estimating Model Performance under Domain Shifts with Class-Specific Confidence Scores]() | 
2022 | MICCAI | [On the Uncertain Single-View Depths in Colonoscopies]() | 
2022 | [A Comprehensive Study of Modern Architectures and Regularization Approaches on CheXpert5000]()
2021 | CVPR | [Uncertainty Reduction for Model Adaptation in Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/S_Uncertainty_Reduction_for_Model_Adaptation_in_Semantic_Segmentation_CVPR_2021_paper.html) | 域自适应语义分割
2021 | CVPR | [Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2101.10979) | 域自适应语义分割
2021 | CVPR | [Uncertainty-aware Joint Salient Object and Camouflaged Object Detection](https://arxiv.org/abs/2104.02628) | 显著性检测 / 伪装目标检测
2021 | ICCV | [Self-Supervised Vessel Segmentation via Adversarial Learning](https://github.com/AISIGSJTU/SSVS) | 
2021 | ICCV [Uncertainty-Aware Pseudo Label Refinery for Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf) | 
2021 | ICCV | [Learning with Noisy Labels via Sparse Regularization](https://arxiv.org/abs/2108.00192)

