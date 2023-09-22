# 半监督/弱监督论文整理
- 任务类别: 语义分割、实例分割、视频理解、图像分类、目标检测、目标定位、域适应、度量学习、人群计数
- 近3年会议/期刊:
  - CVPR: 2023, 2022, 2021 
  - ICCV: 2023, 2021
  - ECCV: 2022
  - NIPS: [2022](https://neurips.cc/virtual/2022/papers.html?filter=titles&search=), [2021](https://neurips.cc/virtual/2021/papers.html?filter=titles&search=)
  - AAAI: [2022](https://aaai-2022.virtualchair.net/papers.html?filter=titles&search=), [2021](http://aaai.org/wp-content/uploads/2023/01/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf)
  - MICCAI: [2022](https://conferences.miccai.org/2022/papers/categories/), [2021](https://miccai2021.org/openaccess/paperlinks/categories/index.html)
  - ICLR, MIA, TPAMI, TMI, IJCAI

## Survey
**Year** |**Pub.** |**Link** |**Contribution**
:-: | :-: | :- | :-
2022 | IET | [Medical image segmentation using deep learning: A survey](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12419) | 医学图像分割综述
2023 | arXiv | [A Survey on Semi-Supervised Semantic Segmentation](https://blog.csdn.net/CV_Autobot/article/details/129234235) | 半监督语义分割综述

## 半监督
### 2023
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-
MIA | *[Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841523001408) | 提出了一类target selection的策略来改进MT模型
TMI | *[Anti-Interference From Noisy Labels: MeanTeacher-Assisted Conﬁdent Learning for Medical Image Segmentation](https://arxiv.org/abs/2106.01860) | 使用置信学习(confident learning)的方法改善MT架构中teacher的预测质量
CVPR | *[MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf) | 双流网络做半监督，labeled data的差异区域额外使用MSE loss, unlabled data使用类似co-training的方法
CVPR | [Pseudo-label Guided Contrastive Learning for Semi-supervised Medical Image Segmentation](https://paperswithcode.com/paper/pseudo-label-guided-contrastive-learning-for) | 对比学习(contrastive learning)用于半监督的setting
CVPR | *[MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery](https://arxiv.org/abs/2212.14310) | 提出了一种数据增强策略，鼓励未标注图像从标注图像的相对位置学习器官语义；并且通过立方体伪标签混合来修正原始伪标签
CVPR | [Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2305.00673) | 将Copy-Paste这种数据增强方法用于有/无标签数据之间的信息传递
CVPR | [Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection](https://arxiv.org/abs/2304.01464) | 3D点云目标检测 - 随机数据增强
CVPR | [Semi-Weakly Supervised Object Kinematic Motion Prediction](https://arxiv.org/abs/2303.17774) | 3D运动预测 - GNN
CVPR | [Semi-Supervised Stereo-based 3D Object Detection via Cross-View Consensus](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Semi-Supervised_Stereo-Based_3D_Object_Detection_via_Cross-View_Consensus_CVPR_2023_paper.pdf) | stereo-based 3D 目标检测，从cross-view的角度student-teacher约束一致性和refine 伪标签
CVPR | [Exploring Intra-Class Variation Factors with Learnable Cluster Prompts for Semi-Supervised Image Synthesis](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploring_Intra-Class_Variation_Factors_With_Learnable_Cluster_Prompts_for_Semi-Supervised_CVPR_2023_paper.pdf) | semi-supervised GAN framework
CVPR | [Semi-Supervised Hand Appearance Recovery via Structure Disentanglement and Dual Adversarial Discrimination](https://arxiv.org/abs/2303.06380)
CVPR | *[Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction Module](https://arxiv.org/abs/2303.04346) | 使用两个teacher模型来修正伪标签，监督student模型
CVPR | *[ProtoCon: Pseudo-Label Refinement via Online Clustering and Prototypical Consistency for Efficient Semi-Supervised Learning](https://arxiv.org/abs/2303.13556) | 利用最近邻居的信息来完善伪标签
CVPR | [Semi-Supervised Domain Adaptation with Source Label Adaptation](https://arxiv.org/abs/2302.02335) | 将源数据视为理想目标数据的带有噪声标签的版本, 从目标角度设计的强大清洁组件来动态清理标签噪声
CVPR | [Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.pdf) | 用一致性约束的方法解决长尾问题
CVPR | [Semi-Supervised Learning Made Simple with Self-Supervised Clustering](https://arxiv.org/abs/2306.07483) | 用自监督的方法辅助半监督框架中对无标注数据的利用
CVPR | [Hunting Sparsity: Density-guided Contrastive Learning for Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Hunting_Sparsity_Density-Guided_Contrastive_Learning_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf) | 用无监督的方法辅助半监督框架中对无标注数据的利用
CVPR | [Semi-DETR: Semi-Supervised Object Detection with Detection Transformers](https://arxiv.org/abs/2307.08095) | 基于 Transformer 的端到端半监督对象检测器
CVPR | *[Ambiguity-Resistant Semi-Supervised Learning for Dense Object Detection](https://arxiv.org/abs/2303.14960) | one-stage 目标检测模型
CVPR | *[MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-Supervised Object Detection](https://arxiv.org/abs/2303.09061)
CVPR | [SOOD: Towards Semi-Supervised Oriented Object Detection](https://arxiv.org/abs/2304.04515) | 目标检测
CVPR | [Out-of-Distributed Semantic Pruning for Robust Semi-Supervised Learning](https://arxiv.org/abs/2305.18158) | 提出了一个称为 OOD 语义剪枝（OSP）的统一框架
CVPR | [Deep Semi-Supervised Metric Learning with Mixed Label Propagation](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_Deep_Semi-Supervised_Metric_Learning_With_Mixed_Label_Propagation_CVPR_2023_paper.pdf) | 度量学习
CVPR | [RefTeacher: A Strong Baseline for Semi-Supervised Referring Expression Comprehension]() | 第一个用半监督做指称表达理解(referring expression comprehension)的框架
CVPR | [Semi-Supervised Parametric Real-World Image Harmonization](https://arxiv.org/abs/2303.00157) | 图像协调
CVPR | [Contrastive Semi-Supervised Learning for Underwater Image Restoration via Reliable Bank](https://arxiv.org/abs/2303.09101) | 水下图像恢复
CVPR | *[The Devil is in the Points: Weakly Semi-Supervised Instance Segmentation via Point-guided Mask Representation](https://arxiv.org/abs/2303.15062) | 弱半监督实例分割
CVPR | [Simultaneously Short- and Long-Term Temporal Modeling for Semi-Supervised Video Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Lao_Simultaneously_Short-_and_Long-Term_Temporal_Modeling_for_Semi-Supervised_Video_Semantic_CVPR_2023_paper.pdf) | 利用帧间的短期和长期相关性做视频语义分割
CVPR | [Fuzzy Positive Learning for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2210.08519) | 模糊正学习(FPL): 自适应地鼓励模糊正预测并抑制高概率的负预测
CVPR | [Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910) | 基于FixMatch, 特征扰动
CVPR | [SemiCVT: Semi-Supervised Convolutional Vision Transformer for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_SemiCVT_Semi-Supervised_Convolutional_Vision_Transformer_for_Semantic_Segmentation_CVPR_2023_paper.pdf) | mean teacher + transformer
CVPR | *[Conflict-based Cross-View Consistency for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2303.01276) | 基于co-training, 利用跨视图一致性
CVPR | [Augmentation Matters: A Simple-yet-Effective Approach to Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2212.04976) | 数据扰动
CVPR | [Instance-Specific and Model-Adaptive Supervision for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2211.11335) | 区分未标记的实例, 动态调整每个实例的增强
CVPR | *[Boosting Semi-Supervised Learning by Exploiting All Unlabeled Data](https://arxiv.org/abs/2303.11066) | 更好地利用所有未标记的示例
CVPR | *[HyperMatch: Noise-Tolerant Semi-Supervised Learning via Relaxed Contrastive Constraint](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_HyperMatch_Noise-Tolerant_Semi-Supervised_Learning_via_Relaxed_Contrastive_Constraint_CVPR_2023_paper.pdf) | noise-tolerant utilization of unlabeled data
CVPR | [MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins](https://arxiv.org/abs/2308.09037) | 随着训练的进行分析模型在伪标记示例上的行为，以确保低质量预测被掩盖了
CVPR | [Decoupled Semantic Prototypes Enable Learning from Diverse Annotation Types for Semi-Weakly Segmentation in Expert-Driven Domains](https://openaccess.thecvf.com/content/CVPR2023/html/Reiss_Decoupled_Semantic_Prototypes_Enable_Learning_From_Diverse_Annotation_Types_for_CVPR_2023_paper.html) | 半弱分割(多种注释类型中进行学习)
CVPR | [DualRel: Semi-Supervised Mitochondria Segmentation from a Prototype Perspective](https://openaccess.thecvf.com/content/CVPR2023/papers/Mai_DualRel_Semi-Supervised_Mitochondria_Segmentation_From_a_Prototype_Perspective_CVPR_2023_paper.pdf)
CVPR | *[PEFAT: Boosting Semi-Supervised Medical Image Classification via Pseudo-Loss Estimation and Feature Adversarial Training](https://openaccess.thecvf.com/content/CVPR2023/papers/Zeng_PEFAT_Boosting_Semi-Supervised_Medical_Image_Classification_via_Pseudo-Loss_Estimation_and_CVPR_2023_paper.pdf) |  trustworthy data selection scheme 
CVPR | [TimeBalance: Temporally-Invariant and Temporally-Distinctive Video Representations for Semi-Supervised Action Recognition](https://arxiv.org/abs/2303.16268) | 视频理解
CVPR | [SVFormer: Semi-Supervised Video Transformer for Action Recognition](https://arxiv.org/abs/2211.13222) | 动作识别 - 数据增强，Transformer
CVPR | [LaserMix for Semi-Supervised LiDAR Semantic Segmentation](https://arxiv.org/abs/2207.00026) | 点云
CVPR | [CHMATCH: Contrastive Hierarchical Matching and Robust Adaptive Threshold Boosted Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_CHMATCH_Contrastive_Hierarchical_Matching_and_Robust_Adaptive_Threshold_Boosted_Semi-Supervised_CVPR_2023_paper.pdf) |  learn robust adaptive thresholds
CVPR | [A New Comprehensive Benchmark for Semi-Supervised Video Anomaly Detection and Anticipation](https://arxiv.org/abs/2305.13611) | 半监督视频异常检测数据集 + Benchmark
CVPR | [Optimal Transport Minimization: Crowd Localization on Density Maps for Semi-Supervised Counting](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.pdf) | 人群计数
CVPR | [Semi-Supervised Video Inpainting with Cycle Consistency Constraints](https://arxiv.org/abs/2208.06807) | 视频修复
CVPR | [Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Class_Balanced_Adaptive_Pseudo_Labeling_for_Federated_Semi-Supervised_Learning_CVPR_2023_paper.pdf) | 联邦半监督学习
CVPR | [Consistent-Teacher: Towards Reducing Inconsistent Pseudo-Targets in Semi-Supervised Object Detection](https://arxiv.org/abs/2209.01589) | 目标检测

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
MICCAI | [Censor-aware Semi-supervised Learning for Survival Time Prediction from Medical Images]() | 
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
ECCV | [Open-Set Semi-Supervised Object Detection]()
ECCV | [Semi-supervised Object Detection via Virtual Category Learning]()
ECCV | [Towards Realistic Semi-Supervised Learning]()
ECCV | [Graph-constrained Contrastive Regularization for Semi-weakly Volumetric Segmentation]()
ECCV | [Semi-Supervised Monocular 3D Object Detection by Multi-View Consistency]()
ECCV | [Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection]()
ECCV | [Semi-Supervised Temporal Action Detection with Proposal-Free Masking]()
ECCV | [Unsupervised and Semi-supervised Bias Benchmarking in Face Recognition]()
ECCV | [S2-VER: Semi-Supervised Visual Emotion Recognition]()
ECCV | [Unsupervised Selective Labeling for More Effective Semi-Supervised Learning]()
ECCV | [Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation]()
ECCV | [PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection]()
ECCV | [RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning]()
ECCV | [Semi-supervised Single-view 3D Reconstruction via Prototype Shape Priors]()
ECCV | [Leveraging Action Affinity and Continuity for Semi-supervised Temporal Action Segmentation]()
ECCV | [S^2Contact: Graph-based Network for 3D Hand-Object Contact Estimation with Semi-Supervised Learning]()
ECCV | [Semi-Supervised Vision Transformers]()
ECCV | [Diverse Learner: Exploring Diverse Supervision for Semi-supervised Object Detection]()
ECCV | [Semi-supervised 3D Object Detection with Proficient Teachers]()
ECCV | [ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization]()
ECCV | [ART-SS: An Adaptive Rejection Technique for Semi-Supervised restoration for adverse weather-affected images]()
ECCV | [RVSL: Robust Vehicle Similarity Learning in Real Hazy Scenes Based on Semi-supervised Learning]()
ECCV | [CA-SSL: Class-Agnostic Semi-Supervised Learning for Detection and Segmentation]()
ECCV | [Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection]()
ECCV | [Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching]()
ECCV | [Semi-Supervised Learning of Optical Flow by Flow Supervisor]()
ECCV | [A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering]()
ECCV | [Stochastic Consensus: Enhancing Semi-Supervised Learning with Consistency of Stochastic Classifiers]()
ECCV | [DetMatch: Two Teachers are Better Than One for Joint 2D and 3D Semi-Supervised Object Detection]()
ECCV | [Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning]()
ECCV | [OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning]()
AAAI | [GALAXY: A Generative Pre-Trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection]()
AAAI | [Semi-Supervised Object Detection with Adaptive Class-Rebalancing Self-Training]()
AAAI | [Not All Parameters Should be Treated Equally: Deep Safe Semi-Supervised Learning under Class Distribution Mismatch]()
AAAI | [REMOTE: Reinforced Motion Transformation Network for Semi-Supervised 2D Pose Estimation in Videos]()
AAAI | [A Semi-Supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues]()
AAAI | [Semi-Supervised Conditional Density Estimation with Wasserstein Laplacian Regularisation]()
AAAI | [Rethinking Pseudo Labels for Semi-Supervised Object Detection]()
AAAI | [Iterative Contrast-Classify for Semi-Supervised Temporal Action Segmentation]()
AAAI | [ASM2TV: An Adaptive Semi-Supervised Multi-Task Multi-View Learning Framework for Human Activity Recognition]()
AAAI | [Dual Decoupling Training for Semi-Supervised Object Detection with Noise-Bypass Head]()
AAAI | [LaSSL: Label-Guided Self-Training for Semi-Supervised Learning]()
AAAI | [SJDL-Vehicle: Semi-Supervised Joint Defogging Learning for Foggy Vehicle Re-Identification]()
AAAI | [Semi-Supervised Learning with Multi-Head Co-Training]()
AAAI | [Tight Neural Network Verification via Semidefinite Relaxations and Linear Reformulations]()
AAAI | [The Semi-Random Likelihood of Doctrinal Paradoxes]()
AAAI | [Ensemble Semi-Supervised Entity Alignment via Cycle-Teaching]()
AAAI | [GuidedMix-Net: Semi-Supervised Semantic Segmentation by Using Labeled Images as Reference]()
AAAI | [Mitigating Reporting Bias in Semi-Supervised Temporal Commonsense Inference with Probabilistic Soft Logic]()
AAAI | [Meta Label Propagation for Few-Shot Semi-Supervised Learning on Graphs]()
AAAI | [Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation]()
AAAI | [Barely-Supervised Learning: Semi-Supervised Learning with very few Labeled Images]()
AAAI | [Contrast-Enhanced Semi-Supervised Text Classification with Few Labels]()
AAAI | [CoCoS: Enhancing Semi-Supervised Learning on Graphs with Unlabeled Data via Contrastive Context Sharing]()
NIPS | [Debiased Self-Training for Semi-Supervised Learning]()
NIPS | [Semi-supervised Vision Transformers at Scale]()
NIPS | [Label-invariant Augmentation for Semi-Supervised Graph Classification]()
NIPS | [Label-invariant Augmentation for Semi-Supervised Graph Classification]()
NIPS | [Batch-Size Independent Regret Bounds for Combinatorial Semi-Bandits with Probabilistically Triggered Arms or Independent Arms]()
NIPS | [Robust Semi-Supervised Learning when Not All Classes have Labels]()
NIPS | [DeepMed: Semiparametric Causal Mediation Analysis with Debiased Deep Learning]()
NIPS | [Learning-Augmented Algorithms for Online Linear and Semidefinite Programming]()
NIPS | *[Semi-Supervised Video Salient Object Detection Based on Uncertainty-Guided Pseudo Labels]()
NIPS | [Semi-supervised Active Linear Regression]()
NIPS | [A Characterization of Semi-Supervised Adversarially Robust PAC Learnability]()
NIPS | [Semi-infinitely Constrained Markov Decision Processes]()
NIPS | [USB: A Unified Semi-supervised Learning Benchmark for Classification]()
NIPS | [Semi-Supervised Learning with Decision Trees: Graph Laplacian Tree Alternating Optimization]()
NIPS | [Semi-Discrete Normalizing Flows through Differentiable Tessellation]()
NIPS | [An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning]()
NIPS | [SemiFL: Semi-Supervised Federated Learning for Unlabeled Clients with Alternate Training]()
NIPS | [DTG-SSOD: Dense Teacher Guidance for Semi-Supervised Object Detection]()
NIPS | [Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and A New Dataset]()
NIPS | [Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization]()
NIPS | [Forward-Backward Latent State Inference for Hidden Continuous-Time semi-Markov Chains]()
NIPS | [Parameter tuning and model selection in Optimal Transport with semi-dual Brenier formulation]()
NIPS | [Finding Optimal Arms in Non-stochastic Combinatorial Bandits with Semi-bandit Feedback and Finite Budget]()
NIPS | [Semi-Supervised Generative Models for Multiagent Trajectories]()

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
MICCAI | [3D Graph-S2Net: Shape-Aware Self-Ensembling Network for Semi-Supervised Segmentation with Bilateral Graph Convolution]() | 
MICCAI | [3D Semantic Mapping from Arthroscopy using Out-of-distribution Pose and Depth and In-distribution Segmentation Training]() | 
MICCAI | [A Deep Network for Joint Registration and Parcellation of Cortical Surfaces]() | 
MICCAI | [Anatomy-Constrained Contrastive Learning for Synthetic Segmentation without Ground-truth]() | 
MICCAI | [Annotation-efficient Cell Counting]()
MICCAI | [Cell Detection from Imperfect Annotation by Pseudo Label Selection Using P-classification]()
MICCAI | [Cell Detection in Domain Shift Problem Using Pseudo-Cell-Position Heatmap]()
MICCAI | [Conditional GAN with an Attention-based Generator and a 3D Discriminator for 3D Medical Image Generation]()
MICCAI | [Context-aware virtual adversarial training for anatomically-plausible segmenation]()
MICCAI | [Detection of critical structures in laparoscopic cholecystectomy using label relaxation and self-supervision]()
MICCAI | [Disentangled Sequential Graph Autoencoder for Preclinical Alzheimer's Disease Characterizations from ADNI Study]()
MICCAI | [Dual-Consistency Semi-Supervised Learning with Uncertainty Quantification for COVID-19 Lesion Segmentation from CT Images]()
MICCAI | [Duo-SegNet: Adversarial Dual-Views for Semi-Supervised Medical Image Segmentation]()
MICCAI | [Efficient Semi-Supervised Gross Target Volume of Nasopharyngeal Carcinoma Segmentation via Uncertainty Rectified Pyramid Consistency]()
MICCAI | [Federated Contrastive Learning for Volumetric Medical Image Segmentation]()
MICCAI | [Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching]()
MICCAI | [FedPerl: Semi-Supervised Peer Learning for Skin Lesion Classification]()
MICCAI | [Few Trust Data Guided Annotation Refinement for Upper Gastrointestinal Anatomy Recognition]()
MICCAI | [Few-Shot Domain Adaptation with Polymorphic Transformers]()
MICCAI | [Functional Magnetic Resonance Imaging data augmentation through conditional ICA]()
MICCAI | [GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference]()
MICCAI | [I-SECRET: Importance-guided fundus image enhancement via semi-supervised contrastive constraining]()
MICCAI | [Joint PVL Detection and Manual Ability Classification using Semi-Supervised Multi-task Learning]()
MICCAI | [MT-UDA: Towards Unsupervised Cross-Modality Medical Image Segmentation with Limited Source Labels]()
MICCAI | [Multimodal Representation Learning via Maximization of Local Mutual Information]()
MICCAI | [Neighbor Matching for Semi-supervised Learning]()
MICCAI | [One-Shot Medical Landmark Detection]()
MICCAI | [Order-Guided Disentangled Representation Learning for Ulcerative Colitis Classification with Limited Labels]()
MICCAI | [OXnet: Deep Omni-supervised Thoracic Disease Detection from Chest X-rays]()
MICCAI | [Partial-supervised Learning for Vessel Segmentation in Ocular Images]()
MICCAI | [Pay Attention with Focus: A Novel Learning Scheme for Classification of Whole Slide Images]()
MICCAI | [POPCORN: Progressive Pseudo-labeling with Consistency Regularization and Neighboring]()
MICCAI | [Positional Contrastive Learning for Volumetric Medical Image Segmentation]()
MICCAI | [Projection-wise Disentangling for Fair and Interpretable Representation Learning: Application to 3D Facial Shape Analysis]()
MICCAI | [Reciprocal Learning for Semi-supervised Segmentation]()
MICCAI | [Scalable joint detection and segmentation of surgical instruments with weak supervision]()
MICCAI | [Segmentation of Left Atrial MR Images via Self-supervised Semi-supervised Meta-learning]()
MICCAI | [Self-Supervised Correction Learning for Semi-Supervised Biomedical Image Segmentation]()
MICCAI | [Self-supervised visual representation learning for histopathological images]()
MICCAI | [Semi-supervised Adversarial Learning for Stain Normalisation in Histopathology Images]()
MICCAI | [Semi-supervised Cell Detection in Time-lapse Images Using Temporal Consistency]()
MICCAI | [Semi-supervised Contrastive Learning for Label-efficient Medical Image Segmentation]()
MICCAI | [Semi-Supervised Learning for Bone Mineral Density Estimation in Hip X-ray Images]()
MICCAI | [Semi-supervised Left Atrium Segmentation with Mutual Consistency Training]()
MICCAI | [Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation]()
MICCAI | [Semi-Supervised Unpaired Multi-Modal Learning for Label-Efficient Medical Image Segmentation]()
MICCAI | [Spectral Embedding Approximation and Descriptor Learning for Craniofacial Volumetric Image Correspondence]()
MICCAI | [Tripled-uncertainty Guided Mean Teacher model for Semi-supervised Medical Image Segmentation]()
MICCAI | [USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning]()
AAAI | [SHOT-VAE: Semi-Supervised Deep Generative Models with Label-Aware ELBO Approximations]()
AAAI | [Semi-Supervised Metric Learning: A Deep Resurrection]()
AAAI | [Semi-Supervised Medical Image Segmentation through Dual-Task Consistency]()
AAAI | [PTN: A Poisson Transfer Network for Semi-Supervised Few-Shot Learning]()
AAAI | [Semi-Supervised Knowledge Amalgamation for Sequence Classification]()
AAAI | [Inferring Emotion from Large-Scale Internet Voice Data: A Semi-Supervised Curriculum Augmentation Based Deep Learning Approach]()
AAAI | [Contrastive and Generative Graph Convolutional Networks for Graph-Based SemiSupervised Learning]()
AAAI | [What the Role Is vs. What Plays the Role: Semi-Supervised Event Argument Extraction via Dual Question Answering]()
AAAI | [Semi-Supervised Node Classification on Graphs: Markov Random Fields vs. Graph Neural Networks]()
AAAI | [On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning]()
AAAI | [Hierarchical Information Passing Based Noise-Tolerant Hybrid Learning for Semi-Supervised Human Parsing]()
AAAI | [Semi-Supervised Sequence Classification through Change Point Detection]()
AAAI | [SSPC-Net: Semi-Supervised Semantic 3D Point Cloud Segmentation Network]()
AAAI | [ PASSLEAF: A Pool-Based Semi-Supervised Learning Framework for Uncertain Knowledge Graph Embedding]()
AAAI | [On the Complexity of Sum-of-Products Problems over Semirings]()
AAAI | [Task Cooperation for Semi-Supervised Few-Shot Learning]()
AAAI | [Explanation Consistency Training: Facilitating Consistency-Based Semi-Supervised Learning with Interpretability]()
AAAI | [Generative Semi-Supervised Learning for Multivariate Time Series Imputation]()
AAAI | [SALNet: Semi-Supervised Few-Shot Text Classification with Attention-Based Lexicon Construction]()
AAAI | [Semi-Supervised Learning for Multi-Task Scene Understanding by Neural Graph Consensus]()
AAAI | [DeHiB: Deep Hidden Backdoor Attack on Semi-Supervised Learning via Adversarial Perturbation]()
AAAI | [Exploiting Unlabeled Data via Partial Label Assignment for Multi-Class Semi-Supervised Learning]()
AAAI | [GraphMix: Improved Training of GNNs for Semi-Supervised Learning]()
AAAI | [Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning]()
AAAI | [Class-Attentive Diffusion Network for Semi-Supervised Classification]()
AAAI | [Semi-Supervised Learning with Variational Bayesian Inference and Maximum Uncertainty Regularization]()
AAAI | [Instance Mining with Class Feature Banks for Weakly Supervised Object Detection]()
AAAI | [ACSNet: Action-Context Separation Network for Weakly Supervised Temporal Action Localization]()
AAAI | [DenserNet: Weakly Supervised Visual Localization Using Multi-Scale Feature Aggregation]()
AAAI | [Weakly Supervised Temporal Action Localization through Learning Explicit Subspaces for Action and Context]()
AAAI | [Query-Memory Re-Aggregation for Weakly-Supervised Video Object Segmentation]()
AAAI | [Diagnose Like a Pathologist: Weakly-Supervised Pathologist-Tree Network for Slide-Level Immunohistochemical Scoring]()
AAAI | [Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation]()
AAAI | [Learning by Fixing: Solving Math Word Problems with Weak Supervision]()
AAAI | [Deductive Learning for Weakly-Supervised 3D Human Pose Estimation via Uncalibrated Cameras]()
AAAI | [Effective Slot Filling via Weakly-Supervised Dual-Model Learning]()
AAAI | [A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization]()
AAAI | [Argumentation Frameworks with Strong and Weak Constraints: Semantics and Complexity]()
AAAI | [Minimizing Labeling Cost for Nuclei Instance Segmentation and Classificationwith CrossDomain Images and Weak Labels]()
AAAI | [StarNet: Towards Weakly Supervised Few-Shot Object Detection]()
AAAI | [Weakly-Supervised Hierarchical Models for Predicting Persuasive Strategies in Good-Faith Textual Requests]()
NIPS | [Topology-Imbalance Learning for Semi-Supervised Node Classification]()
NIPS | [Minimax Optimal Quantile and Semi-Adversarial Regret via Root-Logarithmic Regularizers]()
NIPS | [Universal Semi-Supervised Learning]()
NIPS | [OpenMatch: Open-Set Semi-supervised Learning with Open-set Consistency Regularization]()
NIPS | [CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation]()
NIPS | [Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning]()
NIPS | [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling]()
NIPS | [Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels]()
NIPS | [Contrastive Graph Poisson Networks: Semi-Supervised Learning with Extremely Limited Labels]()
NIPS | [Combating Noise: Semi-supervised Learning by Region Uncertainty Quantification]()
NIPS | [RETRIEVE: Coreset Selection for Efficient and Robust Semi-Supervised Learning]()
NIPS | [DP-SSL: Towards Robust Semi-supervised Learning with A Few Labeled Samples]()
NIPS | [Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose]()
NIPS | [Data driven semi-supervised learning]()
NIPS | [Overcoming the curse of dimensionality with Laplacian regularization in semi-supervised learning]()

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
CVPR | [Weakly-Supervised Single-View Image Relighting]()
CVPR | [Robust Model-based Face Reconstruction through Weakly-Supervised Outlier Segmentation]()
CVPR | [Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation]()
CVPR | [Foundation Model Drives Weakly Incremental Learning for Semantic Segmentation]()
CVPR | [Weakly Supervised Posture Mining for Fine-grained Classification]()
CVPR | [Weak-Shot Object Detection through Mutual Knowledge Transfer]()
CVPR | [BoxTeacher: Exploring High-Quality Pseudo Labels for Weakly Supervised Instance Segmentation]()
CVPR | [Iterative Proposal Refinement for Weakly-Supervised Video Grounding]()
CVPR | [Out-of-Candidate Rectification for Weakly Supervised Semantic Segmentation]()
CVPR | [CLIP is also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation]()
CVPR | [Weakly Supervised Semantic Segmentation via Adversarial Learning of Classifier and Reconstructor]()
CVPR | [Boundary-enhanced Co-Training for Weakly Supervised Semantic Segmentation]()
CVPR | [Test Time Adaptation with Regularized Loss for Weakly Supervised Salient Object Detection]()
CVPR | [Weakly Supervised Segmentation with Point Annotations for Histopathology Images via Contrast-based Variational Model]()
CVPR | [Cascade Evidential Learning for Open-World Weakly-Supervised Temporal Action Localization]()
CVPR | [Improving Weakly Supervised Temporal Action Localization by Bridging Train-Test Gap in Pseudo Labels]()
CVPR | [Two-Stream Networks for Weakly-Supervised Temporal Action Localization with Semantic-Aware Mechanisms]()
CVPR | [Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection]()
CVPR | [Token Contrast for Weakly-Supervised Semantic Segmentation]()
CVPR | [Weakly-Supervised Domain Adaptive Semantic Segmentation with Prototypical Contrastive Learning]()

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
ECCV | [Weakly Supervised Grounding for VQA in Vision-Language Transformers]()
ECCV | [Box2Mask: Weakly Supervised 3D Semantic Instance Segmentation Using Bounding Boxes]()
ECCV | [Language Matters: A Weakly Supervised Vision-Language Pre-training Approach for Scene Text Detection and Spotting]()
ECCV | [Flow graph to Video Grounding for Weakly-supervised Multi-Step Localization]()
ECCV | [184	WeLSA: Learning To Predict 6D Pose From Weakly Labeled Data Using Shape Alignment]()
ECCV | [Joint-Modal Label Denoising for Weakly-Supervised Audio-Visual Video Parsing]()
ECCV | [Weakly Supervised Object Localization through Inter-class Feature Similarity and Intra-class Appearance Consistency]()
ECCV | [Active Learning Strategies for Weakly-Supervised Object Detection]()
ECCV | [C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation]()
ECCV | [End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution]()
ECCV | [Adaptive Spatial-BCE Loss for Weakly Supervised Semantic Segmentation]()
ECCV | [Max Pooling with Vision Transformers reconciles class and shape in weakly supervised semantic segmentation]()
ECCV | [Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation]()
ECCV | [Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation]()
ECCV | [Dual-Evidential Learning for Weakly-supervised Temporal Action Localization]()
ECCV | [Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration]()
ECCV | [W2N: Switching From Weak Supervision to Noisy Supervision for Object Detection]()
ECCV | [Dual Adaptive Transformations for Weakly Supervised Point Cloud Segmentation]()
ECCV | [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds]()
ECCV | [Object Discovery via Contrastive Learning for Weakly Supervised Object Detection]()
ECCV | [Bagging Regional Classification Activation Maps for Weakly Supervised Object Localization]()
ECCV | [Weakly Supervised 3D Scene Segmentation with Region-Level Boundary Awareness and Instance Discrimination]()
ECCV | [Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions]()
AAAI | [Weakly Supervised Video Moment Localization with Contrastive Negative Sample Mining]()
AAAI | [Learning from Weakly-Labeled Web Videos via Exploring Sub-Concepts]()
AAAI | [GearNet: Stepwise Dual Learning for Weakly Supervised Domain Adaptation]()
AAAI | [LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization]()
AAAI | [Weakly-Supervised Salient Object Detection Using Point Supervison]()
AAAI | [Weakly Supervised Neural Symbolic Learning for Cognitive Tasks]()
AAAI | [Activation Modulation and Recalibration Scheme for Weakly Supervised Semantic Segmentation]()
AAAI | [Zero-Shot Audio Source Separation through Query-Based Learning from Weakly-Labeled Data]()
AAAI | [ACGNet: Action Complement Graph Network for Weakly-Supervised Temporal Action Localization]()
AAAI | [What Can We Learn Even from the Weakest? Learning Sketches for Programmatic Strategies]()
AAAI | [Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection]()
AAAI | [Convergence and Optimality of Policy Gradient Methods in Weakly Smooth Settings]()
AAAI | [Explore Inter-Contrast between Videos via Composition for Weakly Supervised Temporal Sentence Grounding]()
AAAI | [Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning over Text]()
AAAI | [Deformable Part Region Learning with Weak Supervision for Object Detection]()
AAAI | [Enhance Weakly-Supervised Aspect Detection with External Knowledge (Student Abstract)]()
AAAI | [Uncertainty Estimation via Response Scaling for Pseudo-Mask Noise Mitigation in Weakly-Supervised Semantic Segmentation]()
AAAI | [Exploring Visual Context for Weakly Supervised Person Search]()
NIPS | [Weakly Supervised Representation Learning with Sparse Perturbations]()
NIPS | [AutoWS-Bench-101: Benchmarking Automated Weak Supervision with 100 Labels]()
NIPS | [SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised Whole-Slide Image Classification]()
NIPS | [Multi-modal Grouping Network for Weakly-Supervised Audio-Visual Video Parsing]()
NIPS | [Lifting Weak Supervision To Structured Prediction]()
NIPS | [Bi-directional Weakly Supervised Knowledge Distillation for Whole Slide Image Classification]()
NIPS | [Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation]()
NIPS | [Understanding Programmatic Weak Supervision via Source-aware Influence Function]()
NIPS | [Training Subset Selection for Weak Supervision]()
NIPS | [Weak-shot Semantic Segmentation via Dual Similarity Transfer]()
NIPS | [Joint Learning of 2D-3D Weakly Supervised Semantic Segmentation]()
NIPS | [Weakly supervised causal representation learning]()
NIPS | [What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs]()
NIPS | [A Closer Look at Weakly-Supervised Audio-Visual Source Localization]()
NIPS | [[Re] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation]()
NIPS | [Expansion and Shrinkage of Localization for Weakly-Supervised Semantic Segmentation]()

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
MICCAI | [A Deep Discontinuity-Preserving Image Registration Network]()
MICCAI | [Accounting for Dependencies in Deep Learning based Multiple Instance Learning for Whole Slide Imaging]()
MICCAI | [Adapting Off-the-Shelf Source Segmenter for Target Medical Image Segmentation]()
MICCAI | [Airway Anomaly Detection by Graph Neural Network]()
MICCAI | [AMINN: Autoencoder-based Multiple Instance Neural Network Improves Outcome Prediction of Multifocal Liver Metastases]()
MICCAI | [ASC-Net: Adversarial-based Selective Network for Unsupervised Anomaly Segmentation]()
MICCAI | [Bounding Box Tightness Prior for Weakly Supervised Image Segmentation]()
MICCAI | [Cell Detection from Imperfect Annotation by Pseudo Label Selection Using P-classification]()
MICCAI | [Combining Attention-based Multiple Instance Learning and Gaussian Processes for CT Hemorrhage Detection]()
MICCAI | [Conditional GAN with an Attention-based Generator and a 3D Discriminator for 3D Medical Image Generation]()
MICCAI | [Consistent Segmentation of Longitudinal Brain MR Images with Spatio-Temporal Constrained Networks]()
MICCAI | [CPNet: Cycle Prototype Network for Weakly-supervised 3D Renal Chamber Segmentation]()
MICCAI | [Deep Simulation of Facial Appearance Changes Following Craniomaxillofacial Bony Movements in Orthognathic Surgical Planning]()
MICCAI | [DeepOPG: Improving Orthopantomogram Finding Summarization with Weak Supervision]()
MICCAI | [DT-MIL: Deformable Transformer for Multi-instance Learning on Histopathological Image]()
MICCAI | [Energy-Based Supervised Hashing for Multimorbidity Image Retrieval]()
MICCAI | [Flip Learning: Erase to Segment]()
MICCAI | [Generative Self-training for Cross-domain Unsupervised Tagged-to-Cine MRI Synthesis]()
MICCAI | [Hybrid Supervision Learning for Whole Slide Image Classification]()
MICCAI | [Implicit Neural Distance Representation for Unsupervised and Supervised Classification of Complex Anatomies]()
MICCAI | [Improving Pneumonia Localization via Cross-Attention on Medical Images and Reports]()
MICCAI | [Inter Extreme Points Geodesics for End-to-End Weakly Supervised Image Segmentation]()
MICCAI | [Label-Free Physics-Informed Image Sequence Reconstruction with Disentangled Spatial-Temporal Modeling]()
MICCAI | [Labels-set Loss Functions for Partial Supervision: Application to Fetal Brain 3D MRI Parcellation]()
MICCAI | [Learning Whole-Slide Segmentation from Inexact and Incomplete Labels using Tissue Graphs]()
MICCAI | [Learning with Noise: Mask-guided Attention Model for Weakly Supervised Nuclei Segmentation]()
MICCAI | [Leveraging Auxiliary Information from EMR for Weakly Supervised Pulmonary Nodule Detection]()
MICCAI | [Local-global Dual Perception based Deep Multiple Instance Learning for Retinal Disease Classification]()
MICCAI | [M-SEAM-NAM: Multi-instance Self-supervised Equivalent Attention Mechanism with Neighborhood Affinity Module for Double Weakly Supervised Segmentation of COVID-19]()
MICCAI | [Multi-modal Multi-instance Learning using Weakly Correlated Histopathological Images and Tabular Clinical Information]()
MICCAI | [Multimodal Representation Learning via Maximization of Local Mutual Information]()
MICCAI | [Multimodal Sensing Guidewire for C-arm Navigation with Random UV Enhanced Optical Sensors using Spatio-temporal Networks]()
MICCAI | [Multiple Instance Learning with Auxiliary Task Weighting for Multiple Myeloma Classification]()
MICCAI | [Observational Supervision for Medical Image Classification using Gaze Data]()
MICCAI | [OXnet: Deep Omni-supervised Thoracic Disease Detection from Chest X-rays]()
MICCAI | [Predicting Symptoms from Multiphasic MRI via Multi-Instance Attention Learning for Hepatocellular Carcinoma Grading]()
MICCAI | [Real-Time Rotated Convolutional Descriptor for Surgical Environments]()
MICCAI | [Superpixel-guided Iterative Learning from Noisy Labels for Medical Image Segmentation]()
MICCAI | [Trainable summarization to improve breast tomosynthesis classification]()
MICCAI | [Training Deep Networks for Prostate Cancer Diagnosis Using Coarse Histopathological Labels]()
MICCAI | [U-DuDoNet: Unpaired dual-domain network for CT metal artifact reduction]()
MICCAI | [Uncertainty-Guided Progressive GANs for Medical Image Translation]()
MICCAI | [Weakly supervised pan-cancer segmentation tool]()
MICCAI | [Weakly-Supervised Ultrasound Video Segmentation with Minimal Annotations]()
MICCAI | [Weakly-Supervised Universal Lesion Segmentation with Regional Level Set Loss]()
MICCAI | [Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks]()
AAAI | [Weakly-Supervised Temporal Action Localization by Uncertainty Modeling]()
AAAI | [Weakly Supervised Deep Hyperspherical Quantization for Image Retrieval]()
AAAI | [Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation]()
AAAI | [Weakly Supervised Semantic Segmentation for Large-Scale Point Cloud]()
AAAI | [GradingNet: Towards Providing Reliable Supervisions for Weakly Supervised Object Detection by Grading the Box Candidates]()
NIPS | [End-to-End Weak Supervision]()
NIPS | [Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing]()
NIPS | [Reducing Information Bottleneck for Weakly Supervised Semantic Segmentation]()
NIPS | [Policy Learning Using Weak Supervision]()
NIPS | [Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection]()

### Previous
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :-

<!-- ## 语义分割
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
  -->
<!-- ## Uncertainty
### 2023
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :- 

### 2022
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :- 

### 2021
**Pub.** |**Link** | **Brief Intro**
:-: | :-: | :- 
AAAI | [UAG: Uncertainty-Aware Attention Graph Neural Network for Defending Adversarial Attacks]()
AAAI | [Model Uncertainty Guides Visual Object Tracking]()
AAAI | [Uncertainty Quantification in CNN through the Bootstrap of Convex Neural Networks]()
AAAI | [Joint Demosaicking and Denoising in the Wild: The Case of Training under Ground Truth Uncertainty]() -->


### Previous
**Year** |**Pub.** |**Link** |**Contribution**
:-: | :-: | :- | :-

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
2022 | MICCAI | [A Comprehensive Study of Modern Architectures and Regularization Approaches on CheXpert5000]()
2022 | ECCV | [Vibration-based Uncertainty Estimation for Learning from Limited Supervision]()
2021 | CVPR | [Uncertainty Reduction for Model Adaptation in Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/S_Uncertainty_Reduction_for_Model_Adaptation_in_Semantic_Segmentation_CVPR_2021_paper.html) | 域自适应语义分割
2021 | CVPR | [Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2101.10979) | 域自适应语义分割
2021 | CVPR | [Uncertainty-aware Joint Salient Object and Camouflaged Object Detection](https://arxiv.org/abs/2104.02628) | 显著性检测 / 伪装目标检测
2021 | ICCV | [Self-Supervised Vessel Segmentation via Adversarial Learning](https://github.com/AISIGSJTU/SSVS) | 
2021 | ICCV | [Uncertainty-Aware Pseudo Label Refinery for Domain Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf) | 
2021 | ICCV | [Learning with Noisy Labels via Sparse Regularization](https://arxiv.org/abs/2108.00192) | 
2021 | MICCAI | [Uncertainty Aware Deep Reinforcement Learning for Anatomical Landmark Detection in Medical Images]() | 
AAAI | [A Continual Learning Framework for Uncertainty-Aware Interactive Image Segmentation]()
AAAI | [Uncertainty-Aware Multi-View Representation Learning]()
AAAI | [Multidimensional Uncertainty-Aware Evidential Neural Networks]()

