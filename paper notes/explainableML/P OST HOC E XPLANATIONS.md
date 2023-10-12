[POST HOC EXPLANATIONS MAY BE INEFFECTIVE FOR DETECTING UNKNOWN SPURIOUS CORRELATION](https://arxiv.org/abs/2212.04629)

本文研究了三种事后解释方法的有效性.

# Motivation
想探究的问题：事后解释方法可以检验出未知的虚假训练记号吗？

# Method & Experiment
## Experimental Design
- **Spurious Score**: 给定一个不包含spurious signal的输入。如果模型的预测结果不是spurious aligned类别，那么模型的Spurious Score就是在输入中加入spurious signal时模型将输入分配到spurious aligned类别的概率
- **Model Conditions**: 
  - spurious models: 某个预先设置的spurious signal所对应的spurious score > 0.85
  - normal models: 在所有类和预先设置的spurious signals上的spurious score < 0.1
- **Reliability performance measures**
  - Known Spurious Signal Detection Measure (K-SSD) - measures the similarity of explanations derived from spurious models on spurious inputs to the ground truth explanation (已知杂散信号时方法的可靠性)
  <!-- Explaination: 就是该可解释性方法得出的结果。比如Feature Attribution得出的是signal的排名 -->
  - Cause-for-Concern Measure (CCM) - measures the similarity of explanations derived from spurious models for normal inputs to explanations derived from normal models for normal inputs (未知信号时方法的可靠性)
  - False Alarm Measure (FAM) - measures the similarity of explanations derived from normal models for spurious inputs to explanations derived from spurious models for spurious inputs (方法对误报的敏感性)

  (?为什么Feature Attribution用SSIM, Concept Activation用KS, Training Point Ranking用ICM)
- **Blinded Study**: 设计了一项用户研究, 以评估最终用户使用事后解释来检测模型对虚假信号的依赖的能力。参与者被分为两组 (根据是否被告知potential spurious signals)，并且对他们对该模型是否能够部署的期望程度打分。

## Results
### Feature Attributions
? 应该是计算saliency map

用SSIM计算两个saliency map的相似性
### Concept Activation Importance
用Kolmogorov-Smirnoff检验来比较两个distribution的相似性
### Training Point Ranking

### Blinded Study