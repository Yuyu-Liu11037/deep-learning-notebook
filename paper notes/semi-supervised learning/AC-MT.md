# Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation

现有的基于mean teacher的模型很少有关注目标选择 (target selection), 即Consistency regulation应该把哪些体素 (voxel) 考虑到。因此作者假设那些分类波动性比较大的区域，即模糊区域 (ambiguous regions)，应该被更多关注 (文中的consistency loss只计算这些区域)。

作者挑选模糊区域的方法是几个即插即用的策略，分别从熵、模型不确定性和噪声标签自识别 (entropy, model uncertainty and label noise self-identification) 的角度考虑, 不增加训练参数，也不改变模型架构。之后，将估计的模糊图纳入一致性损失，以鼓励两个模型在这些区域的预测达成共识。