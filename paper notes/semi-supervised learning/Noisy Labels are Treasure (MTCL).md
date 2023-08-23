# Noisy Labels are Treasure: Mean-Teacher-Assisted Conﬁdent Learning for Hepatic Vessel Segmentation
为了利用噪声标签(noisy labels)，本文提出了MTCL. 在mean teacher的基础上设计了一个逐步自消噪的模块(progressively self-denoising module), 用于处理teacher的预测。处理后的结果与对应的student预测计算loss。