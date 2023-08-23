# Anti-Interference From Noisy Labels: MeanTeacher-Assisted Conﬁdent Learning for Medical Image Segmentation
[置信学习(confident learning)](https://zhuanlan.zhihu.com/p/394985481)

本文提出了一种基于mean teacher的半监督学习框架，旨在充分利用医学图像数据集中的大量低质量标注数据。

本文所考虑的数据集是将高质量数据(Set-HQ)和低质量数据(Set-LQ)分开的, 任务是做二分类(binary segmentation). 主要贡献是提出了label self-denoising process.