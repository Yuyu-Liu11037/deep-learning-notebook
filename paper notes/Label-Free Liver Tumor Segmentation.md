[Label-Free Liver Tumor Segmentation](https://arxiv.org/abs/2303.14869)

---
# Motivation, Challenge, Insight & Solution
![Fig](../images/LabelFreeFig1.png "Contribution")
  
训练一个用于医学图像分割的AI model常常依赖于标注良好的数据集。然而，标注医学图像不仅耗时耗力，还需要很强的专业知识。虽然目前已经有一些生成合成肿瘤(generating synthetic tumors)的尝试，但这些方法生成的图像和真实图像相差较大，且用它们来训练的模型在测试数据上的表现也不好。

为了生成可以替代真实肿瘤图片的合成图片，作者关注了合成肿瘤的四个重要特征：shape, intensity, size, location, and texture，并基于以上特征设计了一个合成方法。该方法主要有以下优点：
 - 合成的图像通过了视觉图灵测试
 - 使用合成图像训练的网络，比使用真实图像训练的网络，在测试时表现更好


---
# Details
## Method: tumor generation
![Fig](../images/LabelFreeFig2.png "Method")

将健康的肝脏的CT图转化成包含肿瘤的CT图的过程如下：
0. 用pre-trained [nnUNet](https://zhuanlan.zhihu.com/p/100014604)得到粗略的肝脏分割掩码图
1. **Location selection**: 选择一个不包含任何血管的位置
    用voxel value thresholding(a method of [Digital image processing](https://en.wikipedia.org/wiki/Digital_image_processing#:~:text=Digital%20image%20processing%20is%20the,advantages%20over%20analog%20image%20processing.))分割血管位置. 
   
    Segmented vessel mask公式为: $$v(x,y,z)=\begin{cases}1,\;f'(x,y,z)>T,\;l(x,y,z)=1\\0,\;otherwise\end{cases}$$其中$f'(x,y,z)=f(x,y,z)\otimes g(x,y,z;\sigma_a)$ 是smoothed CT scan, $g(x,y,z;\sigma_a)$ 是Gaussian filter with standard deviation $\sigma_a$, $\otimes$ 
    是standard image filtering operation. [Threshold] $T$被设置为比mean Hounsfield Unit(UT) *(UT: 用于衡量组织或物质在CT图像中的相对密度)* 略大的值: $T=\overline{f(x,y,z)\odot l(x,y,z)}+b$, 其中 $l(x,y,z)$ 是liver mask(background=0, liver=1), $\odot$ 是point-wise multiplication, $b$ 是超参数.

    有了vessel mask, 当我们随机地选取肝脏区域内的一个点 $(X,Y,Z)$时，就可以检测在tumor radius $r$ 范围内是否有血管. 如果有，则再随机选取一个点进行检测.
2. **Texture generation**: 生成近似于真实组织的纹理

    真实肝脏和肿瘤的纹理服从Gaussian distributions. 因此，首先基于已定义的HU强度$\mu_t$和标准差$\sigma_p$生成3D [Gaussian Noise](https://ai.plainenglish.io/what-is-gaussian-noise-in-deep-learning-how-and-why-it-is-used-af3730449e3a) 作为不包含血管的组织的纹理 $T(x,y,z)\sim \mathcal{N}(\mu_t,\sigma_p)$, 并且soften the texture, soften之后的纹理记为 $T'(x,y,z)$ . 最后blur the texture with Gaussian filter $g(z,y,z;\sigma_b)$: $$T''(x,y,z) = T'(x,y,z)\otimes g(z,y,z;\sigma_b)$$其中$\sigma_b$是标准差.
3. **Shape generation**: 生成椭圆形状的tumor mask
    
    该mask位于 $(x_t,y_t,z_t)$, $x,y,z$方向的半轴长分别通过对均匀分布 $U(0.75r, 1.25r)$ 随机采样得到. 然后对该mask进行弹性形变以丰富diversity，最后对mask应用一个Gaussian filter进行blur，得到 $t''(x,y,z)$.
4. **Post-processing**: 将tumor $t''(x,y,z)$, scanning volumn $f(x,y,z)$和 liver mask$l(x,y,z)$进行合成，得到:
   1. new scanning volumn $f'(x,y,z)=(1-t''(x,y,z))\odot f(x,y,z)+t''(x,y,z)\odot T''(x,y,z)$
   2. new mask with tumor(background=0, liver=1, tumor=2) $l'(x,y,z)=l(x,y,z)+t''(x,y,z)$
   
   合成完之后，再根据[mass effect](https://link.springer.com/10.1007/978-0-387-79948-3_253#:~:text=Mass%20effect%20is%20a%20phenomenon,within%20the%20restricted%20skull%20space.)(the expanding tumor pushes its surrounding tissue apart)和capsule appearance进一步调整.


---
# Implementation & Verification
**Dataset**: [LiTS](https://paperswithcode.com/dataset/lits17)

**Evaluation Metrics**: 
 - 肿瘤分割: Dice similarity coefﬁcient (DSC) and Normalized Surface Dice (NSD) with 2mm tolerance
 - 肿瘤检测: Sensitivity and Speciﬁcity 
        
    (敏感性衡量了肿瘤检测算法在正确识别出真正存在的肿瘤（真阳性）方面的能力。它是指在所有实际存在的肿瘤中，算法能够正确检测出的比例。敏感性越高，表示算法能够更好地检测出真实的肿瘤，减少漏诊率; 特异性衡量了肿瘤检测算法在正确排除无肿瘤区域（真阴性）方面的能力。它是指在所有无肿瘤区域中，算法能够正确排除的比例。特异性越高，表示算法能够更好地排除无肿瘤区域，减少误诊率)

**Implementation**: Based on the [MONAI](https://monai.io/) framework for both U-Net and [Swin UNETR](https://arxiv.org/abs/2201.01266)

## Results
1. **Clinical Validation using Visual Turing Test**
![Fig](../images/LabelFreeFig3.png "Turing Test")
2. **Comparison with State-of-the-art Methods**
![Fig](../images/LabelFreeFig4.png "SOTA comparison")
3. **Generalization to Different Models and Data**
![Fig](../images/LabelFreeFig5.png "Generalization") 
4. **Potential in Small Tumor Detection**
![Fig](../images/LabelFreeFig6.png "Small") 
5. **Controllable Robustness Benchmark**
![Fig](../images/LabelFreeFig7.png "Robustness") 
医学成像的标准评估仅限于确定人工智能在检测肿瘤方面的有效性, 这是因为现有测试数据集中注释的肿瘤数量不够大，不能代表真实器官中发生的肿瘤，特别是只包含有限的非常小的肿瘤。合成肿瘤可以作为一个可获得的、全面的来源，严格评估人工智能在检测各种不同大小和位置的器官中的肿瘤的性能。
6. **Ablation Study on Shape Generation**
![Fig](../images/LabelFreeFig8.png "Ablation")  