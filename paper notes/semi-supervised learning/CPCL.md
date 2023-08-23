# All-Around Real Label Supervision: Cyclic Prototype Consistency Learning for Semi-supervised Medical Introduction
supervised-like consistency via prototypical network

目前基于mean teacher的方法对有标签数据的利用都局限于直接对学生模型做有监督训练。

本文提出了一种新的范式，代替传统的无标签数据，利用有标签(高质量)数据来探索无标签的数据。本质上也是一种regulation consistency, 只不过抛弃了传统的purturbation (即无监督学习的方法)，采用一种更像有监督学习的方法。