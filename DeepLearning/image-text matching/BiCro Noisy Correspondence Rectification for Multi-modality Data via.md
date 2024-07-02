# BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency  

### 主要问题：noisy correspondence problem   

以往使用的三元组损失：

![image-20240616222522813](../../../../software/Typora/Typora/images/image-20240616222522813.png)

但是能使用这种损失的前提是所有的图像文本两种模态间是完美对齐的，但是由于很多标注都有噪声，这会导致使用该损失会过拟合有噪声的数据，把负样本拉近，导致模型效果差。

---

### 解决方案

给训练数据集的图文相似度打分，根据相关性得分计算三元组损失

![image-20240616225422033](../../../../software/Typora/Typora/images/image-20240616225422033.png)

其中 $\alpha$ 与图像文本相关度得分 $\hat{y}$ 有关。

难点就是如何计算 $\hat{y}$​ 

### 计算 $\hat{y}$ 

分为两步：

- 首先找出干净的样本
- 然后根据干净的样本推断有噪声的样本对的相关度

**原理**

**The memorization effect of deep neural networks [56] reveals that DNNs would first memorize training data of clean labels then those of noisy labels.**  

深度学习网络首先会拟合干净的数据然后拟合噪声数据，也就是说在早期训练过程中**噪声数据在训练的时候会有更大的loss**

【56】Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107–115, 2021.  

![image-20240619155429466](../../../../software/Typora/Typora/images/image-20240619155429466.png)

![image-20240619154518315](../../../../software/Typora/Typora/images/image-20240619154518315.png)