# Learning Semantic Relationship among Instances for Image-Text Matching  

## [2023 CVPR]

## Introduction

**以往问题**

<img src="../../../../software/Typora/Typora/images/image-20240612104501641.png" alt="image-20240612104501641" style="zoom:50%;" />

现有方法如图（a），提取图像和文本的局部特征，然后进行整合计算最难负样本的三元组铰链损失（在一个mini-batch），但是这种方法有**两种问题**：

- 不能学习到样本之间的**细微语义差异**，如图b
- 因为不同的样本可能有**相似的行为或者主题**，以往的方法不能从样本之间学习到共享的知识，从而不能有效地学习这些罕见的具有语义稀缺性的样本，如图c

本文提出的Hierarchical RElation Modeling framework (HREM)  能同时捕获碎片级和实例级的关系

**主要贡献：**

1. 第一个提出了能同时捕获模态内的碎片级别的关系和模态间的实例级别的关系的框架：HREM
2. **通过链接关系和学习相关性关系提出cross embedding association graph**  
3. 提出了一个两种关系交互机制来学习关系增强嵌入

**效果**：

Flickr30K and MS-COCO, by 4%-10% rSum

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">1</span></p>

---

## The Proposed Method

![image-20240613154323206](../../../../software/Typora/Typora/images/image-20240613154323206.png)  

#### 1. Feature Extraction 

Visual Representation  

用Faster R-CNN提取显著区域，用预训练的ResNet-101提取局部特征 $R = \{r_1, \cdots, r_{n_r}\} \in \mathbb{R}^{n_r \times d}$

Textual Representation  

(BiGRU) , or pre-trained BERT提取文本局部特征 $C = \{\mathbf{c}_1, \cdots, \mathbf{c}_{n_c}\} \in\mathbb{R}^{n_c \times d}$

#### 2. Fragment-level Relation Modeling

**Visual Regions**  

通过图注意力网络，在一张图片的局部特征之间构建语义关系图，**节点是局部特征，边是他们之间的关系**（基于**缩放点积**的注意力机制），通过图中的注意力权重表示语义关系。

图中原始节点的（局部的）特征：$R = \{r_1, \cdots, r_{n_r}\} \in \mathbb{R}^{n_r \times d}$

原始特征经过自注意力层：$R^V = \{r_1^V, \cdots, r_{n_r}^V\} \in \mathbb{R}^{n_r \times d}$

最终的**全局特征**：$\mathbf{v} = \beta \cdot MaxPool(R) + (1 - \beta) \cdot AvgPool(R^V)$

**Textual Words**  

原始节点的（单词级别）特征：$C = \{c_1, \ldots, c_{n_c}\} \in \mathbb{R}^{n_c \times d}$

原始特征经过自注意力层：$C^U = \{C_1^U, \ldots, C_{n_c}^U\} \in \mathbb{R}^{n_c \times d}$

最终的**全局特征**：$u = \beta \cdot \text{MaxPool}(C) + (1 - \beta) \cdot \text{AvgPool}(C^U)$​

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">2</span></p>

---

#### 3. Instance-level Relation Modeling

继续**通过图**尝试在image-text pairs之间建立关系

##### 3.1 Cross-Embedding Association Graph  

划分为两种关系：**connection and relevance**  

**connection关系的构建**

节点：$V = \{v_1, \ldots, v_N, u_1, \ldots, u_N\} \in \mathbb{R}^{2N \times d},$其中 $v$ 和 $u$ 分别表示图像文本特征（实例级别的），相同数字为一对。

- 使用$A \in \mathbb{R}^{2N \times 2N}$来表示connection关系（节点之间是否存在关联边）

- 使用 $S \in \mathbb{R}^{2N \times 2N}$​表示relevance关系（节点之间的语义关联程度）

- 又把上述两个矩阵A和S分为了**two patterns and four blocks**  

| patterns   | blocks（N×N）  |
| ---------- | -------------- |
| 同一模态内 | Image-to-Image |
|            | Text-to-Text   |
| 不同模态间 | Image-to-Text  |
|            | Text-to-Image  |

$$
A = \begin{bmatrix}
A_{I \to I} & A_{I \to T} \\
A_{T \to I} & A_{T \to T}
\end{bmatrix}, \quad S = \begin{bmatrix}
S_{I \to I} & S_{I \to T} \\
S_{T \to I} & S_{T \to T}
\end{bmatrix},
$$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">3</span></p>

---

**处理A：识别有效连接，过滤掉不相关的语义**
$$
(A_{I \to I})_{ij} = 
\begin{cases} 
1 & \text{if } v_j \in N_{\text{intra}}(v_i) \\
0 & \text{else}
\end{cases},
$$

$$
(A_{T \to T})_{ij} = 
\begin{cases} 
1 & \text{if } u_j \in N_{\text{intra}}(u_i) \\
0 & \text{else}
\end{cases},
$$

基于embedding nodes之间的距离来衡量节点间是否应该有边，其中$N_{\text{intra}}$是以一个模态为基准，基于embedding相似度排序得到的最相似的部分的集合。

然而作者认为仅仅使用全局特征并不足以表示模态间的关系，因此添加**局部特征的关系**：

<img src="../../../../software/Typora/Typora/images/image-20240612221041626.png" alt="image-20240612221041626" style="zoom:67%;" />

首先基于局部特征计算相似度矩阵，而**后根据相似度矩阵的行/列求得所有最大值的和取均值**:
$$
p_{I \rightarrow T} = \frac{1}{n_r} \sum_{m=1}^{n_r} \max_{n \in [1, n_c]} (r^T_m c_n) 
\\
p_{T \rightarrow I} = \frac{1}{n_c} \sum_{m=1}^{n_c} \max_{n \in [1, n_r]} (c^T_m r_n)
$$
然后同样像之前同一模态内的那样计算与该图像文本对相似度最接近的样本对集合 $N_{\text{inter}}$
$$
(A_{I \to T})_{ij} = 
\begin{cases} 
1 & \text{if } u_j \in N_{\text{inter}}(v_i) \\
0 & \text{else}
\end{cases},
$$

$$
(A_{T \to I})_{ij} = 
\begin{cases} 
1 & \text{if } v_j \in N_{\text{inter}}(u_i) \\
0 & \text{else}
\end{cases}
$$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">4</span></p>

---

**relevance关系的构建**

基于局部特征得到的行和列的最大匹配的部分，将它们合并：

<img src="../../../../software/Typora/Typora/images/image-20240612225227755.png" alt="image-20240612225227755" style="zoom: 50%;" />

然后从中取出top-k(基于相似度大小)
$$
q_{I \rightarrow T} = \text{TopK}\left(\left\{\max_{n \in [1, n_c]} (r^m C_n)_{m=1}^{n_r}\right\}\right)
\\
q_{T \rightarrow I} = \text{TopK}\left(\left\{\max_{n \in [1, n_r]} (C^m r_n)_{m=1}^{n_c}\right\}\right)
$$
然后基于多层感知机学习模态间的关联，得到的是一个标量：
$$
s_{I \rightarrow T} = \text{MLP}(q_{I \rightarrow T}) + p_{I \rightarrow T}\\

s_{T \rightarrow I} = \text{MLP}(q_{T \rightarrow I}) + p_{T \rightarrow I}
$$
用上式来表示图像到文本和文本到图像**跨模态**的relevance大小。

---

而对于**同一模态**的relevance大小，基于全局特征计算：
$$
(S_{T \rightarrow T})_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}}, \quad (S_{I \rightarrow I})_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}} \tag{10}
$$
公式(10)中，通过计算欧氏距离并通过一个指数函数转换成（0，1] 的相关性值，作为同一模态内的relevance值。

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">5</span></p>

---

##### **3.2 Relation Interaction Mechanisms**  

**Fusion Mechanism**  

![image-20240613152133952](../../../../software/Typora/Typora/images/image-20240613152133952.png)

$connection \space A$ 矩阵就是用来控制哪些特征要作为输入（非0部分输入，0部分舍弃）

$relevance \space S$ 矩阵作为额外的注意力权重，其中 $\lambda$ 用来控制 $S$ 和原始注意力权重的比例 
$$
\text{Att}(QKV; A, S) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \lambda S\right) V
$$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">6</span></p>

---

**Standalone Mechanism**  

<img src="../../../../software/Typora/Typora/images/image-20240613153838049.png" alt="image-20240613153838049" style="zoom:67%;" />

首先原始的图像文本特征警告过多头交叉注意力（QKV来自不同模态），然后经过同一模态自身的自注意力得到输出。而connection矩阵A和relevance矩阵S里面四个Block**按照图中红色箭头**和如下公式添加到注意力模块中：
$$
\text{Att}(QKV; A, S) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \lambda S\right) V
$$
最终得到的特征：$\{\overline{v_1}, \ldots, \overline{v_N}\} \text{ and } \{\overline{u_1}, \ldots, \overline{u_N}\}.$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">7</span></p>

---

## Optimization  

**Neighbor Batch Sampling**  

并没有用随机采样，而是基于k-means算法**根据图像特征先**进行聚类，然后从挑选P个簇，并从每个簇中挑选K张图片，构成一个batch（$N = P × K  $），然后再根据这些图像的batch获得对应的文本描述。

**Objective Function  **

- **损失一：**使用三元组损失（使用distance-weighted sampling采样最难负样本）

$$
\mathcal{L} = [\alpha - s(v, u) + s(v, u')]_+ + [\alpha - s(v', u) + s(v', u')]_+
$$

**根据原始图像文本特征，以及经过关系增强的图像文本特征**计算三元组损失并相加。
$$
\mathcal{L}_{\text{cross}} = \mathcal{L}(\overline{v}, \overline{u}) + [\mathcal{L}(v, u) + \mathcal{L}(\overline{v}, u) + \mathcal{L}(v, \overline{u})]
$$
原因：<font color='red'>since we need to encode embeddings directly without sample interaction at the inference stage.  </font>

- **损失二：**相关性正则化损失

$$
\mathcal{L}_{\text{reg}} = L_{\text{kl}} (S_{I \rightarrow T}, S^g_{I \rightarrow T}) + L_{\text{kl}} (S_{T \rightarrow I}, S^g_{T \rightarrow I})
$$

添加这个损失的目的是为了**确保在训练过程中学习到的模态间相关性矩阵S不会崩溃**。其中：
$$
(S_{T \rightarrow T})_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}}, \quad (S_{I \rightarrow I})_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}} \\


(S_{I \rightarrow T})^g_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}}, \quad (S_{T \rightarrow I})^g_{ij} = e^{-\frac{\|u_i - v_j\|^2}{\sigma}}
$$
同时通过使用KL散度，作者期望**学习到的模态间相关性与全局特征的语义相关性比较接近**。

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">8</span></p>

---

## Discussion  

1. 推理阶段：因为在实际应用过程并没有batch data（**可以用来聚类？**），但由于采用端到端的框架，因此编码网络会在训练过程中受益
2. 时间复杂度：时间复杂度提升了，但是精度变高了

## 效果

![image-20240613183520296](../../../../software/Typora/Typora/images/image-20240613183520296.png)