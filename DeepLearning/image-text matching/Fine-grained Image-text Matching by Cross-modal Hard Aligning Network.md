# Fine-grained Image-text Matching by Cross-modal Hard Aligning Network   

## [2023 CVPR]

## Introduction  

<img src="../../../../software/Typora/Typora/images/image-20240601121247283.png" alt="image-20240601121247283" style="zoom: 25%;" />

| Global Embedding Method                                      | Fragments Embedding Method                                   | Fragments Aligning Method                                    | CHAN Method                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 整体图片与句子                                               | 关注核心，物体与关键词，运用权重，自注意力机制               | 聚合局部的相似度来判断，而不是通过局部表示计算相似度         | 只保留最相关的部分                                           |
| ![image-20240601110937782](../../../../software/Typora/Typora/images/image-20240601110937782.png) | ![image-20240601110952716](../../../../software/Typora/Typora/images/image-20240601110952716.png) | ![image-20240601111009209](../../../../software/Typora/Typora/images/image-20240601111009209.png) | ![image-20240601111021959](../../../../software/Typora/Typora/images/image-20240601111021959.png) |
| 包含了背景噪声，影响结果                                     | 仅关注同一模态的关键信息并未学到模态间的语义一致             | 这种方式不仅仅匹配单词最相关的局部图像信息，也指向一些其他不相关的区域，导致一定程度的语义和图像区域不一致，影响效果。同时时间空间开销大。 | 摒弃冗余的alignment，效率更高                                |

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">1</span></p>

---

**动机：**

1. 冗余的对齐对检索精度有害
2. 存储交叉注意力的权重耗费大量时间与空间

**贡献：**

1. 提出了一个编码框架，详细阐述了交叉注意机制的对齐过程
2. 提出了hard assignment coding scheme 命名为： Cross-modal Hard Aligning Network (CHAN)  
3. 在Flickr30K  and MS-COCO  达到SOTA

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">2</span></p>

---

## Cross-modal Hard Aligning Network

### 3.1 Coding Framework for Fragment Alignment

**句子$T$与图像$V$之间的语义相关性可以通过编码和池化两个过程表示。**

1. **编码**

文本特征
$$
T = \{ t_i \mid i \in [1, \ldots, L], \, t_i \in \mathbb{R}^d \}
$$
$t_i$表示每个单词的特征

图像特征
$$
\mathbf{V} = \{ v_j \mid j \in [1, \ldots, K], \, v_j \in \mathbb{R}^d \}
$$

$v_i$​表示图像每个显眼的区域的特征

单词和图像的相似度被转化为**单词和局部图像特征加权的相似度**：
$$
s(t_i, \mathbf{V}) = \mathcal{S}(t_i, \hat{t}_i) \\
$$
$\mathcal{S}$ 为余弦相似度，上式中的$\hat{t}_i$其实就是表示单词 $t_i$ 对应的 $v_j$ （$V = \{v_j\}_{j=1}^K$​）的加权相似度的和：
$$
\hat{t}_i = \sum_{j=1}^{K} \omega_{ij} v_j
$$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">3</span></p>

---

2. **池化**

而最终图像与文本的相似度可以通过一个池化操作得到
$$
s(\mathcal{T}, \mathcal{V}) = \frac{1}{\lambda} \log \sum_{i=1}^{L} \exp(\lambda s(t_i, \mathcal{V}))
$$
LogSumExp pooling (LSE-Pooling) ，其中 $\lambda$​ 为缩放因子

---

在公式 $\hat{t}_i = \sum_{j=1}^{K} \omega_{ij} v_j$ 中 $\omega_{ij}$ 权重因子是与**使用高斯核函数的 $s_{ij}$ 相关联的**（这建立在单词和局部图像之间的相似度可以用正态分布表示的基础上）：
$$
\omega_{ij} = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(\frac{s_{ij}}{2\sigma^2}\right)
$$
其中，$s_{ij}$ 表示相似度，$\sigma$ 决定了核的大小，在归一化后，表示为：
$$
\omega_{ij} = \frac{\exp(s_{ij}/\tau)}{\sum_{j=1}^{K} \exp(s_{ij}/\tau)}
$$
其中，分母 $\sum_{j=1}^{K} \exp(s_{ij}/\tau)$ 是归一化因子，$\tau$​ 是一个平滑参数。

---

在这篇文章中，作者认为这种软分配编码（soft assignment coding）并不适用于跨模态检索任务，**因为总会有一个codeword（其实就是 $v_{j}$​）能够恰好匹配到描述中的一个单词。**

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">4</span></p>

---

### 3.2. Hard Assignment Coding

本文的思想就是如果一个句子描述一张图片，那么每一个单词肯定在表达一个明确的图像中的区域：
$$
\omega_{ij} = 
\begin{cases} 
1 & \text{if } j = \arg\max\limits_{j'=1 \ldots K} (s_{ij'}) \\ 
0 & \text{otherwise} 
\end{cases}
$$
这种方式能够有效降低时间空间复杂度。

**有效性解释——概率**
$$
P(t_i, \mathbf{v}) = 1 - \prod_{j=1}^R (1 - P(t_i, v_j))

\geq 1 - (1 - P(t_i, v_k)) = P(t_i, v_k)
$$
That is, the semantic consistency between query $t_i$ and its most relevant codeword $v_k$​ is a **lower bound** of the probability of the presence of a word in an image.  

**就是说一个单词和图像语义一致的概率下限是这个单词和其对应的局部图像特征语义一致的概率**

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">5</span></p>

---

**效率解释**

visual feature set $\mathcal{V} \in \mathbb{R}^{B_1 \times K \times d}$ 

text feature set $\mathcal{T} \in \mathbb{R}^{B_2 \times L \times d}$

其中$B_1$ and $B_2$表示图像和图像描述的数量

hard assignment coding和soft assignment coding都需要计算 assignment matrix $A \in \mathbb{R}^{B_1 \times B_2 \times K \times L}$, 造成同样的时间复杂度： $\mathcal{O}(B_1 B_2 K L d)$. 

但是hard assignment coding **has a linearly better efficiency** compared to soft assignment coding under the condition of infinite memory, as it **no longer needs to** calculate the attended version of the text feature set $\hat{\mathcal{T}} \in \mathbb{R}^{B_1 \times B_2 \times L \times d}$​, as shown in:

<img src="../../../../software/Typora/Typora/images/image-20240611195824311.png" alt="image-20240611195824311" style="zoom:67%;" /> 

Furthermore, due to the fact that $K \ll d$, the spatial complexity of hard assignment coding ($\mathcal{O}(B_1 B_2 K L)$) is significantly lower than that of soft assignment coding ($\mathcal{O}(B_1 B_2 L d)$​), which inherently suffers from the issue of **high memory consumption**. This makes hard assignment coding much more efficient than soft assignment coding without the need for iterations.

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">6</span></p>

---

### 3.3. Cross-modal Hard Alignment Network

<img src="../../../../software/Typora/Typora/images/image-20240611200641221.png" alt="image-20240611200641221" style="zoom:67%;" />

四个模块：

- Visual representation 

基于预训练的Faster R-CNN提取Top-K的区域特征，

- Text representation  

BiGRU（tokenized to several words  ）  或者 pre-trained Bert  （word-level vectors）

- Hard assignment coding  

首先通过L2范式归一化，然后基于局部特征计算余弦相似度，在**行层面max pooling**，之后用LSE-pooling 整合
$$
s(T, V) = \frac{1}{\lambda} \log \sum_{i=1}^L \exp(\lambda \max_{j=1 \ldots K} S)
$$

- Objective function  

**hinge-based bi-direction triplet ranking loss** with online **hard negative mining** proposed by VSE++ [11]
$$
\mathcal{L} = \sum_{(T,V) \sim D} \left[ \alpha + s(T, \hat{V}) - s(T, V) \right]_+ + \left[ \alpha + s(\tilde{T}, V) - s(T, V) \right]_+
$$

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">7</span></p>

---

## 效果

### COCO

![image-20240611222738843](../../../../software/Typora/Typora/images/image-20240611222738843.png)

![image-20240611222751393](../../../../software/Typora/Typora/images/image-20240611222751393.png)

### Flickr 30K

<img src="../../../../software/Typora/Typora/images/image-20240611222800871.png" alt="image-20240611222800871" style="zoom:67%;" />![image-20240611222805336](../../../../software/Typora/Typora/images/image-20240611222805336.png)

---

<p style="text-align: center;"><span style="font-size: 30px; color: red;">8</span></p>

---

### Effects of the Size of Codebook K  

K越大越好，与下面文献中说K=36效果最好不一致

【Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In ECCV, pages 201–216, 2018】

<img src="../../../../software/Typora/Typora/images/image-20240611223338585.png" alt="image-20240611223338585" style="zoom:50%;" />

作者的解释为：因为硬分配编码可以挖掘最多信息区域并保留最多共享语义的属性， 因此在K较大时表现更好。

