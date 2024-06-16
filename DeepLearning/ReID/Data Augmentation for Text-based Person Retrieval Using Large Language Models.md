# Data Augmentation for Text-based Person Retrieval Using Large Language Models  

## 问题

LLM能生成多样的文本描述同时保持正确的句子结构，但是会出现幻觉

## 解决方案

引入**Text Faithfulness Filter (TFF)**来过滤掉不准确的文本描述

引入**Balanced Sampling Strategy (BSS)** 来平衡原始文本描述和数据增广（生成的文本）的描述

该方式既不改变原始的模型架构，也不更改损失的形式——plug-and-play  

## 方法

![image-20240530222617789](../../../../software/Typora/Typora/images/image-20240530222617789.png)

#### 1. LLM-based Data Augmentation

- 使用LLM（Vicuna）重写原描述，TFF判定

![image-20240530222631861](../../../../software/Typora/Typora/images/image-20240530222631861.png)

#### 2. Text Faithfulness Filter  

![image-20240530223448134](../../../../software/Typora/Typora/images/image-20240530223448134.png)

Sentence Transformers framework to implement semantic similarity calculation.  

计算原始和生成文本的嵌入向量的余弦相似度，![image-20240530224806515](../../../../software/Typora/Typora/images/image-20240530224806515.png)判定是否需要丢弃

#### 3. Balanced Sampling Strategy  

![image-20240531114709014](../../../../software/Typora/Typora/images/image-20240531114709014.png)

根据随机数 $r_i$ 是否大于 $\beta$ 来判断用原始文本还是增广的文本

基于CLIP，使用contrastive learning loss   

![image-20240531115037381](../../../../software/Typora/Typora/images/image-20240531115037381.png)

当这个损失函数被最小化时，它实际上在做两件事情：

- **最大化分子**：使 $V_i$ 与 $T_i^*$ 之间的得分 $s(V_i, T_i^*)$ 尽可能大。通过优化模型使这个得分最大，意味着增强模型识别和加强正确的图像-文本对应关系的能力。
- **最小化分母中其他项的影响**：分母包含了 $V_i$ 与所有文本的匹配得分，包括不应该匹配的 $T_j^*$（$j \neq i$）。通过降低这些不相关匹配的得分，损失函数鼓励模型区分不相关的图像和文本对，从而避免错误匹配。
  这样设计的结果是，对于每个图像 $V_i$，模型学习到如何与正确的文本 $T_i^*$ 关联，并抑制与其他文本的关联。

## 效果

![image-20240531120353599](../../../../software/Typora/Typora/images/image-20240531120353599.png)![image-20240531120402776](../../../../software/Typora/Typora/images/image-20240531120402776.png)

**与其他三种数据增广方法对比**

![image-20240531120516648](../../../../software/Typora/Typora/images/image-20240531120516648.png)

![image-20240531120456275](../../../../software/Typora/Typora/images/image-20240531120456275.png)