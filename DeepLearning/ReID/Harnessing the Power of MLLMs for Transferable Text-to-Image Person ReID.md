# Harnessing the Power of MLLMs for Transferable Text-to-Image Person ReID  

**address two key challenges in utilizing the obtained textual descriptions  **

## 摘要

##### 1. 问题

- Multi-modal Large Language Models (MLLMs)  会产生相似结构的文本描述，导致模型过拟合一种文本描述的模式
- Multi-modal Large Language Models (MLLMs)  可能会产生错误的描述，引入一种方法自动识别与图片内容不相干的单词——基于相似度计算 + mask

##### **2. 效果**

- 显著提升了模型的迁移效果（text-to-image ReID）在传统评估场景SOTA

## Introduction

- cross-dataset generalization ability of their approaches is significantly low [41], limiting real-world applications.   
- collecting training data for each target domain is infeasible  

使用LUPerson数据集作为图片来源，用MLLMs生成文本描述，但是生成过程需要注意两个问题：

##### **对一个图片产生多样性的描述——防止模型过拟合某种特定描述范式**

**Templatebased Diversity Enhancement (TDE)** method that instructs MLLMs to conduct image captioning according to given description templates.   

与chatgpt对话生成多种模板，作为指令输入MLLMs

##### 减少描述的噪声——少数生成的单词与图像并不匹配

propose a novel **Noise-aware Masking (NAM)** method  

通过计算text token and all image tokens in the paired image for a specific textual description的相似度，基于相似度mask掉不相关的那些单词（作为输入，**在下一轮**）

## 方法

##### 1. Generating Diverse Descriptions

**指令设计**

LUPerson database as the image source  

designing an effective instruction ：用ChatGPT 生成指令（**static instruction**）

![image-20240530145912361](../../../../software/Typora/Typora/images/image-20240530145912361.png)

**多样化增强**

Template-based Diversity Enhancement (TDE)   

1. 使用两个MLLMs为每个图片生成两个描述
2. 把这些描述给Chatgpt捕获描述范式，并基dynamic instruction  于此生成新的范式
3. 共生成46个范式

**dynamic instruction**

**数据集**

Qwen [3] and Shikra [8]两个MLLMs生成LUPerson-MLLM dataset  

- 10 million数据
- 每图像4描述
- 前二：静态指令
- 后二：动态指令

##### 2. Noise-Aware Masking

——identifies noisy text tokens and fully uses the matched text tokens for model training.  

通过计算文本和图像token的余弦相似度，计算单词是噪声的水准：![image-20240530154117312](../../../../software/Typora/Typora/images/image-20240530154117312.png)

但是不是光光基于 $r_i$ 因为在开始训练阶段，这个值可能会比较大，导致很多单词都会被mask，因此

![image-20240530154719345](../../../../software/Typora/Typora/images/image-20240530154719345.png)

p is the average masking ratio, computes $r′$​ for the next training epoch, which requires only one forward pass for each iteration. 

##### 3. Optimization  

SDM loss

## 结果

![image-20240530213956203](../../../../software/Typora/Typora/images/image-20240530213956203.png)![image-20240530214212683](../../../../software/Typora/Typora/images/image-20240530214212683.png)

