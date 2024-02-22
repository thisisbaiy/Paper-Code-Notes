# Deep Learning for Person Re-identification: A Survey and Outlook  

> 综述论文：https://arxiv.org/pdf/2001.04193

## 1. 摘要

we categorize it into the closed-world and open-world settings.  

- closed-world：学术环境下
- open-world ：实际应用场景下

## 2. 引言

引言部分主要讨论了跨非重叠摄像头的行人重识别（Re-ID）问题，强调其在智能监控系统中的重要性和挑战。作者提到Re-ID面临的挑战，如**视角变化、低分辨率、光照变化**等，并指出早期研究主要集中在手工特征构建和距离度量学习上。随着深度学习的发展，虽然在一些标准数据集上取得了显著进展，但实际应用与研究场景之间仍存在较大差距。此外，作者提出了一个新的基线方法AGW和一个新的评估指标mINP，旨在推动未来的Re-ID研究，并讨论了一些未来的研究方向，以期缩小封闭世界和开放世界应用之间的差距。

### 2.1 构建一个ReID系统需要的五个步骤

![image-20240218160059146](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240218160059146.png)

1. 原始数据采集
2. 生成边界框：框出其中的行人，借助算法：**person detection** or **tracking algorithms**
3. 对训练数据进行标注：标注这些**个体在不同摄像头下的相同身份**。这意味着，对于给定的个体，需要在不同摄像头捕获的图像中识别出该个体，并为其分配相同的标识符。
4. 训练模型（核心）：
   - feature representation learning  
   - distance metric learning  
   - their combinations  
5. 检索：给定一个疑犯(查询对象)和一个图库集，我们使用上一阶段学习的Re-ID模型提取特征表示。通过对计算的查询到库的相似性进行排序，获得检索到的排名列表。（Some methods have also investigated the ranking optimization to improve the retrieval performance）

### 2.2 学术环境与实际应用场景对比

![image-20240218160236511](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240218160236511.png)

|                    | **学术环境**                                                 | **实际应用场景**                                             |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数据               | all the persons are represented by images/videos captured by **single-modality visible cameras** in the closed-world setting | process heterogeneous data, which are **infrared images** [21], [60], **sketches** [61], **depth images** [62], or even **text descriptions** |
| 人物框选           | generated bounding boxes  ——已经框选好的                     | require end-to-end person search from the raw images or videos   ——端到端也就是要自己处理 |
| 标注               | 大量且已经标注好                                             | 少量或没有标注                                               |
| 标注正确性         | assume that all the annotations are correct, with clean labels | annotation noise                                             |
| query是否在gallery | assume that the **query must occur in the gallery set** by calculating the CMC [68] and mAP | query person may not appear in the gallery set [69], [70], or we need to perform the verification rather than retrieval [26]. This brings us to the open-set person Re-ID |

## 3. CLOSED-WORLD PERSON RE-IDENTIFICATION  

**假设条件**：

- 单一模态捕捉的图像或视频
- 人物已经被框选，大多数是同一人
- 有足够的标注
- 标注正确
- query person肯定在
- gallery set.   

**standard closed-world Re-ID system  三个主要组件：**

- **Feature Representation Learning**  ——focuses on developing the **feature construction strategies**  
- **Deep Metric Learning**——which aims at **designing the training objectives with different loss** functions or **sampling strategies** 
- **Ranking Optimization**    ——concentrates on optimizing the retrieved ranking list.   

### 2.1 Feature Representation Learning

#### four main categories  

1. Global Feature ——global feature representation vector for each person image without additional annotation cues  
2. Local Feature—— it aggregates part-level local features to formulate a combined representation for each person image  
3. Auxiliary Feature——it improves the feature representation learning **using auxiliary information,** e.g., **attributes** [71], [72], [78], GAN generated images [42], etc  
4. Video Feature——it learns video representation for video-based Re-ID [7] **using multiple image frames and temporal information** [73], [74  

![image-20240218205124813](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240218205124813.png)

#### 2.1.1 Global Feature Representation Learning

- joint learning framework consisting of a singleimage representation (SIR) and cross-image representation (CIR)     **training process as a multi-class classification problem by treating each identity as a distinct class.  **

- Attention Information.  

- > Attention information in person re-identification refers to techniques used to enhance feature learning by **focusing on specific parts of the data**. It includes **pixel-level** attention which emphasizes individual pixels, **part-level** attention which focuses on different regions of a person's image, and **spatial or background suppression to reduce noise from irrelevant areas**. It also includes **context-aware attention** for handling multiple person images, which improves the feature learning by considering the relationships between different images or sequences. These attention mechanisms contribute to more accurate identification by highlighting relevant features and suppressing irrelevant ones.

**Global Feature Representation Learning in person re-identification primarily focuses on extracting a comprehensive feature vector for the entire person image. It utilizes networks originally designed for image classification and applies them to re-ID, leveraging fine-grained cues for learning distinctive features.**

#### 2.1.2 Local Feature Representation Learning

**Local Feature Representation Learning aims to be robust against issues such as misalignment of person images. It divides the body into parts or regions and extracts features from these specific areas. This method helps in accurately matching body parts across different images and is especially useful in dealing with variations in pose or when parts of the body are occluded.**

#### 2.1.3 Auxiliary Feature Representation Learning

**usually requires additional annotated information (e.g., semantic attributes [71]) or generated/augmented training samples to reinforce the feature representation  **

1. **Semantic Attributes**: 这些是描述性特征，例如“男性”、“短发”、“戴红帽子”等，可用于提供额外的上下文并提高特征表示的准确性。学习模型可以使用这些语义属性来更有效地区分个体，尤其是在并非所有数据都可以标记的半监督学习环境中。
2. **Viewpoint Information**: 这考虑了在不同摄像机上捕捉人物的角度。通过考虑视点，模型可以学会从不同角度识别同一个人，这对于跨多个摄像头进行强大的 Re-ID 至关重要。
3. **Domain Information**: 将来自不同相机的图像视为不同的域，此方法旨在提取考虑跨相机变化的全局最优特征集。这可能涉及对这些域的要素进行对齐，以确保一致的标识。
4. **GAN Generation**: 使用生成对抗网络（GAN）创建合成图像有助于解决跨相机变化问题，并增强模型的鲁棒性。这些生成的图像可以为训练提供额外的数据点，当实际图像稀缺或尝试对不同的环境条件进行建模时，特别有用。
5. **Data Augmentation**: 使用随机调整大小、裁剪和翻转等自定义数据增强方法，人为地扩展数据集，使训练后的模型更具泛化性，不易出现过度拟合。更复杂的技术可能包括生成遮挡样本或应用随机擦除策略来模拟 Re-ID 系统可能遇到的更多实际情况。

#### 2.1.4 Video Feature Representation Learning

additional challenges   

1. accurately capture the temporal information.   
2. unavoidable outlier tracking frames within the videos  
3. handle the varying lengths of video sequences  

#### 2.1.5 Architecture Design

设计不同架构来解决

### 2.2 Deep Metric Learning

#### 2.2.1 Loss Function Design

identity loss, verification loss and triplet loss  

![image-20240219160950101](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240219160950101.png)

Re-ID领域中常用的四种损失函数及其作用：

1. **Identity Loss（身份损失）**:
   - 将人员Re-ID的训练过程视为一个**图像分类问题**，其中每个身份被视为一个独立的类别。
   - 在测试阶段，使用池化层或嵌入层的输出作为特征提取器。
   - 使用softmax函数计算输入图像被正确识别为其类别的概率，并通过交叉熵计算身份损失。
   - 身份损失在训练过程中自动挖掘难样本，简单易训练，且通常与标签平滑等策略结合使用以提高模型的泛化能力。

2. **Verification Loss（验证损失）**:
   - 优化成对关系，使用对比损失或二元验证损失来改善相对成对距离的比较。
   - 对比损失关注于增强样本对之间的欧氏距离比较，通过最大化同一身份内样本的相似性和不同身份样本的差异性。
   - 二元验证损失区分图像对的正负，关注于识别输入图像对是否属于同一身份。

3. **Triplet Loss（三元组损失）**:
   - 将Re-ID模型训练过程视为一个检索排序问题，确保同一身份的样本对距离小于不同身份样本对的距离。
   - 三元组包含一个锚点样本、一个正样本（与锚点同一身份）和一个负样本（不同身份），通过预定义的边际参数优化这三者之间的距离。
   - 为了提高训练的效果，采用了各种信息三元组挖掘方法，以选择更具信息量的三元组进行训练。

4. **OIM Loss（在线实例匹配损失）**:
   - 设计了一个包含存储实例特征的内存库，通过内存库优化在线实例匹配。
   - OIM损失通过比较输入特征与内存库中存储的特征之间的相似度，处理大量非目标身份的实例。
   - 这种方法在无监督领域自适应Re-ID中也得到了应用，通过控制相似度空间的温度参数优化实例匹配分数。

#### 2.2.2 Training strategy

> 训练策略（Training strategy）是机器学习和深度学习中一组用于指导模型训练过程的方法和技术。它包括各种技巧和方法，旨在提高模型的学习效率、性能和泛化能力。

1. **批量采样策略的挑战**：
   - 由于每个身份标注的训练图像数量差异很大，以及正负样本对之间严重不平衡。
2. **身份采样**：
   - 处理样本不平衡问题的最常见策略是身份采样。在这种策略下，每个训练批次会随机选取一定数量的身份，然后从每个选定的身份中采样几张图像。这种批量采样策略保证了有效的正负样本挖掘。
3. **适应性采样**：
   - 为了处理正负样本之间的不平衡问题，流行的方法是适应性采样，通过调整正负样本的贡献来应对不平衡，例如采样率学习（SRL）和课程采样等。
4. **样本重权**：
   - 另一种方法是样本重权，通过使用样本分布或相似性差异来调整样本权重。这有助于平衡训练过程中样本的影响，提高模型对不同样本的区分能力。
5. **高效的参考约束**：
   - 设计了高效的参考约束来将成对/三元组相似性转化为样本到参考的相似性，这不仅解决了不平衡问题，而且增强了区分性，并且对异常值具有鲁棒性。
6. **多损失动态训练策略**：
   - 通过适应性地重新加权身份损失和三元组损失，动态组合多个损失函数，可以提取它们之间的共享组件。这种多损失训练策略导致了一致的性能提升。

### 2.3 Ranking Optimization

#### 2.3.1 Re-ranking  

![image-20240219165610757](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240219165610757.png)

通过不同的技术和方法（如重排、查询适应性和人机交互）来实现更准确的排名顺序。这些方法能够根据不同的情境和需求，灵活地优化排名结果，从而提高检索的准确性和效率。

#### 2.3.2 Rank Fusion  

在实际应用中，不同的检索或识别算法可能对同一数据集有不同的理解和表现，某些算法在特定情境下表现良好，而在其他情境下表现可能较差。排名融合通过合理地结合这些算法产生的结果，旨在充分利用每种方法的优势，从而提供一个综合考虑了多种视角和信息的更准确、更可靠的排名结果。

### 2.4 Datasets and Evaluation  

**数据集**：

https://github.com/NEU-Gou/awesome-reid-dataset?tab=readme-ov-file

**评估指标**：

1. **累积匹配特性（CMC）**:
   - CMC曲线或CMC-k指标（又称Rank-k匹配准确率）反映了在前k个检索结果中找到正确匹配的概率。当每个查询仅对应一个正确结果时，CMC提供了一个准确的评估。然而，在包含多个正确匹配项的大型摄像头网络中，CMC可能无法完全反映模型跨多个摄像头的区分能力。
2. **平均平均精确度（mAP）**:
   - mAP衡量的是在有多个正确匹配项时的平均检索性能，它在图像检索领域被广泛使用。对于Re-ID评估，mAP可以解决两个系统在查找第一个正确匹配（可能是容易的匹配）时表现相同，但在检索其他难度较大的匹配项时能力不同的问题。
3. **FLOPs（浮点操作次数每秒）**:
   - FLOPs是衡量模型复杂度和运算效率的指标，特别是在计算资源受限的训练/测试设备上，FLOPs成为了一个重要的考量因素。它反映了执行某个操作或运行模型一次所需的浮点运算次数。
4. **网络参数大小**:
   - 网络参数大小指的是构成模型的参数总量，这直接影响模型的存储需求和计算复杂度。在资源受限的环境中，参数越少的模型越受欢迎，因为它们占用的内存少，运行速度可能更快。

**一些代表性方法**

![image-20240219200811035](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240219200811035.png)

![image-20240219200833755](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240219200833755.png)

## 3 OPEN-WORLD PERSON RE-IDENTIFICATION

### 3.1 Heterogeneous Re-ID

1. **基于深度的Re-ID**：
   - 利用深度图像捕捉人体形状和骨骼信息，这对于在光照变化或衣物更换环境下的Re-ID尤为重要，也适用于个性化人机交互应用。提到了一种基于recurrent attention-based 模型来识别人体的小型、区分性的局部区域，并利用RGB到深度的转换方法桥接深度图像与RGB图像间的差距。
2. **文本到图像的Re-ID**：
   - 处理文本描述与RGB图像间的匹配问题，特别重要于当查询人物的视觉图像无法获取，只能提供文本描述的情况。采用**门控神经注意力模型学习文本描述与人物图像间的共享特征，以实现文本到图像行人检索的端到端训练。**
3. **可见光到红外的Re-ID**：
   - 解决日间可见光与夜间红外图像间的跨模态匹配问题，对低光照条件下的应用至关重要。采用深度零填充框架学习模态可共享的特征，并引入双流网络模型同时处理模态内和跨模态的变化，以及使用GAN技术生成跨模态人物图像以减少跨模态差异。
4. **跨分辨率的Re-ID**：
   - 在低分辨率和高分辨率图像间进行匹配，解决大的分辨率变化问题。使用级联的SR-GAN生成高分辨率人物图像，并采用对抗学习技术获取分辨率不变的图像表示

### 3.2 End-to-End Re-ID

这个任务要求模型在一个框架内同时执行人员检测和重新识别。这是具有挑战性的，因为这两个主要组成部分的关注点不同。

1. **原始图像/视频中的Re-ID**：
   - Zheng等人提出了一个两阶段框架，系统评估了人员检测对后续人员Re-ID的好处和限制。
   - Xiao等人设计了一个端到端的人员搜索系统，使用单个卷积神经网络同时进行人员检测和重新识别。
   - 开发了一种神经人员搜索机器（NPSM），通过充分利用查询和检测到的候选区域之间的上下文信息，递归细化搜索区域并定位目标人员。
   - 类似地，在图学习框架中学习了一个上下文实例扩展模块，以改进端到端的人员搜索。
   - 开发了一个以查询为指导的端到端人员搜索系统，该系统使用Siamese squeeze-and-excitation网络捕获全局上下文信息，并生成查询引导的区域提议。
   - 引入了一个具有区分性Re-ID特征学习的定位细化方案，以生成更可靠的边界框。
   - 身份识别注意力增强学习（IDEAL）方法选择信息丰富的区域，以自动生成边界框，从而提高Re-ID性能。
2. **多摄像头跟踪**：
   - 端到端的人员Re-ID也与多人多摄像头跟踪密切相关。
   - 提出了一种基于图的公式，用于多人跟踪中链接人员假设，其中结合了整个人体的整体特征和身体姿势布局作为每个人的表示。
   - Ristani等人学习了多目标多摄像头跟踪与人员Re-ID之间的关联，通过困难身份挖掘和自适应加权三元组学习。
   - 最近，提出了一种具有内部和外部摄像头关系建模的局部感知外观度量（LAAM）。
3. **具有挑战性的问题**：
   - Yamaguchi等人研究了一个更具挑战性的问题，即使用**文本描述从原始视频中搜索人员。提出了一个具有时空人员检测和多模态检索的多阶段方法。**

### 3.3 Semi-supervised and Unsupervised Re-ID

#### 3.3.1 Unsupervised Re-ID

Early unsupervised Re-ID mainly **learns invariant components, i.e., dictionary [203], metric [204] or saliency [66], which leads to limited discriminability or scalability**.

1. **深度无监督方法**：
   - 采用跨摄像头标签估计作为流行方法之一。动态图匹配（DGM）将标签估计问题形式化为二分图匹配问题，并利用全局摄像头网络约束以实现一致匹配。其他方法包括逐步挖掘标签、强健的锚点嵌入方法，以及迭代聚类和Re-ID模型学习，利用估计的标签通过深度学习来学习Re-ID模型。
2. **端到端无监督Re-ID**：
   - 提出了一个迭代聚类和Re-ID模型学习的方法，以及利用样本间关系的层次聚类框架。软多标签学习从参考集中挖掘软标签信息进行无监督学习。跟踪片段关联无监督深度学习（TAUDL）框架同时进行摄像头内跟踪片段关联和跨摄像头跟踪片段相关性建模。
3. **局部特征表示学习**：
   - 一些方法尝试基于局部特征表示学习，因为在局部部分挖掘标签信息比整幅图像更容易。PatchNet旨在通过挖掘局部相似性来学习区分性的局部特征。
4. **半监督/弱监督Re-ID**：
   - 在有限的标签信息条件下，提出了一种一次性度量学习方法，结合了深度纹理表示和颜色度量。针对基于视频的Re-ID，提出了一种逐步的一次性学习方法（EUG），逐渐从未标注的跟踪片段中选择少数候选者以丰富标注的跟踪片段集合。还有利用视频级标签进行表示学习的多实例注意力学习框架，减轻了对完全标注的依赖。

#### 3.3.2 Unsupervised Domain Adaptation

> 无监督域适应（Unsupervised Domain Adaptation, UDA）是机器学习中的一种策略，特别是在迁移学习领域中。它旨在解决在源域（source domain）上有大量标记数据，而在目标域（target domain）上几乎没有或完全没有标记数据的情况。源域和目标域在特征分布上存在差异，这种差异会导致直接将在源域训练的模型应用到目标域上时性能下降。无监督域适应的目标是通过某种方式减少这两个域之间的分布差异，以提高模型在目标域上的泛化能力。

- **目标图像生成**：使用GAN生成技术将源域图像转换为目标域风格，通过生成的图像在未标记的目标域中进行有监督的Re-ID模型学习。提及了多种方法，如PTGAN、SPGAN、HHL、自适应转移网络等。
- **目标域监督挖掘**：一些方法直接从源数据集训练好的模型中挖掘未标记目标数据集的监督信息，如示例记忆学习、域不变映射网络（DIMN）、自训练方法和自步调对比学习框架等。

#### 3.3.3 State-of-The-Arts for Unsupervised Re-ID

- **性能提升**：无监督Re-ID性能在过去几年里显著提高，例如在Market-1501数据集上，Rank-1准确率/mAP从54.5%/26.3%提高到90.3%/76.7%。
- **进一步改进的空间**：包括在无监督Re-ID中应用强大的注意力机制，目标域图像生成的有效性，以及使用注释源数据进行跨数据集学习。
- **无监督与监督Re-ID之间的差距**：尽管无监督Re-ID取得了显著进展，但与监督Re-ID相比仍有较大差距。例如，监督方法ConsAtt在Market-1501数据集上的rank-1准确率达到96.1%，而无监督方法SpCL的最高准确率为90.3%。

![image-20240222181346734](https://raw.githubusercontent.com/thisisbaiy/PicGo/main/image-20240222181346734.png)

### 3.4 Noise-Robust Re-ID

**面对重度遮挡的部分Re-ID**

- **问题描述**：当人体只有部分可见时的Re-ID问题。
- 解决策略：
  - 使用全卷积网络生成不完整人像的固定大小空间特征图。
  - 深度空间特征重构（DSR）通过利用重构误差避免显式对齐。
  - 可见性感知部分模型（VPM）提取共享区域级特征，抑制不完整图像中的空间错位。
  - 前景感知金字塔重建方案从未被遮挡的区域学习。
  - 姿态引导的特征对齐（PGFA）利用姿态标记挖掘遮挡噪声中的判别性部分信息。

**因检测或跟踪错误导致的样本噪声**

- **问题描述**：人像图像或视频序列包含由于检测不良/跟踪不准确结果导致的异常区域/帧的问题。
- 解决策略：
  - 利用姿态估计线索或注意力线索来抑制噪声区域在最终整体表示中的贡献。
  - 对于视频序列，采用集合级特征学习或帧级重加权来减少噪声帧的影响。

**因标注错误引起的标签噪声**

- **问题描述**：由于标注错误而无法避免的标签噪声问题。
- 解决策略：
  - 郑等人采用标签平滑技术来避免标签过拟合问题。
  - 提及一个模型特征不确定性的分布网络（DNet），以对抗标签噪声，减少具有高特征不确定性样本的影响。
  - 与一般分类问题不同，鲁棒的Re-ID模型学习受限于每个身份的有限训练样本。
  - 未知的新身份为鲁棒的Re-ID模型学习增加了额外的难度。

### 3.5 Open-set Re-ID and Beyond

Open-set Re-ID is usually formulated as a person verification problem, i.e., **discriminating whether or not two person images belong to the same identity** .   

1. **群组Re-ID**

> 群组Re-ID（Group Re-identification）是指在监控视频或图片中识别并关联整个群组的过程，而不仅仅是个体。它的目的是在不同的摄像头视角、时间或地点中，识别并跟踪由多个人组成的群组。与传统的个体Re-ID不同，群组Re-ID需要处理群组内部的动态变化，如群组成员的进出、群组形态的变化以及不同环境下的群组外观差异。

- **目标**：旨在将个体识别扩展到群体识别，识别和关联群组中的人员而非单个个体。

2. **动态多摄像头网络**

> 动态多摄像头网络指的是由多个摄像头组成的监控系统，在这个系统中，摄像头的数量、位置或监控范围可以随时间变化或根据需求进行调整和更新。

- **问题**：需要对新加入的摄像头或探测器进行模型适应的动态更新网络。

## 4 AN OUTLOOK: RE-ID IN NEXT ERA

### 4.1 mINP: A New Evaluation Metric for Re-ID

### 4.2 A New Baseline for Single-/Cross-Modality Re-ID

### 4.3 Under-Investigated Open Issues

1. **不可控的数据收集**（Uncontrollable Data Collection）

- 挑战：现实环境中的数据收集往往是不可预测和不可控的，包括不同模式的数据、模式组合，甚至是衣物变换等情况。
- 多异构数据：实际应用中，Re-ID数据可能来自多种异构模态，如不同分辨率的图像、可见光、热成像、深度或文本描述等，这导致了多种异构人员Re-ID的挑战。

2. **人工标注最小化**（Human Annotation Minimization）

- 方法：除了无监督学习之外，主动学习或人机交互提供了减轻对人工标注依赖的可能解决方案。
- 主动学习：通过人机交互，为新到达的数据提供标签，并随后更新模型，旨在最小化人工标注的努力。

3. **特定域**/**泛化架构设计**（Domain-Specific/Generalizable Architecture Design）

- Re-ID特定架构：探索特定于Re-ID任务的架构设计，以提高特征提取的效果。
- 域泛化Re-ID：学习一个能够泛化到未见数据集的模型，以适应不同数据集之间存在的大域差异，而无需额外训练。

4. **动态模型更新**（Dynamic Model Updating）

- 模型适应新域/摄像头：针对新域或新摄像头的模型适应，解决因摄像头网络动态更新而导致的持续识别问题。
- 新数据的模型更新：使用新收集的数据更新模型，采用增量学习方法和人机交互，而不需要从头开始训练模型。

5. **高效模型部署**（Efficient Model Deployment）

- 快速Re-ID：通过哈希等方法提高检索速度，近似最邻近搜索。
- 轻量级模型：设计轻量级Re-ID模型来解决可扩展性问题，包括网络架构修改和模型蒸馏。
- 资源感知Re-ID：根据硬件配置适应性调整模型，解决可扩展性问题。
