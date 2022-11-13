Projects in Knowledge Engineering class (2022 Spring, BIT)  北理工 知识工程 大作业

See report.pdf in each folder for detailed information of each project.

## [Named Entity Recognition （NER）](https://github.com/Rebabit/Knowledge-Engineering/tree/main/NER)

命名实体识别，即对文本中具有特定意义的词进行识别，在自然语言处理领域具有基础性意义。本实验采用北京大学《人民日报》基本标注语料库， 通过构建广义线性模型，实现了对人名、地名、机构名三类实体的识别，并采用BIO表示法进行了相应的标注。

本实验主要通过以下步骤，实现了命名实体识别的目标：

1. 对原始数据集进行向量化表示，选取特征将每个词语表示为一个向量，并将对应的分类标签转化为可以参与运算的数字。
2. 生成训练集、验证集和测试集，并对训练集中的数据进行数据清洗。
3. 初始化参数W，确定学习率和迭代次数，重复步骤4~7直到达到选定的迭代次数：
4. 采用mini-batch方法，抽取一定数量的训练集数据，计算Softmax概率。对于一个输入，概率最大的即为当前模型所预测的分类。
5. 计算当前模型的损失函数值，并对其求导。
6. 根据损失函数的导数，采用梯度下降法对参数W进行更新。
7. 将模型预测的分类与实际分类进行比较，求出用于评估当前模型精确程度的训练集与验证集的F1-measure，并保存使得验证集F1-measure最大值及对应的参数W。
8. 通过改变输入词的特征选取、训练集中数据清洗的方式、训练时的学习率和迭代次数、参数W更新的方法，比较其对应的验证集F1-measure的大小，选择最好的模型用于预测。
9. 将对应模型在步骤7保留的W值用于测试集，计算其F1-measure。

![fig](NER/fig.png)

## [Paper Reading](https://github.com/Rebabit/Knowledge-Engineering/tree/main/paper%20reading)
Brief introduction to [*TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking*](https://arxiv.org/abs/2010.13415v1) in COLING 2020.

- report.pdf: text version
- video.mp4: video version

### main idea

![fig1](paper%20reading/fig1.png)

### main method

![fig2](paper%20reading/fig2.png)

## [Relation Extraction](https://github.com/Rebabit/Knowledge-Engineering/tree/main/relation%20extraction)
Pytorch version of [GPLINKER](https://kexue.fm/archives/8888) based on the code [xhw205/GPLinker_torch](https://github.com/xhw205/GPLinker_torch).

Changes:
- add the evaluation part and a lot of annotations
- write the main part in jupyternotebook, a more reader-friendly way.

### 模型结构
![fig1](relation%20extraction/fig1.png)

### 代码各文件说明

1. **模型配置**：config.ini文件

   包括数据集与预训练模型的路径、训练时的参数设置，方便了参数的管理，其中：

   - [paths]：预训练模型RoBERTa、数据集的位置
   - [paras]：训练时的参数，其中，head_size对应Multi-Head Attention中head的维度

2. **数据集存储**：datasets文件夹

   本实验采用医学实体关系抽取CMeIE数据集，以json文件的形式存储，包含以下文件：

   - 53_schema.json: SPO关系约束表
   - CMeIE_train.json: 训练集
   - CMeIE_dev.json: 验证集
   - CMeIE_test.json: 测试集

3. **数据预处理**：utils/data_loader.py文件

   该部分将数据集数据进行编码，并将数据、标签转化为可用于GPLinker模型所需要的形式。其中，data_generator类提供了pytorch的DataLoader在训练、推理时的数据生成方式。

4. **GPLinker模型**：GPlinker.py文件

   采用pytorch的神经网络库，实现了关系抽取GPLinker模型，并规定了损失函数的计算方式。

5. **优化方法**：utils/bert_optimization.py文件

   采用HuggingFace针对Bert的Adam优化器，即文件中的BertAdam函数，并在训练过程中采用权重衰减、学习率预热、梯度剪裁等方式。

6. **模型参数存储**：result/GPLinker_para.pth文件

   训练结束后，存储模型的所有参数，用于评估模式下的关系抽取预测。

7. **训练与评估**：main.ipynb文件

   完成GPLinker的正反向传播，存储各项参数，并将其用于模型的评估。

