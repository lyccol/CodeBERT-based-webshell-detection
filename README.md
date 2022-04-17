news
等找到实习后，整理一下代码，供大家方便使用。


# CodeBERT-based-webshell-detection

BERT : https://arxiv.org/abs/1810.04805

CodeBERT：[sxjscience/CodeBERT: CodeBERT (github.com)](https://github.com/sxjscience/CodeBERT)

我们使用CodeBERT作为预训练模型，作为模型的encode部分，使用TextCNN作为decoder部分对php代码进行二分类Fine-Tune，其中webshell为黑样本，其他为白样本。





### Code Documentation Generation

#### Dependency

- pip install torch==1.7.1
- pip install transformers==4.0.1
- pip install filelock more_itertools

#### database

数据集黑样本为github中开源项目[tennc/webshell: This is a webshell open source project (github.com)](https://github.com/tennc/webshell)

和论文Webshell Detection Based on the Word Attention Mechanism中的数据集：[leett1/Programe (github.com)](https://github.com/leett1/Programe/)

取出php文件并去重之后为2000多个。

白样本约8000个。



 

由于数据集质量不高，Acc仅供参考

Accuracy Score = 1

F1 Score (Micro) = 1

F1 Score (Macro) = 1



数据集和模型太大所以没有上传，有疑问6517465@qq.com
