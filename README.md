# CodeBERT-based-webshell-detection
![image-20201231101214855](https://github.com/lyccol/CodeBERT-based-webshell-detection/blob/main/QQ%E6%88%AA%E5%9B%BE20201231102121.jpg)


Pytorch implementation of Google AI's 2018 BERT, with simple annotation

BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding Paper URL : https://arxiv.org/abs/1810.04805

Dependency
pip install torch==1.7.1
pip install transformers==4.0.1
pip install filelock more_itertools


demo1.py : 用于使用模型进行预测

NNModel.py: 模型

train.py:



encode: CodeBERT

decode:TextCNN



Accuracy Score = 0.96875
F1 Score (Micro) = 0.96875
F1 Score (Macro) = 0.9625730994152046
