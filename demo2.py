import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import numpy as np
from dataSet import PhpDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from pretreatment.code_pre import code_pre
import NNModel


if __name__ == '__main__':

    print('initing model...')
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # model = RobertaModel.from_pretrained("microsoft/codebert-base")


    print('open file.....')
    try:
        rf = open('/home/liuyi/Document/data/odata/a.php', 'r', encoding='utf-8', errors='ignore')
        data = rf.read()
    finally:
        rf.close()



    data = code_pre(data)[:10000]
    print(data)

    inputs = tokenizer.encode_plus(
        data,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    data = {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
    }

    # targets = data['targets']
    # targets = targets.view(-1, 1)
    # targets = torch.LongTensor(targets)
    # targets = torch.zeros(batch_size, 2).scatter_(1, targets, 1)

    ids = data['ids'].view(1, 512).cpu()
    mask = data['mask'].view(1, 512).cpu()
    token_type_ids = data['token_type_ids'].view(1, 512).cpu()

    # print(type(torch.load('model/cls_model_9.pth')))
    model = NNModel.CodeBERTClassifer().cpu()
    model.load_state_dict(torch.load('model/cls_model_0.pth'))

    outputs = model(ids, mask, token_type_ids)

    print(outputs)


    # print(model)