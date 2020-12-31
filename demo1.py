import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import numpy as np
from dataSet import PhpDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from pretreatment.code_pre import code_pre
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import NNModel
import os


def walkFile(file):
    file_list = []
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            if os.path.join(root, f)[-3:] == 'php':
                file_list.append(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            os.path.join(root, d)
    return file_list


class testDataset(Dataset):
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")

        # self.df = walkFile('/home/liuyi/Document/data/odata/wordpress')
        self.df = walkFile('/home/liuyi/Document/data/webshell_data/bla')
        # self.df = walkFile('/home/liuyi/Document/data/webshell_data/whi')

    def __getitem__(self, item):
        try:
            rf = open(self.df[item], 'r', encoding='utf-8', errors='ignore')
            data = rf.read()
        finally:
            # print(data)
            # print(self.df[item])
            rf.close()

        data = code_pre(data)[:10000]
        # print(data)
        # data = data
        # print(len(data))
        # print(len(data))
        inputs = self.tokenizer.encode_plus(
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

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'filename': self.df[item]
        }

        #
        # outputs = self.model(torch.tensor([inputs['input_ids']]))
        #
        # label = 1 if self.df[item][40:43] == 'bla' else 0
        # return outputs, label

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':


    data_set = testDataset()
    data_loading = DataLoader(dataset=data_set, batch_size=32, shuffle=True, num_workers=2, drop_last=False)

    model = NNModel.TextCNNClassifer().cpu()
    model.load_state_dict(torch.load('model/cls_model_0.pth'))

    for _, data in enumerate(data_loading, 0):
        # print(data['filename'])
        time_start = time.time()
        ids = data['ids'].cpu()
        mask = data['mask'].cpu()
        token_type_ids = data['token_type_ids'].cpu()

        outputs = model(ids, mask, token_type_ids)
        pred_choice = outputs.max(1)[1]

        index = torch.where(pred_choice == 0)

        for i in index:
            index = i.numpy().tolist()
        for i in index:
            print(outputs[i])
            print(data['filename'][i])

        # print(outputs)
        # print(pred_choice)
        # print(ids.size())
        time_end = time.time()
        print('totally cost', time_end - time_start)
        # break
