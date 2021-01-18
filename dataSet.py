import torch
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pretreatment.code_pre import code_pre
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW


class PhpDataset(Dataset):
    def __init__(self):

        self.black_file_list = os.listdir(r'./')
        self.white_file_list = os.listdir(r'./')

        self.black = [r'./' + i for i in self.black_file_list]
        self.white = [r'./' + i for i in self.white_file_list]

        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")



        self.df = self.black + self.white

    def __getitem__(self, item):


        try:
            rf = open(self.df[item], 'r', encoding='utf-8', errors='ignore')
            data = rf.read()
        finally:
            # print(data)
            # print(self.df[item])
            rf.close()

        data = code_pre(data)[:10000]
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
            'targets': torch.tensor(1 if self.df[item][40:43] == 'bla' else 0)
        }

        #
        # outputs = self.model(torch.tensor([inputs['input_ids']]))
        #
        # label = 1 if self.df[item][40:43] == 'bla' else 0
        # return outputs, label

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    dataset = PhpDataset()
    print(dataset[0])

