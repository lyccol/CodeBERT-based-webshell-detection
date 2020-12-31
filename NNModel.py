import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import transformers
import torch.nn.functional as F


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        o = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print(o['last_hidden_state'].size())
        # print(o['pooler_output'].size())
        output_2 = self.l2(o['pooler_output'])
        output_2 = output_2.view(-1, 768)
        output = self.l3(output_2)
        return output

class CodeBERTClassifer(torch.nn.Module):
    def __init__(self):
        super(CodeBERTClassifer, self).__init__()
        self.transformer = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()
        self.fc = nn.Linear(768, 768)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(0.6)
        self.classifier = nn.Linear(768, 2)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(768)

    def forward(self, ids, mask, token_type_ids):
        h = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # only use the first h in the sequence
        # print(h['last_hidden_state'].size())
        h = h['pooler_output']

        # h = self.bn1(h)
        h = self.fc(h)
        # h = self.bn2(h)
        # h = nn.BatchNorm2d(h)
        pooled_h = self.activ(h)
        logits = self.classifier(self.drop(pooled_h))
        return logits
        # return F.log_softmax(logits)

class GRUClassifer(torch.nn.Module):
    def __init__(self):
        super(GRUClassifer, self).__init__()
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()

        self.GRU = nn.LSTM(768, 64, 2, batch_first=True)
        self.fc = nn.Linear(768, 768)
        self.activ = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        h = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # only use the first h in the sequence
        pooled_h = self.activ(h['pooler_output'])

        self.GRU()

        logits = self.classifier(self.drop(pooled_h))
        return logits

class TextCNNClassifer(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer, self).__init__()
        self.encode = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (1, 2, 3, 4, 6, 8)])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * len((1, 2, 3, 4, 6, 8)), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, ids, mask, token_type_ids):
        h = self.encode(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # only use the first h in the sequence
        out = h['last_hidden_state']
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

# class TCodeBERTClassifer(torch.nn.Module):
#     def __init__(self):
#         super(TCodeBERTClassifer, self).__init__()
#         self.encode = RobertaModel.from_pretrained("microsoft/codebert-base").cuda()
#
#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=12)
#         self.decode = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
#
#         self.fc = nn.Linear(768, 768)
#         self.activ = nn.Tanh()
#         self.drop = nn.Dropout(0.3)
#         self.classifier = nn.Linear(768, 1)
#
#     def forward(self, ids, mask, token_type_ids):
#         h = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
#         # only use the first h in the sequence
#         pooled_h = self.activ(h['pooler_output'])
#         logits = self.classifier(self.drop(pooled_h))
#         return logits

if __name__ == '__main__':
    model = BERTClass()
    model()