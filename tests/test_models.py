from types import SimpleNamespace

import torch

from webshell_detection.models import (
    BERTClassifier,
    CodeBERTClassifier,
    ModelConfig,
    TextCNNClassifier,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int = 50, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.pooler = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True):
        embedded = self.embeddings(input_ids)
        pooled = self.activation(self.pooler(embedded.mean(dim=1)))
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            return BaseModelOutputWithPooling(last_hidden_state=embedded, pooler_output=pooled)
        return embedded, pooled


def _dummy_batch(batch_size: int = 2, seq_len: int = 5):
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).view(batch_size, seq_len) % 10
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def test_bert_classifier_produces_logits():
    encoder = DummyEncoder()
    model = BERTClassifier(config=ModelConfig(num_labels=2), encoder=encoder)
    input_ids, attention_mask = _dummy_batch()
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 2)


def test_codebert_classifier_produces_logits():
    encoder = DummyEncoder()
    model = CodeBERTClassifier(config=ModelConfig(num_labels=3), encoder=encoder)
    input_ids, attention_mask = _dummy_batch()
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 3)


def test_textcnn_classifier_handles_variable_filters():
    encoder = DummyEncoder()
    config = ModelConfig(num_labels=2, filter_sizes=(1, 2, 3))
    model = TextCNNClassifier(config=config, encoder=encoder)
    input_ids, attention_mask = _dummy_batch(seq_len=6)
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 2)
