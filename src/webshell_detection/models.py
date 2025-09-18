"""Model architectures for CodeBERT-based webshell detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

DEFAULT_MODEL_NAME = "microsoft/codebert-base"

__all__ = [
    "ModelConfig",
    "BERTClassifier",
    "CodeBERTClassifier",
    "TextCNNClassifier",
]


@dataclass
class ModelConfig:
    """Configuration used by the provided classifier architectures."""

    encoder_name: str = DEFAULT_MODEL_NAME
    num_labels: int = 2
    dropout: float = 0.1
    filter_sizes: Sequence[int] = (1, 2, 3, 4, 6, 8)
    filter_channels: int = 256


def _load_encoder(encoder: PreTrainedModel | None, encoder_name: str) -> PreTrainedModel:
    if encoder is not None:
        return encoder
    return AutoModel.from_pretrained(encoder_name)


def _pooled_output(outputs: BaseModelOutputWithPooling) -> torch.Tensor:
    if outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


class BERTClassifier(nn.Module):
    """Single linear classifier on top of a pooled CodeBERT encoder."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        encoder: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = _load_encoder(encoder, self.config.encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = _pooled_output(outputs)
        return self.classifier(self.dropout(pooled))


class CodeBERTClassifier(nn.Module):
    """Two-layer classifier with tanh activation on top of CodeBERT."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        encoder: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = _load_encoder(encoder, self.config.encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = self.activation(self.hidden(_pooled_output(outputs)))
        return self.classifier(self.dropout(pooled))


class TextCNNClassifier(nn.Module):
    """TextCNN-style classifier using contextual embeddings from CodeBERT."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        encoder: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig(dropout=0.2)
        self.encoder = _load_encoder(encoder, self.config.encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, self.config.filter_channels, (k, hidden_size))
                for k in self.config.filter_sizes
            ]
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(
            self.config.filter_channels * len(self.config.filter_sizes), self.config.num_labels
        )

    def conv_and_pool(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state.unsqueeze(1)
        features = torch.cat(
            [self.conv_and_pool(hidden_states, conv) for conv in self.convs], dim=1
        )
        features = self.dropout(features)
        return self.classifier(features)
