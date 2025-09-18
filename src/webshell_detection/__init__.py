"""CodeBERT-based webshell detection utilities."""

from __future__ import annotations

from .dataset import DEFAULT_MODEL_NAME, PhpDataset, PhpSample, collect_php_files
from .models import BERTClassifier, CodeBERTClassifier, ModelConfig, TextCNNClassifier
from .predictor import Prediction, WebshellPredictor
from .preprocessing import code_pre, preprocess_php_code
from .training import TrainingConfig, create_dataloader, train

__all__ = [
    "DEFAULT_MODEL_NAME",
    "PhpDataset",
    "PhpSample",
    "collect_php_files",
    "ModelConfig",
    "BERTClassifier",
    "CodeBERTClassifier",
    "TextCNNClassifier",
    "code_pre",
    "preprocess_php_code",
    "TrainingConfig",
    "create_dataloader",
    "train",
    "Prediction",
    "WebshellPredictor",
    "__version__",
]

__version__ = "0.2.0"
