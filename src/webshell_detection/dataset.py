"""Dataset utilities for webshell detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .preprocessing import preprocess_php_code

DEFAULT_MODEL_NAME = "microsoft/codebert-base"

__all__ = [
    "DEFAULT_MODEL_NAME",
    "PhpSample",
    "PhpDataset",
    "collect_php_files",
]


@dataclass(frozen=True)
class PhpSample:
    """Representation of a PHP file and its associated label."""

    path: Path
    label: int


def _ensure_paths(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def collect_php_files(directories: Sequence[Path], pattern: str = "*.php") -> List[Path]:
    """Collect PHP files from the provided directories."""

    files: List[Path] = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        if not directory.exists():
            continue
        if directory.is_file():
            if directory.match(pattern):
                files.append(directory)
            continue
        files.extend(sorted(directory.rglob(pattern)))
    return files


def _tensorise(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.clone().detach().long()
    return torch.tensor(value, dtype=torch.long)


class PhpDataset(Dataset):
    """Dataset yielding tokenised PHP files and labels.

    Parameters
    ----------
    malicious_dirs:
        Directories containing malicious PHP files.
    benign_dirs:
        Directories containing benign PHP files.  When omitted the dataset will
        only contain samples labelled as malicious which can be useful for
        running inference on unlabelled data.
    tokenizer:
        Optional tokenizer used to encode PHP snippets. If omitted the
        CodeBERT tokenizer will be loaded automatically.
    model_name:
        Transformer model name used when instantiating the tokenizer.
    max_length:
        Maximum sequence length for tokenisation.
    file_encoding:
        Encoding used when reading files from disk.
    limit:
        Optional maximum number of samples to keep. When provided the dataset
        keeps an equal number of malicious and benign samples whenever
        possible.
    """

    def __init__(
        self,
        malicious_dirs: Union[str, Path, Sequence[Union[str, Path]]],
        benign_dirs: Union[str, Path, Sequence[Union[str, Path]]] | None,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 512,
        file_encoding: str = "utf-8",
        limit: int | None = None,
    ) -> None:
        self.malicious_dirs = _ensure_paths(malicious_dirs)
        self.benign_dirs = _ensure_paths(benign_dirs) if benign_dirs is not None else []
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.file_encoding = file_encoding

        malicious_files = [PhpSample(path, 1) for path in collect_php_files(self.malicious_dirs)]
        benign_files = [PhpSample(path, 0) for path in collect_php_files(self.benign_dirs)]

        samples: List[PhpSample] = []
        if limit is not None:
            if malicious_files and benign_files:
                half = max(limit // 2, 1)
                samples.extend(malicious_files[:half])
                samples.extend(benign_files[: limit - len(samples)])
            else:
                combined = malicious_files + benign_files
                samples.extend(combined[:limit])
        else:
            samples.extend(malicious_files)
            samples.extend(benign_files)

        if not samples:
            raise ValueError("No PHP files were discovered in the provided directories.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        text = sample.path.read_text(encoding=self.file_encoding, errors="ignore")
        processed = preprocess_php_code(text)
        tokenised = self.tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids="token_type_ids" in self.tokenizer.model_input_names,
        )

        batch = {
            "input_ids": _tensorise(tokenised["input_ids"]),
            "attention_mask": _tensorise(tokenised["attention_mask"]),
            "labels": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.path),
        }

        if "token_type_ids" in tokenised:
            batch["token_type_ids"] = _tensorise(tokenised["token_type_ids"])

        return batch
