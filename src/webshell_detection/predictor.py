"""Inference helpers for trained webshell detection models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import PhpDataset

__all__ = ["Prediction", "WebshellPredictor"]


@dataclass(frozen=True)
class Prediction:
    """Prediction result for a single PHP file."""

    path: Path
    predicted_label: int
    malicious_probability: float
    true_label: int | None = None


class WebshellPredictor:
    """Utility class for running inference with trained classifiers."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_dataset(self, dataloader: DataLoader) -> List[Prediction]:
        results: List[Prediction] = []
        softmax = torch.nn.Softmax(dim=1)

        for batch in dataloader:
            paths = batch.get("path")
            labels = batch.get("labels")
            tensor_batch = {
                key: value.to(self.device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            logits = self.model(
                tensor_batch["input_ids"],
                attention_mask=tensor_batch.get("attention_mask"),
                token_type_ids=tensor_batch.get("token_type_ids"),
            )
            probabilities = softmax(logits)
            preds = probabilities.argmax(dim=1)
            malicious_probs = (
                probabilities[:, 1] if probabilities.size(1) > 1 else probabilities[:, 0]
            )

            for idx, path in enumerate(paths):
                predicted_label = int(preds[idx].cpu().item())
                prob = float(malicious_probs[idx].cpu().item())
                true_label = None
                if labels is not None:
                    true_label = (
                        int(labels[idx])
                        if not isinstance(labels, torch.Tensor)
                        else int(labels[idx].cpu().item())
                    )
                results.append(Prediction(Path(path), predicted_label, prob, true_label))
        return results

    @torch.no_grad()
    def predict_directories(
        self,
        malicious_dirs: Sequence[Path | str],
        benign_dirs: Sequence[Path | str] | None = None,
        *,
        batch_size: int = 8,
        tokenizer=None,
        max_length: int = 512,
    ) -> List[Prediction]:
        dataset = PhpDataset(
            malicious_dirs,
            benign_dirs,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return self.predict_dataset(dataloader)
