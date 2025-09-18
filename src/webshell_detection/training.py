"""Training helpers for webshell detection models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = ["TrainingConfig", "create_dataloader", "train_one_epoch", "train"]


@dataclass
class TrainingConfig:
    """Configuration options for the training loop."""

    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    device: str | None = None
    gradient_clip: float | None = None
    output_dir: Path | None = None
    save_every_epoch: bool = False
    seed: int | None = 13
    show_progress: bool = True

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloader(
    dataset, *, batch_size: int, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """Create a :class:`torch.utils.data.DataLoader` with sane defaults."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _move_to_device(
    batch: MutableMapping[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    gradient_clip: float | None = None,
    show_progress: bool = True,
) -> float:
    """Train *model* for a single epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(dataloader, disable=not show_progress, leave=False)
    for batch in iterator:
        batch = _move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += float(loss.detach())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> List[float]:
    """Train *model* using batches from *dataloader* and return loss history."""

    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = config.resolve_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    history: List[float] = []

    output_dir: Path | None = config.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            device,
            gradient_clip=config.gradient_clip,
            show_progress=config.show_progress,
        )
        history.append(avg_loss)

        if output_dir is not None and (config.save_every_epoch or epoch == config.epochs):
            checkpoint_path = output_dir / f"model-epoch-{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    return history
