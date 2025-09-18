"""Train a webshell detection model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from webshell_detection import DEFAULT_MODEL_NAME, ModelConfig, PhpDataset, TextCNNClassifier
from webshell_detection.training import TrainingConfig, create_dataloader, train


def _merge_directory_args(primary: List[str] | None, legacy: List[str] | None) -> List[str]:
    directories: List[str] = []
    if primary:
        directories.extend(primary)
    if legacy:
        directories.extend(legacy)
    return directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a webshell detector using CodeBERT embeddings."
    )
    parser.add_argument(
        "--malicious-dirs", nargs="+", help="Directories containing malicious PHP files."
    )
    parser.add_argument(
        "--benign-dirs",
        nargs="+",
        help="Directories containing benign PHP files.",
    )
    parser.add_argument(
        "--black-dir",
        dest="legacy_malicious",
        action="append",
        help="Legacy alias for --malicious-dirs.",
    )
    parser.add_argument(
        "--white-dir",
        dest="legacy_benign",
        action="append",
        help="Legacy alias for --benign-dirs.",
    )
    parser.add_argument(
        "--encoder-name",
        default=DEFAULT_MODEL_NAME,
        help="Transformer checkpoint to use as encoder.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Optimizer learning rate."
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store checkpoints. When omitted checkpoints are not saved.",
    )
    parser.add_argument(
        "--save-final-model",
        type=Path,
        help="Optional path to store the trained model weights.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    malicious_dirs = _merge_directory_args(args.malicious_dirs, args.legacy_malicious)
    benign_dirs = _merge_directory_args(args.benign_dirs, args.legacy_benign)

    if not malicious_dirs:
        parser.error("At least one directory with malicious files must be provided.")
    if not benign_dirs:
        parser.error("At least one directory with benign files must be provided.")

    args.malicious_dirs = malicious_dirs
    args.benign_dirs = benign_dirs
    return args


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    dataset = PhpDataset(
        args.malicious_dirs,
        args.benign_dirs,
        model_name=args.encoder_name,
        max_length=args.max_length,
    )
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TextCNNClassifier(config=ModelConfig(encoder_name=args.encoder_name))
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    history = train(model, dataloader, config)

    if args.save_final_model:
        args.save_final_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_final_model)

    print("Training completed. Average losses per epoch:")
    for epoch, loss in enumerate(history, start=1):
        print(f"  Epoch {epoch}: {loss:.4f}")


if __name__ == "__main__":
    main()
