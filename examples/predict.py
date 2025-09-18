"""Run inference with a trained webshell detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from webshell_detection import DEFAULT_MODEL_NAME, ModelConfig, PhpDataset, TextCNNClassifier
from webshell_detection.predictor import WebshellPredictor
from webshell_detection.training import create_dataloader

LABELS = {0: "benign", 1: "malicious"}


def _merge_directory_args(primary: List[str] | None, legacy: List[str] | None) -> List[str]:
    directories: List[str] = []
    if primary:
        directories.extend(primary)
    if legacy:
        directories.extend(legacy)
    return directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference on PHP files using a trained model."
    )
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to the trained model weights."
    )
    parser.add_argument("--malicious-dirs", nargs="*", help="Directories with malicious PHP files.")
    parser.add_argument("--benign-dirs", nargs="*", help="Directories with benign PHP files.")
    parser.add_argument(
        "--black-dir",
        dest="legacy_malicious",
        action="append",
        help="Legacy alias for malicious dirs.",
    )
    parser.add_argument(
        "--white-dir", dest="legacy_benign", action="append", help="Legacy alias for benign dirs."
    )
    parser.add_argument(
        "--encoder-name",
        default=DEFAULT_MODEL_NAME,
        help="Transformer checkpoint used during training.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for labelling a file as malicious.",
    )
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    malicious_dirs = _merge_directory_args(args.malicious_dirs, args.legacy_malicious)
    benign_dirs = _merge_directory_args(args.benign_dirs, args.legacy_benign)

    if not malicious_dirs and not benign_dirs:
        parser.error("Provide at least one directory with PHP files to analyse.")

    args.malicious_dirs = malicious_dirs
    args.benign_dirs = benign_dirs
    return args


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    dataset = PhpDataset(
        args.malicious_dirs or [],
        args.benign_dirs or [],
        model_name=args.encoder_name,
        max_length=args.max_length,
    )
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    model = TextCNNClassifier(config=ModelConfig(encoder_name=args.encoder_name))
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)

    predictor = WebshellPredictor(model)
    predictions = predictor.predict_dataset(dataloader)

    for prediction in predictions:
        verdict = "malicious" if prediction.malicious_probability >= args.threshold else "benign"
        truth = LABELS.get(prediction.true_label, "unknown")
        print(
            f"{prediction.path}: predicted={verdict} "
            f"(p_malicious={prediction.malicious_probability:.3f}) expected={truth}"
        )


if __name__ == "__main__":
    main()
