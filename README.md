# CodeBERT-based Webshell Detection

Detect malicious PHP webshells using transfer learning on top of the
[`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base)
model. The repository provides an installable Python package with reusable
components, training utilities and command line tooling that follow common open
source project conventions.

## Key Features

- **Reusable dataset loader** that normalises PHP code and tokenises it for
  transformer-based models.
- **Multiple classifier heads** (linear, MLP and TextCNN) implemented with
  PyTorch and easily configurable.
- **Training helpers** with sensible defaults, checkpointing support and
  reproducibility options.
- **Inference utilities and scripts** for running predictions on labelled or
  unlabelled PHP files.
- **Extensive documentation** with contribution guidelines, code style
  recommendations and automated tests to validate the package behaviour.

## Installation

It is recommended to work inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package together with the runtime dependencies:

```bash
pip install -r requirements.txt
```

Alternatively you can install the project in editable mode which also registers
the console entry point:

```bash
pip install -e .
```

## Preparing a Dataset

The project expects two folders containing PHP files:

- `malicious/` (or "black") with known webshells.
- `benign/` (or "white") with legitimate PHP code.

Each folder can contain nested sub-directories and the loader will recursively
collect every `.php` file. If you want to evaluate the model on individual
files, you can pass the paths directly to the loader or inference script.

## Training

Use the `scripts/train.py` helper to fine-tune a classifier head on your data.
The CLI accepts both the new option names (`--malicious-dirs` / `--benign-dirs`)
 and the legacy `--black-dir` / `--white-dir` flags.

```bash
python scripts/train.py \
  --malicious-dirs data/train/malicious \
  --benign-dirs data/train/benign \
  --epochs 3 \
  --batch-size 8 \
  --output-dir runs/2024-04-01
```

The script prints the average training loss per epoch and, when `--output-dir`
is specified, stores checkpoints such as `model-epoch-3.pt` in that folder. Use
`--save-final-model` to persist the final weights to an arbitrary path.

If you prefer using the package programmatically you can combine
`webshell_detection.dataset.PhpDataset`, `webshell_detection.models.TextCNNClassifier`
and the helpers in `webshell_detection.training` to build a custom loop.

## Inference

Run predictions against labelled validation data or unlabelled PHP sources using
`examples/predict.py`:

```bash
python examples/predict.py \
  --model-path runs/2024-04-01/model-epoch-3.pt \
  --malicious-dirs data/val/malicious \
  --benign-dirs data/val/benign \
  --threshold 0.55
```

Each processed file is logged together with the predicted label, the probability
assigned to the malicious class and, when available, the expected label. The
predictor utilities exposed via `webshell_detection.predictor` can be embedded in
other applications to score arbitrary inputs.

## Project Layout

```
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── requirements.txt
├── src
│   └── webshell_detection
│       ├── __init__.py
│       ├── dataset.py
│       ├── models.py
│       ├── predictor.py
│       ├── preprocessing.py
│       └── training.py
├── scripts
│   └── train.py
├── examples
│   └── predict.py
└── tests
    ├── test_dataset.py
    ├── test_models.py
    └── test_preprocessing.py
```

## Development

The repository ships with a lightweight test-suite covering the dataset
parsing, preprocessing utilities and model outputs. After installing the
runtime dependencies, install the optional development requirements and run the
checks with `pytest`:

```bash
pip install -r requirements-dev.txt
pytest
```

When contributing, please format Python code with
[`black`](https://github.com/psf/black) and lint with
[`ruff`](https://github.com/astral-sh/ruff). Configuration for both tools is
included in `pyproject.toml`.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
full workflow, coding standards and tips on how to propose improvements. By
participating in this project you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
complete terms.
