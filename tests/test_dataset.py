from pathlib import Path

import torch

from webshell_detection.dataset import PhpDataset, collect_php_files


class DummyTokenizer:
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __call__(
        self,
        text,
        *,
        padding,
        truncation,
        max_length,
        return_attention_mask,
        return_token_type_ids,
    ):
        length = min(len(text.split()), max_length)
        input_ids = list(range(length)) + [0] * (max_length - length)
        attention_mask = [1] * length + [0] * (max_length - length)
        token_type_ids = [0] * max_length if return_token_type_ids else None
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        return result


def create_php_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_collect_php_files_supports_files_and_directories(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    file_path = create_php_file(root / "shell.php", "<?php echo 'hi'; ?>")
    nested_dir = root / "benign"
    nested_dir.mkdir()
    create_php_file(nested_dir / "index.php", "<?php echo 'ok'; ?>")

    files = collect_php_files([root])
    assert len(files) == 2
    assert file_path.resolve() in files


def test_php_dataset_returns_tensorised_batches(tmp_path):
    malicious = tmp_path / "mal"
    benign = tmp_path / "ben"
    malicious.mkdir()
    benign.mkdir()

    create_php_file(malicious / "a.php", "<?php echo 'hi'; ?>")
    create_php_file(benign / "b.php", "<?php echo 'bye'; ?>")

    dataset = PhpDataset(malicious, benign, tokenizer=DummyTokenizer(), max_length=8)
    assert len(dataset) == 2

    item = dataset[0]
    assert set(item.keys()) >= {"input_ids", "attention_mask", "labels", "path"}
    assert item["input_ids"].shape[0] == 8
    assert item["labels"].dtype == torch.long


def test_dataset_allows_unlabelled_inference(tmp_path):
    malicious = tmp_path / "mal"
    malicious.mkdir()
    create_php_file(malicious / "shell.php", "<?php echo 'bad'; ?>")

    dataset = PhpDataset(malicious, benign_dirs=None, tokenizer=DummyTokenizer(), max_length=4)
    assert len(dataset) == 1
    assert dataset[0]["labels"].item() == 1
