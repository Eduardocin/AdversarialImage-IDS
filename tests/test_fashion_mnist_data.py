from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.fashion_mnist import (  # noqa: E402
    FASHION_MNIST_CLASSES,
    load_fashion_mnist_evaluation_split,
    load_fashion_mnist_training_split,
)


def _write_fashion_csv(path: Path, rows_per_class: int = 4) -> None:
    header = ["label"] + ["pixel{0}".format(index) for index in range(1, 785)]
    rows = [",".join(header)]
    for label in range(10):
        for offset in range(rows_per_class):
            pixel = str(label * 10 + offset)
            rows.append(",".join([str(label)] + [pixel] * 784))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _dataset_config(csv_path: Path) -> dict:
    return {
        "name": "fashion_mnist",
        "domain": "mnist_compatible",
        "split": "test",
        "csv_path": str(csv_path),
        "split_strategy": {
            "name": "balanced_by_class",
            "train_samples": 20,
            "evaluation_samples": 20,
            "train_class_quota": 2,
            "evaluation_class_quota": 2,
        },
        "image_shape": [28, 28, 1],
        "class_order": list(FASHION_MNIST_CLASSES),
        "class_indices": {
            class_name: index for index, class_name in enumerate(FASHION_MNIST_CLASSES)
        },
    }


def test_fashion_mnist_csv_loader_returns_balanced_nonoverlapping_splits(tmp_path) -> None:
    csv_path = tmp_path / "fashion-mnist_test.csv"
    _write_fashion_csv(csv_path, rows_per_class=4)
    config = _dataset_config(csv_path)

    train_images, train_labels, train_metadata = load_fashion_mnist_training_split(config)
    eval_images, eval_labels, eval_metadata = load_fashion_mnist_evaluation_split(config)

    assert train_images.shape == (20, 28, 28, 1)
    assert train_labels.shape == (20, 10)
    assert eval_images.shape == (20, 28, 28, 1)
    assert eval_labels.tolist() == [label for label in range(10) for _ in range(2)]
    assert train_metadata["split_strategy"]["train_class_quota"] == 2
    assert eval_metadata["split_strategy"]["evaluation_class_quota"] == 2
    assert train_metadata["training_sample"] == {
        "split_name": "train",
        "method": "first_n_per_class",
        "per_class_start": 0,
        "per_class_end": 2,
        "class_quota": 2,
        "total_samples": 20,
    }
    assert eval_metadata["evaluation_sample"] == {
        "split_name": "evaluation",
        "method": "next_n_per_class",
        "per_class_start": 2,
        "per_class_end": 4,
        "class_quota": 2,
        "total_samples": 20,
        "excludes_training": True,
    }
    assert set(train_metadata["train_indices"]).isdisjoint(
        set(eval_metadata["evaluation_indices"])
    )


def test_fashion_mnist_csv_loader_rejects_insufficient_balanced_rows(tmp_path) -> None:
    csv_path = tmp_path / "fashion-mnist_test.csv"
    _write_fashion_csv(csv_path, rows_per_class=3)
    config = _dataset_config(csv_path)

    with pytest.raises(ValueError, match="Insufficient Fashion-MNIST rows"):
        load_fashion_mnist_evaluation_split(config)
