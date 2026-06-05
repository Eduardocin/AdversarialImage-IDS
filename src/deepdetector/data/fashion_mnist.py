"""Fashion-MNIST CSV loading with balanced train/evaluation splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepdetector.io.paths import resolve_project_path


FASHION_MNIST_CLASSES: list[str] = [
    "t_shirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]

FASHION_MNIST_CLASS_INDICES: dict[str, int] = {
    class_name: index for index, class_name in enumerate(FASHION_MNIST_CLASSES)
}


def _resolve_csv_path(path_value: Any) -> Path:
    path = resolve_project_path(path_value)
    if path is None:
        raise ValueError("Fashion-MNIST dataset.csv_path is required.")
    if not path.is_file():
        raise IOError("Fashion-MNIST CSV not found: {0}".format(path))
    return path


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    output = np.zeros((len(label_array), int(num_classes)), dtype=np.float32)
    output[np.arange(len(label_array)), label_array] = 1.0
    return output


def load_fashion_mnist_csv(csv_path: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST CSV rows as normalized NHWC images and integer labels."""
    path = _resolve_csv_path(csv_path)
    data = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape((1, data.shape[0]))
    if data.shape[1] != 785:
        raise ValueError(
            "Fashion-MNIST CSV must contain one label and 784 pixels per row."
        )

    labels = np.asarray(data[:, 0], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels > 9):
        raise ValueError("Fashion-MNIST labels must be in the range 0..9.")

    pixels = np.asarray(data[:, 1:], dtype=np.float32) / 255.0
    images = pixels.reshape((pixels.shape[0], 28, 28, 1)).astype(np.float32)
    return images, labels


def _class_order(dataset_config: dict[str, Any]) -> list[str]:
    class_order = dataset_config.get("class_order", FASHION_MNIST_CLASSES)
    ordered = [str(class_name) for class_name in class_order]
    if len(ordered) != 10:
        raise ValueError("Fashion-MNIST class_order must contain exactly 10 classes.")
    return ordered


def _class_indices(dataset_config: dict[str, Any]) -> dict[str, int]:
    configured = dataset_config.get("class_indices", FASHION_MNIST_CLASS_INDICES)
    class_indices = {str(key): int(value) for key, value in dict(configured).items()}
    missing = [class_name for class_name in _class_order(dataset_config) if class_name not in class_indices]
    if missing:
        raise ValueError(
            "Fashion-MNIST class_indices missing classes: {0}".format(
                ", ".join(missing)
            )
        )
    return class_indices


def _split_strategy(dataset_config: dict[str, Any]) -> dict[str, int | str]:
    strategy = dict(dataset_config.get("split_strategy", {}))
    name = str(strategy.get("name", "balanced_by_class"))
    if name != "balanced_by_class":
        raise ValueError("Fashion-MNIST split_strategy.name must be balanced_by_class.")
    return {
        "name": name,
        "train_samples": int(strategy.get("train_samples", 9000)),
        "evaluation_samples": int(strategy.get("evaluation_samples", 1000)),
        "train_class_quota": int(strategy.get("train_class_quota", 900)),
        "evaluation_class_quota": int(strategy.get("evaluation_class_quota", 100)),
    }


def validate_fashion_mnist_dataset_config(dataset_config: dict[str, Any]) -> None:
    """Validate Fashion-MNIST dataset configuration before loading."""
    if str(dataset_config.get("name", "")).lower() != "fashion_mnist":
        raise ValueError("Fashion-MNIST evaluation requires dataset.name=fashion_mnist.")
    if str(dataset_config.get("domain", "")).lower() != "mnist_compatible":
        raise ValueError("Fashion-MNIST dataset.domain must be mnist_compatible.")
    image_shape = list(dataset_config.get("image_shape", [28, 28, 1]))
    if image_shape != [28, 28, 1]:
        raise ValueError("Fashion-MNIST image_shape must be [28, 28, 1].")
    _resolve_csv_path(dataset_config.get("csv_path"))
    _class_indices(dataset_config)
    strategy = _split_strategy(dataset_config)
    expected_train = int(strategy["train_class_quota"]) * 10
    expected_eval = int(strategy["evaluation_class_quota"]) * 10
    if int(strategy["train_samples"]) != expected_train:
        raise ValueError("Fashion-MNIST train_samples must match train_class_quota * 10.")
    if int(strategy["evaluation_samples"]) != expected_eval:
        raise ValueError(
            "Fashion-MNIST evaluation_samples must match evaluation_class_quota * 10."
        )


def split_fashion_mnist_balanced(
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    """Return balanced train/evaluation partitions and split metadata."""
    validate_fashion_mnist_dataset_config(dataset_config)
    images, labels = load_fashion_mnist_csv(dataset_config.get("csv_path"))
    class_order = _class_order(dataset_config)
    class_indices = _class_indices(dataset_config)
    strategy = _split_strategy(dataset_config)
    train_quota = int(strategy["train_class_quota"])
    eval_quota = int(strategy["evaluation_class_quota"])

    train_indices: list[int] = []
    evaluation_indices: list[int] = []
    candidate_counts: dict[str, int] = {}
    for class_name in class_order:
        label = class_indices[class_name]
        indices = np.flatnonzero(labels == label).astype(np.int64)
        candidate_counts[class_name] = int(len(indices))
        required = train_quota + eval_quota
        if len(indices) < required:
            raise ValueError(
                "Insufficient Fashion-MNIST rows for class {0}: required {1}, found {2}.".format(
                    class_name,
                    required,
                    len(indices),
                )
            )
        train_indices.extend(indices[:train_quota].tolist())
        evaluation_indices.extend(indices[train_quota : train_quota + eval_quota].tolist())

    train_set = set(train_indices)
    evaluation_set = set(evaluation_indices)
    if train_set.intersection(evaluation_set):
        raise ValueError("Fashion-MNIST train/evaluation split must not overlap.")

    metadata = {
        "name": "fashion_mnist",
        "domain": "mnist_compatible",
        "split": dataset_config.get("split", "test"),
        "csv_path": str(dataset_config.get("csv_path")),
        "split_strategy": strategy,
        "training_sample": {
            "split_name": "train",
            "method": "first_n_per_class",
            "per_class_start": 0,
            "per_class_end": train_quota,
            "class_quota": train_quota,
            "total_samples": int(strategy["train_samples"]),
        },
        "evaluation_sample": {
            "split_name": "evaluation",
            "method": "next_n_per_class",
            "per_class_start": train_quota,
            "per_class_end": train_quota + eval_quota,
            "class_quota": eval_quota,
            "total_samples": int(strategy["evaluation_samples"]),
            "excludes_training": True,
        },
        "image_shape": [28, 28, 1],
        "value_range": dataset_config.get("value_range", {"min": 0.0, "max": 1.0}),
        "class_order": class_order,
        "class_indices": class_indices,
        "class_quotas": {
            class_name: eval_quota for class_name in class_order
        },
        "candidate_counts": candidate_counts,
        "train_indices": train_indices,
        "evaluation_indices": evaluation_indices,
    }
    return {
        "train_images": images[train_indices],
        "train_labels": labels[train_indices],
        "train_labels_one_hot": _one_hot(labels[train_indices]),
        "evaluation_images": images[evaluation_indices],
        "evaluation_labels": labels[evaluation_indices],
        "evaluation_labels_one_hot": _one_hot(labels[evaluation_indices]),
        "metadata": metadata,
    }


def load_fashion_mnist_evaluation_split(
    dataset_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the balanced 1000-sample Fashion-MNIST evaluation split."""
    split = split_fashion_mnist_balanced(dataset_config)
    return (
        np.asarray(split["evaluation_images"], dtype=np.float32),
        np.asarray(split["evaluation_labels"], dtype=np.int64),
        dict(split["metadata"]),
    )


def load_fashion_mnist_training_split(
    dataset_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the balanced 9000-sample Fashion-MNIST training split."""
    split = split_fashion_mnist_balanced(dataset_config)
    return (
        np.asarray(split["train_images"], dtype=np.float32),
        np.asarray(split["train_labels_one_hot"], dtype=np.float32),
        dict(split["metadata"]),
    )
