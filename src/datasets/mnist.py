"""MNIST dataset utilities for the modern PyTorch pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

MnistSplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class MnistSplitSpec:
    image_key: str
    label_key: str
    start_index: int
    end_index: int

    @property
    def expected_count(self) -> int:
        return self.end_index - self.start_index + 1


MNIST_SPLIT_SPECS: dict[str, MnistSplitSpec] = {
    "train": MnistSplitSpec("x_train", "y_train", 0, 4499),
    "validation": MnistSplitSpec("x_validation", "y_validation", 4500, 5499),
    "test": MnistSplitSpec("x_test", "y_test", 5500, 9999),
}

MNIST_SPLIT_KEYS = {
    split: (spec.image_key, spec.label_key) for split, spec in MNIST_SPLIT_SPECS.items()
}


@dataclass(frozen=True)
class MnistArrays:
    images: np.ndarray
    labels: np.ndarray
    spec: MnistSplitSpec


def load_mnist_npz(
    dataset_path: str | Path,
    split: MnistSplitName,
    *,
    validate_expected_count: bool = True,
) -> MnistArrays:
    """Load one split from the repository's prepared MNIST npz file."""

    if split not in MNIST_SPLIT_SPECS:
        valid_splits = ", ".join(MNIST_SPLIT_SPECS)
        raise ValueError(f"Unknown split {split!r}; expected one of: {valid_splits}.")

    spec = MNIST_SPLIT_SPECS[split]
    with np.load(Path(dataset_path)) as dataset:
        images = np.asarray(dataset[spec.image_key])
        labels = np.asarray(dataset[spec.label_key])

    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Unexpected MNIST image shape for {split}: {images.shape}.")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"Unexpected MNIST labels shape for {split}: {labels.shape}; "
            f"expected ({images.shape[0]},)."
        )
    if validate_expected_count and images.shape[0] != spec.expected_count:
        raise ValueError(
            f"Unexpected MNIST {split} count: {images.shape[0]}; "
            f"expected {spec.expected_count} for original indices "
            f"{spec.start_index}-{spec.end_index}."
        )

    return MnistArrays(
        images=images,
        labels=labels.astype(np.int64, copy=False),
        spec=spec,
    )


class MnistNpzDataset:
    """Torch-compatible dataset backed by ``scripts/download_mnist_test.py`` output."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: MnistSplitName,
        *,
        normalize: bool = True,
        add_channel_dim: bool = True,
        validate_expected_count: bool = True,
    ) -> None:
        arrays = load_mnist_npz(
            dataset_path,
            split,
            validate_expected_count=validate_expected_count,
        )
        self.images = arrays.images
        self.labels = arrays.labels
        self.spec = arrays.spec
        self.normalize = normalize
        self.add_channel_dim = add_channel_dim

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def original_index(self, index: int) -> int:
        return self.spec.start_index + index

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32, copy=False)
        label = int(self.labels[index])

        if self.normalize:
            image = image / 255.0
        if self.add_channel_dim:
            image = image[np.newaxis, :, :]

        try:
            import torch
        except ModuleNotFoundError:
            return image, label

        return torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(label).long()
