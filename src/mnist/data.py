"""MNIST loading and normalization helpers."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np

from .config import MnistPaths, MnistSplitConfig

MNIST_IDX_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

DEFAULT_MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"


def clip_m1_images(images: np.ndarray) -> np.ndarray:
    """Clip M1/CleverHans image data to the [0, 1] range."""

    return np.clip(images, 0.0, 1.0)


def clip_m2_images(images: np.ndarray) -> np.ndarray:
    """Clip M2/Carlini image data to the [-0.5, 0.5] range."""

    return np.clip(images, -0.5, 0.5)


def to_m2_range(images: np.ndarray) -> np.ndarray:
    """Convert [0, 1] MNIST images to Carlini's [-0.5, 0.5] range."""

    return images.astype(np.float32) - 0.5


def to_m1_range(images: np.ndarray) -> np.ndarray:
    """Convert Carlini's [-0.5, 0.5] images to the [0, 1] range."""

    return images.astype(np.float32) + 0.5


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Return one-hot labels as float32 arrays."""

    return (np.arange(num_classes) == labels[:, None]).astype(np.float32)


def extract_images(path: Path, num_images: int, *, m2_range: bool = False) -> np.ndarray:
    """Read MNIST IDX image gzip data."""

    with gzip.open(path, "rb") as bytestream:
        bytestream.read(16)
        buffer = bytestream.read(num_images * 28 * 28)
    images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    images = images.reshape(num_images, 28, 28, 1) / 255.0
    if m2_range:
        images -= 0.5
    return images


def extract_labels(path: Path, num_images: int) -> np.ndarray:
    """Read MNIST IDX label gzip data as one-hot vectors."""

    with gzip.open(path, "rb") as bytestream:
        bytestream.read(8)
        buffer = bytestream.read(num_images)
    return one_hot(np.frombuffer(buffer, dtype=np.uint8))


def missing_idx_files(directory: Path) -> List[str]:
    """Return MNIST IDX files missing from a directory."""

    return [name for name in MNIST_IDX_FILES if not (directory / name).exists()]


def download_idx_files(target_dir: Path, mirror: str = DEFAULT_MNIST_MIRROR) -> List[Path]:
    """Download MNIST IDX files to a local directory."""

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in MNIST_IDX_FILES:
        target = target_dir / name
        if not target.exists():
            urlretrieve(f"{mirror.rstrip('/')}/{name}", target)
        downloaded.append(target)
    return downloaded


def copy_idx_files(source_dir: Path, target_dir: Path) -> List[Path]:
    """Copy MNIST IDX files without changing their contents."""

    missing = missing_idx_files(source_dir)
    if missing:
        raise FileNotFoundError(f"MNIST IDX files missing in {source_dir}: {', '.join(missing)}")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in MNIST_IDX_FILES:
        target = target_dir / name
        shutil.copy2(source_dir / name, target)
        copied.append(target)
    return copied


def ensure_carlini_mnist_data(paths: MnistPaths, source_dir: Optional[Path] = None) -> List[Path]:
    """Ensure Carlini's code can read MNIST from its local data directory."""

    if not missing_idx_files(paths.carlini_data_dir):
        return [paths.carlini_data_dir / name for name in MNIST_IDX_FILES]
    source = source_dir or Path(tempfile.gettempdir())
    return copy_idx_files(source, paths.carlini_data_dir)


def load_mnist_from_idx(directory: Path, *, m2_range: bool = False) -> Tuple[np.ndarray, ...]:
    """Load train/test MNIST arrays from IDX gzip files."""

    missing = missing_idx_files(directory)
    if missing:
        raise FileNotFoundError(f"MNIST IDX files missing in {directory}: {', '.join(missing)}")

    train_images = extract_images(directory / "train-images-idx3-ubyte.gz", 60000, m2_range=m2_range)
    train_labels = extract_labels(directory / "train-labels-idx1-ubyte.gz", 60000)
    test_images = extract_images(directory / "t10k-images-idx3-ubyte.gz", 10000, m2_range=m2_range)
    test_labels = extract_labels(directory / "t10k-labels-idx1-ubyte.gz", 10000)
    return train_images, train_labels, test_images, test_labels


def detector_test_slice(
    images: np.ndarray,
    labels: np.ndarray,
    splits: MnistSplitConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the original test slice used for final detector evaluation."""

    return images[splits.detector_test_start :], labels[splits.detector_test_start :]


def generate_untargeted_data(
    test_data: np.ndarray,
    test_labels: np.ndarray,
    samples: int,
    start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match Carlini's untargeted MNIST input/target generation."""

    inputs = []
    targets = []
    for index in range(samples):
        inputs.append(test_data[start + index])
        targets.append(test_labels[start + index])
    return np.array(inputs), np.array(targets)


def iter_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield contiguous numpy batches."""

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        yield images[start:end], labels[start:end]
