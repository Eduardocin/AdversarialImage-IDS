"""ImageNet image loading utilities."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)

PreprocessFn = Callable[[np.ndarray], np.ndarray]


def resize_normalized_image(image: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Resize one RGB image while keeping normalized float32 pixel values."""
    from PIL import Image

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected image shape (H, W, 3).")

    clipped = np.clip(image_array, 0.0, 1.0)
    pil_image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
    resized = pil_image.resize((image_size, image_size), Image.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32)


def _read_rgb_image(path: Path) -> np.ndarray:
    """Load one image as normalized RGB float32 data."""
    if not path.is_file():
        raise FileNotFoundError(str(path))

    from PIL import Image

    with Image.open(str(path)) as image:
        rgb_image = image.convert("RGB")
        return (np.asarray(rgb_image, dtype=np.float32) / 255.0).astype(np.float32)


def _read_rows_with_pandas(csv_path: Path) -> List[Tuple[str, int]]:
    """Read CSV rows with pandas when the installed package is usable."""
    import pandas as pd

    frame = pd.read_csv(
        str(csv_path),
        header=None,
        names=["filename", "label_index"],
        dtype={"filename": str, "label_index": np.int32},
    )
    return [
        (str(row.filename), int(row.label_index))
        for row in frame.itertuples(index=False)
    ]


def _read_rows_with_csv(csv_path: Path) -> List[Tuple[str, int]]:
    """Read ImageNet CSV rows with the standard library."""
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 2:
                raise ValueError("Expected two columns in {0}: {1}".format(csv_path, row))
            rows.append((str(row[0]), int(row[1])))
    return rows


def _shuffled_rows(csv_path: Path, rng: np.random.RandomState) -> List[Tuple[str, int]]:
    """Read a two-column CSV and return rows in a deterministic random order."""
    try:
        rows = _read_rows_with_pandas(csv_path)
    except ImportError as exc:
        logger.warning("pandas import failed; reading CSV with stdlib: %s", exc)
        rows = _read_rows_with_csv(csv_path)

    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def load_imagenet_images(
    csv_path: str,
    images_dir: str,
    n_samples: int,
    preprocess_fn: PreprocessFn,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ImageNet images listed in a CSV index.

    The CSV must have two columns without a header: filename and integer label.
    Images are loaded as RGB arrays in NHWC layout with normalized pixel values
    before ``preprocess_fn`` is applied. If ``preprocess_fn`` changes the layout,
    the returned batch keeps that layout for every loaded sample.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if preprocess_fn is None:
        raise ValueError("preprocess_fn must be callable.")
    if rng is None:
        raise ValueError("rng must be a numpy RandomState.")

    csv_path_obj = Path(csv_path)
    images_dir_obj = Path(images_dir)
    rows = _shuffled_rows(csv_path_obj, rng)[:n_samples]

    images = []
    labels = []
    expected_shape = None

    for filename, label_index in rows:
        image_path = images_dir_obj / str(filename)
        try:
            image = _read_rgb_image(image_path)
        except FileNotFoundError:
            logger.warning("ImageNet image not found: %s", image_path)
            continue
        except OSError as exc:
            logger.warning("Skipping unreadable ImageNet image %s: %s", image_path, exc)
            continue

        processed = np.asarray(preprocess_fn(image), dtype=np.float32)
        if expected_shape is None:
            expected_shape = processed.shape
        elif processed.shape != expected_shape:
            raise ValueError(
                "preprocess_fn returned inconsistent shapes: {0} and {1}".format(
                    expected_shape,
                    processed.shape,
                )
            )

        images.append(processed)
        labels.append(int(label_index))

    if len(images) < n_samples:
        logger.warning(
            "Loaded %s ImageNet images out of the requested %s samples.",
            len(images),
            n_samples,
        )

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )
