from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.imagenet import load_imagenet_images, resize_normalized_image


def test_load_imagenet_images_skips_missing_and_corrupt_files(tmp_path) -> None:
    """Check ImageNet CSV loading with unreadable files."""
    Image = pytest.importorskip("PIL.Image")

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image = np.full((10, 12, 3), 128, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(str(images_dir / "valid.JPEG"))
    (images_dir / "corrupt.JPEG").write_text("not an image", encoding="utf-8")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "valid.JPEG,3\nmissing.JPEG,4\ncorrupt.JPEG,5\n",
        encoding="utf-8",
    )

    rng = np.random.RandomState(0)
    images, labels = load_imagenet_images(
        csv_path=str(csv_path),
        images_dir=str(images_dir),
        n_samples=3,
        preprocess_fn=lambda array: resize_normalized_image(array, image_size=8),
        rng=rng,
    )

    assert images.shape == (1, 8, 8, 3)
    assert images.dtype == np.float32
    assert labels.shape == (1,)
    assert labels.dtype == np.int32
    assert int(labels[0]) == 3
    assert images.min() >= 0.0
    assert images.max() <= 1.0
