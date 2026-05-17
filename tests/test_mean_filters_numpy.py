from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.mean_filters import (
    box_mean_filter,
    cross_mean_filter,
    diamond_mean_filter,
)


def test_box_mean_filter_preserves_shape_and_range():
    image = np.random.rand(28, 28, 1).astype(np.float32)

    output = box_mean_filter(image, kernel_size=3)

    assert output.shape == image.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_cross_mean_filter_preserves_shape_and_range():
    image = np.random.rand(28, 28, 1).astype(np.float32)

    output = cross_mean_filter(image, radius=1)

    assert output.shape == image.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_diamond_mean_filter_preserves_shape_and_range():
    image = np.random.rand(28, 28, 1).astype(np.float32)

    output = diamond_mean_filter(image, radius=1)

    assert output.shape == image.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_mean_filters_keep_constant_image_unchanged():
    image = np.full((28, 28, 1), 0.37, dtype=np.float32)

    for filter_fn in (box_mean_filter, cross_mean_filter, diamond_mean_filter):
        output = filter_fn(image)
        np.testing.assert_allclose(output, image, atol=1e-6)


def test_box_mean_filter_uses_reflect_padding_at_border():
    image = np.zeros((28, 28, 1), dtype=np.float32)
    image[0, :, 0] = 1.0

    output = box_mean_filter(image, kernel_size=3)

    np.testing.assert_allclose(output[0, 0, 0], 1.0 / 3.0)
