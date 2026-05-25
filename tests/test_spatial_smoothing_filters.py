from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.spatial_smoothing import (
    build_box_mask,
    build_cross_mask,
    build_diamond_mask,
    spatial_smoothing_filter,
)
from deepdetector.filters.table7_filters import table7_filter


def test_build_cross_mask_5_has_nine_active_positions() -> None:
    mask = build_cross_mask(5)

    assert mask.shape == (5, 5)
    assert int(mask.sum()) == 9


def test_build_diamond_mask_5_uses_manhattan_distance() -> None:
    mask = build_diamond_mask(5)
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(mask, expected)


def test_build_box_mask_5_has_twenty_five_active_positions() -> None:
    mask = build_box_mask(5)

    assert mask.shape == (5, 5)
    assert int(mask.sum()) == 25


def test_spatial_smoothing_filter_preserves_shape() -> None:
    image = np.random.RandomState(0).rand(3, 8, 9).astype(np.float32) * 255.0

    output = spatial_smoothing_filter(image, mask_type="cross", size=3)

    assert output.shape == image.shape


def test_spatial_smoothing_filter_preserves_float32_dtype() -> None:
    image = np.random.RandomState(1).rand(3, 8, 9).astype(np.float32) * 255.0

    output = spatial_smoothing_filter(image, mask_type="diamond", size=5)

    assert output.dtype == np.float32


def test_spatial_smoothing_filter_preserves_spatial_borders() -> None:
    image = np.arange(3 * 6 * 7, dtype=np.float32).reshape((3, 6, 7))

    output = spatial_smoothing_filter(image, mask_type="box", size=5)

    np.testing.assert_array_equal(output[:, :2, :], image[:, :2, :])
    np.testing.assert_array_equal(output[:, -2:, :], image[:, -2:, :])
    np.testing.assert_array_equal(output[:, :, :2], image[:, :, :2])
    np.testing.assert_array_equal(output[:, :, -2:], image[:, :, -2:])


def test_table7_filter_preserves_shape_and_255_range() -> None:
    image = np.random.RandomState(2).rand(3, 8, 9).astype(np.float32) * 255.0

    output = table7_filter(image, mask_type="box", size=3)

    assert output.shape == image.shape
    assert output.dtype == np.float32
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 255.0


def test_table7_filter_does_not_apply_scalar_quantization() -> None:
    image = np.full((3, 5, 5), 42.0, dtype=np.float32)

    output = table7_filter(image, mask_type="box", size=3)

    np.testing.assert_array_equal(output, image)


def test_spatial_smoothing_filter_rejects_invalid_mask_type() -> None:
    image = np.zeros((3, 8, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="mask_type"):
        spatial_smoothing_filter(image, mask_type="triangle", size=3)


def test_spatial_smoothing_filter_rejects_even_size() -> None:
    image = np.zeros((3, 8, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="odd"):
        spatial_smoothing_filter(image, mask_type="cross", size=4)
