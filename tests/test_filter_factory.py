from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.factory import build_filter_from_config  # noqa: E402


def test_factory_builds_cross_mean_filter() -> None:
    """Cross filters should normalize radius-derived metadata."""
    name, filter_fn, metadata = build_filter_from_config(
        {"name": "cross_7x7", "type": "cross_mean", "radius": 3}
    )

    assert name == "cross_7x7"
    assert metadata == {
        "filter_name": "cross_7x7",
        "filter_type": "cross_mean",
        "mask_type": "cross",
        "mask_size": 7,
        "radius": 3,
        "kernel_size": None,
    }
    output = filter_fn(np.ones((5, 5, 1), dtype=np.float32))
    assert output.shape == (5, 5, 1)


def test_factory_builds_diamond_mean_filter() -> None:
    """Diamond filters should normalize radius-derived metadata."""
    name, filter_fn, metadata = build_filter_from_config(
        {"name": "diamond_5x5", "type": "diamond_mean", "radius": 2}
    )

    assert name == "diamond_5x5"
    assert metadata["filter_type"] == "diamond_mean"
    assert metadata["mask_type"] == "diamond"
    assert metadata["mask_size"] == 5
    assert metadata["radius"] == 2
    assert metadata["kernel_size"] is None
    assert filter_fn(np.ones((5, 5, 1), dtype=np.float32)).shape == (5, 5, 1)


def test_factory_builds_box_mean_filter() -> None:
    """Box filters should normalize kernel-size-derived metadata."""
    name, filter_fn, metadata = build_filter_from_config(
        {"name": "box_5x5", "type": "box_mean", "kernel_size": 5}
    )

    assert name == "box_5x5"
    assert metadata["filter_type"] == "box_mean"
    assert metadata["mask_type"] == "box"
    assert metadata["mask_size"] == 5
    assert metadata["radius"] is None
    assert metadata["kernel_size"] == 5
    assert filter_fn(np.ones((5, 5, 1), dtype=np.float32)).shape == (5, 5, 1)


def test_factory_rejects_unknown_filter_type() -> None:
    """Unknown filter types should fail explicitly."""
    with pytest.raises(ValueError, match="Unknown filter type"):
        build_filter_from_config({"name": "mystery", "type": "unknown"})


def test_factory_rejects_missing_name() -> None:
    """Every configured filter needs a stable output name."""
    with pytest.raises(ValueError, match="name"):
        build_filter_from_config({"type": "cross_mean", "radius": 1})


def test_factory_rejects_invalid_radius() -> None:
    """Radius-based filters require positive integer radius values."""
    with pytest.raises(ValueError, match="radius"):
        build_filter_from_config({"name": "cross_bad", "type": "cross_mean", "radius": 0})


def test_factory_rejects_even_kernel_size() -> None:
    """Box filters require odd kernel sizes."""
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        build_filter_from_config({"name": "box_bad", "type": "box_mean", "kernel_size": 4})


def test_factory_builds_adaptive_quantization_filter() -> None:
    """Split experiments should configure adaptive quantization via the factory."""
    name, filter_fn, metadata = build_filter_from_config(
        {"name": "adaptive_quantization", "type": "adaptive_quantization"}
    )

    assert name == "adaptive_quantization"
    assert metadata == {
        "filter_name": "adaptive_quantization",
        "filter_type": "adaptive_quantization",
    }
    output = filter_fn(np.zeros((4, 4, 1), dtype=np.float32))
    assert output.shape == (4, 4, 1)


def test_factory_builds_proposed_detection_filter() -> None:
    """Split experiments should configure the proposed final filter via the factory."""
    name, filter_fn, metadata = build_filter_from_config(
        {"name": "proposed_detection_filter", "type": "proposed_detection_filter"}
    )

    assert name == "proposed_detection_filter"
    assert metadata == {
        "filter_name": "proposed_detection_filter",
        "filter_type": "proposed_detection_filter",
    }
    output = filter_fn(np.zeros((4, 4, 1), dtype=np.float32))
    assert output.shape == (4, 4, 1)
