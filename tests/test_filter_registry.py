from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.registry import FILTER_REGISTRY


EXPECTED_FILTER_NAMES = [
    "scalar_128",
    "scalar_64",
    "scalar_43",
    "nonuniform",
    "entropy_adaptive",
    "box_3",
    "box_5",
    "cross_3",
    "diamond_3",
]


def test_filter_registry_contains_exactly_expected_filters():
    assert list(FILTER_REGISTRY.keys()) == EXPECTED_FILTER_NAMES


def test_filter_registry_entries_are_callable_image_filters():
    image = np.random.rand(28, 28, 1).astype(np.float32)

    for filter_fn in FILTER_REGISTRY.values():
        output = filter_fn(image)
        assert output.shape == image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
