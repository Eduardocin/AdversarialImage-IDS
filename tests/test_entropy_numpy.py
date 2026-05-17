from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.filters.entropy import one_d_entropy


def test_constant_image_entropy_is_zero():
    img = np.zeros((28, 28, 1), dtype=np.float32)

    assert one_d_entropy(img) == 0.0


def test_entropy_nonnegative():
    img = np.random.rand(28, 28, 1).astype(np.float32)

    assert one_d_entropy(img) >= 0.0


def test_entropy_upper_bound():
    img = np.random.rand(28, 28, 1).astype(np.float32)

    assert one_d_entropy(img) <= 8.0


def test_adaptive_output_shape_preserved():
    img = np.random.rand(28, 28, 1).astype(np.float32)

    out, _ = entropy_based_quantization(img)

    assert out.shape == img.shape


def test_adaptive_output_in_range():
    img = np.random.rand(28, 28, 1).astype(np.float32)

    out, _ = entropy_based_quantization(img)

    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_adaptive_metadata_keys():
    img = np.random.rand(28, 28, 1).astype(np.float32)

    _, meta = entropy_based_quantization(img)

    assert set(meta.keys()) == {"entropy", "range", "interval_used"}


def test_low_entropy_uses_interval_128():
    img = np.zeros((28, 28, 1), dtype=np.float32)
    img[0, 0, 0] = 1.0

    _, meta = entropy_based_quantization(img)

    assert meta["interval_used"] == 128
