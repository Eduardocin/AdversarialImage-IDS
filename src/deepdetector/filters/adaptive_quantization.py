"""Adaptive scalar quantization for normalized images."""

from __future__ import print_function

from typing import Any, Dict, Tuple

import numpy as np

from .entropy import one_d_entropy
from .quantization import scalar_quantization


def entropy_based_quantization(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quantize an image using an entropy-selected interval."""
    entropy = one_d_entropy(image)

    if entropy < 4.0:
        entropy_range = "low"
        interval = 128
    elif entropy < 5.0:
        entropy_range = "mid"
        interval = 64
    else:
        entropy_range = "high"
        interval = 43

    quantized = scalar_quantization(image, interval=interval, left=True)
    metadata = {
        "entropy": float(entropy),
        "range": entropy_range,
        "interval_used": int(interval),
    }
    return quantized, metadata
