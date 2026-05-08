"""Feature extraction functions migrated from the DeepDetector notebooks."""

from __future__ import annotations

import numpy as np


def _entropy_from_uint8_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.int16)
    if values.size == 0:
        raise ValueError("Cannot compute entropy for an empty array.")
    if values.min() < 0 or values.max() > 255:
        raise ValueError("Entropy inputs must map to uint8 values in [0, 255].")

    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = counts / float(values.size)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def one_d_entropy_mnist(image: np.ndarray) -> float:
    """Return the legacy 1D entropy for a normalized MNIST image.

    This mirrors ``Uniform_vs_NonUniform_Quantization.oneDEntropy``: the
    normalized image is scaled by 255, cast to int16, and histogrammed.
    """

    uint8_values = np.asarray(image, dtype=np.float32) * 255.0
    return _entropy_from_uint8_values(uint8_values.astype(np.int16))


def one_d_entropy_uint8_chw(image: np.ndarray) -> float:
    """Return average per-channel entropy for a CHW image in [0, 255].

    This mirrors the ImageNet notebooks, which compute one entropy per RGB/BGR
    channel and average the three channel entropies.
    """

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(f"Expected a CHW image with 3 dimensions, got {image_array.shape}.")

    channel_entropies = [
        _entropy_from_uint8_values(channel.astype(np.int16)) for channel in image_array
    ]
    return float(np.mean(channel_entropies))
