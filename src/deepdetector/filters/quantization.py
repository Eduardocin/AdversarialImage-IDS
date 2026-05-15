"""NumPy quantization filters faithful to the DeepDetector MNIST logic."""

from __future__ import annotations

import numpy as np


MNIST_HALF_PIXELS = 392


def normalize_image_range(image_data: np.ndarray) -> np.ndarray:
    """Clip image values to the normalized pixel range [0, 1]."""
    return np.clip(np.asarray(image_data, dtype=np.float32), 0.0, 1.0)


def scalar_quantization(input_digit: np.ndarray, interval: int, left: bool = True) -> np.ndarray:
    """Apply the legacy scalar quantization rule on normalized image data."""
    if interval <= 0:
        raise ValueError("interval must be positive.")

    ret = np.asarray(input_digit, dtype=np.float32) * 255.0
    ret //= interval
    ret *= interval
    if not left:
        ret += interval // 2
    ret /= 255.0
    return normalize_image_range(ret)


def find_border(image: np.ndarray) -> int:
    """Find the non-uniform quantization border for a single MNIST image."""
    image_array = normalize_image_range(image)
    if image_array.ndim == 4:
        raise ValueError("find_border expects a single image, not a batch.")

    counts, _ = np.histogram(image_array.ravel(), bins=256, range=(0.0, 1.0))
    running_count = 0
    border = 0
    for index, count in enumerate(counts):
        running_count += int(count)
        if running_count >= MNIST_HALF_PIXELS:
            border = index + 1
            break

    for index in range(border, 256):
        if counts[index] > 0:
            return index
    return border


def _nonuniform_quantization_single(image: np.ndarray) -> np.ndarray:
    """Apply non-uniform quantization to one image."""
    border = find_border(image)
    quantized = normalize_image_range(image) * 255.0
    quantized[quantized <= border] = 0.0
    quantized[quantized > border] = float(border)
    quantized /= 255.0
    return normalize_image_range(quantized)


def nonuniform_quantization(image: np.ndarray) -> np.ndarray:
    """Apply legacy non-uniform quantization to an image or a batch."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        quantized = [_nonuniform_quantization_single(single) for single in image_array]
        return np.asarray(quantized, dtype=np.float32).reshape(image_array.shape)
    return _nonuniform_quantization_single(image_array).reshape(image_array.shape)
