"""Adaptive noise reduction used by the MNIST detector."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import DetectionConfig


def scalar_quantization(image: np.ndarray, interval: int, *, left: bool = True, offset: float = 0.0) -> np.ndarray:
    """Apply the scalar quantization used by the DeepDetector MNIST scripts."""

    quantized = (image + offset) * 255.0
    quantized //= interval
    quantized *= interval
    if not left:
        quantized += interval // 2
    quantized /= 255.0
    quantized -= offset
    return quantized.astype(np.float32)


def one_d_entropy(image: np.ndarray, *, offset: float = 0.0) -> float:
    """Compute the 1D pixel histogram entropy used by the adaptive filter."""

    expanded = np.array((image + offset) * 255.0, dtype=np.int16)
    expanded = np.clip(expanded, 0, 255)
    frequencies = np.zeros(256, dtype=np.float64)
    for row in range(28):
        for col in range(28):
            frequencies[expanded[row][col]] += 1
    frequencies /= 784.0

    entropy = 0.0
    for value in frequencies:
        if value > 0:
            entropy += value * math.log(value, 2)
    return -entropy


def cross_mean_filter(
    image: np.ndarray,
    start: int,
    end: int,
    coefficient: int,
) -> np.ndarray:
    """Apply the cross mean filter from the MNIST reference scripts."""

    filtered = np.array(image, dtype=np.float32)
    for row in range(start, end):
        for col in range(start, end):
            total = image[row][col]
            for distance in range(1, start + 1):
                total += image[row - distance][col]
                total += image[row + distance][col]
                total += image[row][col - distance]
                total += image[row][col + distance]
            filtered[row][col] = total / coefficient
    return filtered


def choose_closer_filter(
    original: np.ndarray,
    filter_data1: np.ndarray,
    filter_data2: np.ndarray,
) -> np.ndarray:
    """Choose per-pixel values closer to the original image."""

    distance1 = np.abs(filter_data1 - original)
    distance2 = np.abs(filter_data2 - original)
    return np.where(distance1 < distance2, filter_data1, filter_data2).astype(np.float32)


def adaptive_reduce(
    image: np.ndarray,
    config: Optional[DetectionConfig] = None,
    *,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply the MNIST adaptive quantization/filtering rule."""

    cfg = config or DetectionConfig()
    entropy = one_d_entropy(image, offset=offset)
    if entropy < cfg.low_entropy_threshold:
        return scalar_quantization(image, cfg.low_entropy_interval, offset=offset)
    if entropy < cfg.mid_entropy_threshold:
        return scalar_quantization(image, cfg.mid_entropy_interval, offset=offset)

    quantized = scalar_quantization(image, cfg.high_entropy_interval, offset=offset)
    filtered = cross_mean_filter(quantized, cfg.cross_start, cfg.cross_end, cfg.cross_coefficient)
    return choose_closer_filter(image, quantized, filtered)
