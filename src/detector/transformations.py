"""Image transformations migrated from the DeepDetector notebooks."""

from __future__ import annotations

import numpy as np


def normalize_values(
    image: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """Clip image values to a fixed range."""

    if inplace:
        np.clip(image, min_value, max_value, out=image)
        return image
    return np.clip(np.asarray(image), min_value, max_value)


def scalar_quantization_uint8(
    image: np.ndarray,
    interval: int,
    *,
    left: bool = True,
) -> np.ndarray:
    """Apply legacy scalar quantization to an image already in [0, 255]."""

    if interval <= 0:
        raise ValueError("interval must be positive.")

    quantized = np.array(image, dtype=np.float32)
    quantized //= interval
    quantized *= interval
    if not left:
        quantized += interval // 2
    return quantized


def scalar_quantization_mnist(
    image: np.ndarray,
    interval: int,
    *,
    left: bool = True,
) -> np.ndarray:
    """Apply legacy MNIST quantization to a normalized image in [0, 1]."""

    if interval <= 0:
        raise ValueError("interval must be positive.")

    quantized = np.asarray(image, dtype=np.float32) * 255.0
    quantized //= interval
    quantized *= interval
    if not left:
        quantized += interval // 2
    quantized /= 255.0
    return quantized


def find_nonuniform_border_mnist(image: np.ndarray, cutoff_count: int = 392) -> int:
    """Find the legacy non-uniform MNIST quantization border.

    The original notebook used ``plt.hist(image.flatten(), bins=256)`` and then
    worked with histogram bin indexes. This implementation keeps that bin-index
    behavior without creating figures as a side effect.
    """

    counts, _ = np.histogram(np.asarray(image).ravel(), bins=256)
    running_count = 0.0
    border = 0
    for index, count in enumerate(counts):
        running_count += count
        if running_count >= cutoff_count:
            border = index + 1
            break

    for index in range(border, 256):
        if counts[index] > 0:
            return index
    return border


def nonuniform_quantization_mnist(image: np.ndarray) -> np.ndarray:
    """Apply the legacy non-uniform quantization used in the MNIST notebook."""

    border = find_nonuniform_border_mnist(image)
    quantized = np.asarray(image, dtype=np.float32) * 255.0
    quantized[quantized <= border] = 0.0
    quantized[quantized > border] = float(border)
    quantized /= 255.0
    return quantized


def box_mean_filter_chw(
    image: np.ndarray,
    start: int,
    end: int,
    coefficient: int,
) -> np.ndarray:
    """Apply the legacy box mean filter to a CHW image."""

    image_array = np.asarray(image)
    filtered = np.array(image_array, dtype=np.float32)
    for channel in range(image_array.shape[0]):
        for row in range(start, end):
            for col in range(start, end):
                window = image_array[
                    channel,
                    row - start : row + start + 1,
                    col - start : col + start + 1,
                ]
                filtered[channel][row][col] = np.sum(window) / coefficient
    return filtered


def diamond_mean_filter_chw(
    image: np.ndarray,
    kernel: np.ndarray,
    start: int,
    end: int,
    coefficient: int,
) -> np.ndarray:
    """Apply the legacy diamond mean filter to a CHW image."""

    image_array = np.asarray(image)
    kernel_array = np.asarray(kernel)
    filtered = np.array(image_array, dtype=np.float32)
    for channel in range(image_array.shape[0]):
        for row in range(start, end):
            for col in range(start, end):
                window = image_array[
                    channel,
                    row - start : row + start + 1,
                    col - start : col + start + 1,
                ]
                filtered[channel][row][col] = np.sum(window * kernel_array) / coefficient
    return filtered


def cross_mean_filter_chw(
    image: np.ndarray,
    start: int,
    end: int,
    coefficient: int,
) -> np.ndarray:
    """Apply the legacy cross mean filter to a CHW image."""

    image_array = np.asarray(image)
    filtered = np.array(image_array, dtype=np.float32)
    for channel in range(image_array.shape[0]):
        for row in range(start, end):
            for col in range(start, end):
                total = image_array[channel][row][col]
                for offset in range(1, start + 1):
                    total += image_array[channel][row - offset][col]
                    total += image_array[channel][row + offset][col]
                    total += image_array[channel][row][col - offset]
                    total += image_array[channel][row][col + offset]
                filtered[channel][row][col] = total / coefficient
    return filtered


def choose_closer_filter(
    original_image: np.ndarray,
    first_filtered: np.ndarray,
    second_filtered: np.ndarray,
) -> np.ndarray:
    """Choose, pixel by pixel, the filtered value closer to the original image."""

    original = np.asarray(original_image)
    first = np.asarray(first_filtered)
    second = np.asarray(second_filtered)
    if original.shape != first.shape or original.shape != second.shape:
        raise ValueError("All inputs must have the same shape.")

    first_distance = np.abs(first - original)
    second_distance = np.abs(second - original)
    return np.where(first_distance < second_distance, first, second)
