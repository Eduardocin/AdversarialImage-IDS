"""Local mean filters for MNIST grayscale images."""

from __future__ import print_function

import numpy as np


MNIST_IMAGE_SHAPE = (28, 28, 1)


def _as_mnist_image(image: np.ndarray) -> np.ndarray:
    """Return a single MNIST image as float32 data."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.shape != MNIST_IMAGE_SHAPE:
        raise ValueError("Expected image shape (28, 28, 1).")
    return image_array


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return a float32 kernel whose values sum to one."""
    kernel_array = np.asarray(kernel, dtype=np.float32)
    kernel_sum = float(kernel_array.sum())
    if kernel_sum <= 0.0:
        raise ValueError("kernel must contain at least one positive value.")
    return kernel_array / kernel_sum


def _validate_positive_odd(value: int, name: str) -> None:
    """Validate a positive odd integer parameter."""
    if value <= 0:
        raise ValueError("{0} must be positive.".format(name))
    if value % 2 == 0:
        raise ValueError("{0} must be odd.".format(name))


def _validate_positive_radius(radius: int) -> None:
    """Validate a positive mask radius."""
    if radius <= 0:
        raise ValueError("radius must be positive.")


def _convolve_reflect(image_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution with reflect padding."""
    kernel_array = _normalize_kernel(kernel)
    kernel_height, kernel_width = kernel_array.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded = np.pad(
        np.asarray(image_2d, dtype=np.float32),
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="reflect",
    )

    output = np.empty_like(image_2d, dtype=np.float32)
    height, width = image_2d.shape
    for row in range(height):
        for col in range(width):
            window = padded[row : row + kernel_height, col : col + kernel_width]
            output[row, col] = float(np.sum(window * kernel_array))
    return output


def _cross_convolve_reflect(image_2d: np.ndarray, radius: int) -> np.ndarray:
    """Apply a cross-mask mean filter with reflect padding."""
    _validate_positive_radius(radius)
    padded = np.pad(
        np.asarray(image_2d, dtype=np.float32),
        ((radius, radius), (radius, radius)),
        mode="reflect",
    )
    height, width = image_2d.shape
    output = np.zeros_like(image_2d, dtype=np.float32)

    for offset in range(-radius, radius + 1):
        output += padded[
            radius + offset : radius + offset + height,
            radius : radius + width,
        ]
        if offset != 0:
            output += padded[
                radius : radius + height,
                radius + offset : radius + offset + width,
            ]

    output /= float(4 * radius + 1)
    return output


def _clip_and_restore_shape(image_2d: np.ndarray) -> np.ndarray:
    """Clip a 2D filtered image and restore MNIST channel shape."""
    return np.clip(image_2d, 0.0, 1.0).astype(np.float32).reshape(MNIST_IMAGE_SHAPE)


def _cross_kernel(radius: int) -> np.ndarray:
    """Build a normalized cross mask before normalization."""
    _validate_positive_radius(radius)
    size = 2 * radius + 1
    center = radius
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[center, :] = 1.0
    kernel[:, center] = 1.0
    return kernel


def _diamond_kernel(radius: int) -> np.ndarray:
    """Build a normalized diamond mask before normalization."""
    _validate_positive_radius(radius)
    size = 2 * radius + 1
    center = radius
    kernel = np.zeros((size, size), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            if abs(row - center) + abs(col - center) <= radius:
                kernel[row, col] = 1.0
    return kernel


def _box_kernel(kernel_size: int) -> np.ndarray:
    """Build a square mask before normalization."""
    _validate_positive_odd(kernel_size, "kernel_size")
    return np.ones((kernel_size, kernel_size), dtype=np.float32)


def box_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a square local mean filter to a grayscale image."""
    image_array = _as_mnist_image(image)
    filtered = _convolve_reflect(image_array[:, :, 0], _box_kernel(kernel_size))
    return _clip_and_restore_shape(filtered)


def cross_mean_filter(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a cross-shaped local mean filter to a grayscale image."""
    image_array = _as_mnist_image(image)
    filtered = _cross_convolve_reflect(image_array[:, :, 0], radius)
    return _clip_and_restore_shape(filtered)


def diamond_mean_filter(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a Manhattan-distance local mean filter to a grayscale image."""
    image_array = _as_mnist_image(image)
    filtered = _convolve_reflect(image_array[:, :, 0], _diamond_kernel(radius))
    return _clip_and_restore_shape(filtered)
