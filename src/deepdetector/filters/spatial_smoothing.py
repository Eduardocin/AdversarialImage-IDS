"""Spatial smoothing masks for CxHxW images in 0-255 pixel scale."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_size(size: int) -> int:
    """Validate a spatial mask size and return it as an int."""
    size_int = int(size)
    if size_int < 3:
        raise ValueError("size must be greater than or equal to 3.")
    if size_int % 2 == 0:
        raise ValueError("size must be odd.")
    return size_int


def build_cross_mask(size: int) -> np.ndarray:
    """Return a binary cross mask with the center row and column active."""
    size_int = _validate_size(size)
    center = size_int // 2
    mask = np.zeros((size_int, size_int), dtype=np.float32)
    mask[center, :] = 1.0
    mask[:, center] = 1.0
    return mask


def build_diamond_mask(size: int) -> np.ndarray:
    """Return a binary Manhattan-distance diamond mask."""
    size_int = _validate_size(size)
    center = size_int // 2
    mask = np.zeros((size_int, size_int), dtype=np.float32)
    for row in range(size_int):
        for col in range(size_int):
            if abs(row - center) + abs(col - center) <= center:
                mask[row, col] = 1.0
    return mask


def build_box_mask(size: int) -> np.ndarray:
    """Return a binary square mask."""
    size_int = _validate_size(size)
    return np.ones((size_int, size_int), dtype=np.float32)


def _mask_for_type(mask_type: str, size: int) -> np.ndarray:
    """Build one supported spatial smoothing mask."""
    builders = {
        "cross": build_cross_mask,
        "diamond": build_diamond_mask,
        "box": build_box_mask,
    }
    try:
        return builders[str(mask_type)](size)
    except KeyError:
        raise ValueError("mask_type must be one of: cross, diamond, box.")


def _validate_chw_image(image: np.ndarray) -> np.ndarray:
    """Return image as float32 CxHxW data."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must have shape (C, H, W).")
    if image_array.shape[0] <= 0 or image_array.shape[1] <= 0 or image_array.shape[2] <= 0:
        raise ValueError("image dimensions must be positive.")
    return image_array


def _normalized_kernel(mask: np.ndarray) -> np.ndarray:
    """Return a mean-filter kernel from a binary mask."""
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    if mask_array.shape[0] != mask_array.shape[1]:
        raise ValueError("mask must be square.")

    active = mask_array > 0.0
    mask_sum = float(np.sum(active))
    if mask_sum <= 0.0:
        raise ValueError("mask must contain at least one active element.")
    return active.astype(np.float32) / mask_sum


def apply_mask_mean_filter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a per-channel masked mean filter to CxHxW data.

    Borders use SciPy's deterministic ``reflect`` mode. The returned array keeps
    the input shape and uses ``float32`` values.
    """
    image_array = _validate_chw_image(image)
    kernel = _normalized_kernel(mask)

    filtered_channels = [
        ndimage.convolve(channel, weights=kernel, mode="reflect")
        for channel in image_array
    ]
    return np.stack(filtered_channels, axis=0).astype(np.float32)


def spatial_smoothing_filter(
    image: np.ndarray,
    mask_type: str,
    size: int,
) -> np.ndarray:
    """Apply one supported spatial smoothing mask to CxHxW image data."""
    mask = _mask_for_type(mask_type=mask_type, size=size)
    return apply_mask_mean_filter(image=image, mask=mask)
