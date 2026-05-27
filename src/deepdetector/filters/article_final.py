"""Final entropy-aware detector filter used by the article reproductions."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from deepdetector.filters.entropy import image_entropy_255_chw, one_d_entropy
from deepdetector.filters.quantization import scalar_quantization
from deepdetector.filters.spatial_smoothing import spatial_smoothing_filter


def _is_normalized(image: np.ndarray) -> bool:
    """Return whether the image appears to use the normalized [0, 1] scale."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0:
        return True
    return bool(float(np.nanmin(image_array)) >= 0.0 and float(np.nanmax(image_array)) <= 1.0)


def _layout_for_single_image(image: np.ndarray) -> str:
    """Return the most likely layout for a 2D/3D image."""
    if image.ndim == 2:
        return "hw"
    if image.ndim != 3:
        raise ValueError("image must have shape HxW, HxWxC, CxHxW, or a batch of images.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        return "chw"
    if image.shape[-1] in (1, 3):
        return "hwc"
    raise ValueError("image must use HWC or CHW layout with 1 or 3 channels.")


def _entropy_for_image(image: np.ndarray) -> float:
    """Compute entropy in the image's native scale."""
    image_array = np.asarray(image, dtype=np.float32)
    if _is_normalized(image_array):
        return one_d_entropy(image_array)

    layout = _layout_for_single_image(image_array)
    if layout == "chw":
        return image_entropy_255_chw(image_array)
    if layout == "hwc":
        return image_entropy_255_chw(np.transpose(image_array, (2, 0, 1)))
    return one_d_entropy(np.clip(image_array, 0.0, 255.0) / 255.0)


def _step_for_entropy(entropy: float) -> int:
    """Return the Table 9 scalar quantization step for one entropy value."""
    if float(entropy) < 4.0:
        return 128
    if float(entropy) < 5.0:
        return 64
    return 43


def _quantize_native_scale(image: np.ndarray, step: int) -> np.ndarray:
    """Apply scalar quantization while preserving normalized or Caffe scale."""
    image_array = np.asarray(image, dtype=np.float32)
    if _is_normalized(image_array):
        return scalar_quantization(image_array, interval=int(step), left=True).reshape(image_array.shape)

    quantized = np.clip(image_array, 0.0, 255.0)
    quantized //= int(step)
    quantized *= int(step)
    return np.clip(quantized, 0.0, 255.0).astype(np.float32).reshape(image_array.shape)


def _to_chw_255(image: np.ndarray) -> Tuple[np.ndarray, str, bool]:
    """Convert one image to CHW 0-255 data for spatial smoothing."""
    image_array = np.asarray(image, dtype=np.float32)
    layout = _layout_for_single_image(image_array)
    normalized = _is_normalized(image_array)

    if layout == "hw":
        chw = image_array.reshape((1,) + image_array.shape)
    elif layout == "hwc":
        chw = np.transpose(image_array, (2, 0, 1))
    else:
        chw = image_array

    if normalized:
        chw = chw * 255.0
    return np.clip(chw, 0.0, 255.0).astype(np.float32), layout, normalized


def _restore_from_chw_255(chw_255: np.ndarray, layout: str, normalized: bool) -> np.ndarray:
    """Restore smoothed CHW 0-255 data to the original layout and scale."""
    restored = np.asarray(chw_255, dtype=np.float32)
    if normalized:
        restored = restored / 255.0

    if layout == "hw":
        restored = restored[0]
    elif layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    elif layout != "chw":
        raise ValueError("Unknown image layout: {0}".format(layout))

    clip_max = 1.0 if normalized else 255.0
    return np.clip(restored, 0.0, clip_max).astype(np.float32)


def _cross_smoothing_7x7_native_scale(image: np.ndarray) -> np.ndarray:
    """Apply 7x7 cross smoothing while preserving layout and scale."""
    chw_255, layout, normalized = _to_chw_255(image)
    smoothed = spatial_smoothing_filter(chw_255, mask_type="cross", size=7)
    return _restore_from_chw_255(smoothed, layout=layout, normalized=normalized).reshape(image.shape)


def _article_final_single(image: np.ndarray) -> np.ndarray:
    """Apply the final detector filter to one image."""
    image_array = np.asarray(image, dtype=np.float32)
    entropy = _entropy_for_image(image_array)
    step = _step_for_entropy(entropy)
    quantized = _quantize_native_scale(image_array, step=step)

    if entropy < 5.0:
        return quantized.astype(np.float32).reshape(image_array.shape)

    smoothed = _cross_smoothing_7x7_native_scale(quantized)
    use_quantized = np.abs(quantized - image_array) <= np.abs(smoothed - image_array)
    return np.where(use_quantized, quantized, smoothed).astype(np.float32).reshape(image_array.shape)


def article_final_detection_filter(image: np.ndarray) -> np.ndarray:
    """Apply the article's final entropy-aware detection filter."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        filtered = [_article_final_single(single) for single in image_array]
        return np.asarray(filtered, dtype=np.float32).reshape(image_array.shape)
    return _article_final_single(image_array)
