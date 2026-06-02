"""Final entropy-aware detector filter used by the article reproductions."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from deepdetector.filters.entropy import image_entropy_255_chw, one_d_entropy
from deepdetector.filters.quantization import scalar_quantization
from deepdetector.filters.spatial_smoothing import spatial_smoothing_filter


UNIT_SCALE = "unit"
CENTERED_SCALE = "centered"
UINT8_SCALE = "uint8"


def _image_scale(image: np.ndarray) -> str:
    """Return the image scale used by the final detector."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0:
        return UNIT_SCALE

    min_value = float(np.nanmin(image_array))
    max_value = float(np.nanmax(image_array))
    if min_value < 0.0 and min_value >= -0.5 and max_value <= 0.5:
        return CENTERED_SCALE
    if min_value >= 0.0 and max_value <= 1.0:
        return UNIT_SCALE
    return UINT8_SCALE


def _to_unit_scale(image: np.ndarray, scale: str) -> np.ndarray:
    """Convert supported image scales to [0, 1]."""
    image_array = np.asarray(image, dtype=np.float32)
    if scale == CENTERED_SCALE:
        return np.clip(image_array + 0.5, 0.0, 1.0).astype(np.float32)
    if scale == UINT8_SCALE:
        return (np.clip(image_array, 0.0, 255.0) / 255.0).astype(np.float32)
    return np.clip(image_array, 0.0, 1.0).astype(np.float32)


def _to_255_scale(image: np.ndarray, scale: str) -> np.ndarray:
    """Convert supported image scales to [0, 255]."""
    return (_to_unit_scale(image, scale) * 255.0).astype(np.float32)


def _restore_scale(image_255: np.ndarray, scale: str) -> np.ndarray:
    """Restore [0, 255] data to the original image scale."""
    restored_255 = np.asarray(image_255, dtype=np.float32)
    if scale == CENTERED_SCALE:
        return (np.clip(restored_255, 0.0, 255.0) / 255.0 - 0.5).astype(np.float32)
    if scale == UINT8_SCALE:
        return np.clip(restored_255, 0.0, 255.0).astype(np.float32)
    return (np.clip(restored_255, 0.0, 255.0) / 255.0).astype(np.float32)


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
    scale = _image_scale(image_array)
    if scale in {UNIT_SCALE, CENTERED_SCALE}:
        return one_d_entropy(_to_unit_scale(image_array, scale))

    layout = _layout_for_single_image(image_array)
    if layout == "chw":
        return image_entropy_255_chw(_to_255_scale(image_array, scale))
    if layout == "hwc":
        return image_entropy_255_chw(np.transpose(_to_255_scale(image_array, scale), (2, 0, 1)))
    return one_d_entropy(_to_unit_scale(image_array, scale))


def _step_for_entropy(entropy: float) -> int:
    """Return the Table 9 scalar quantization step for one entropy value."""
    if float(entropy) < 4.0:
        return 128
    if float(entropy) < 5.0:
        return 64
    return 43


def _quantize_native_scale(image: np.ndarray, step: int, scale: Optional[str] = None) -> np.ndarray:
    """Apply scalar quantization while preserving the input image scale."""
    image_array = np.asarray(image, dtype=np.float32)
    image_scale = scale or _image_scale(image_array)
    if image_scale == UNIT_SCALE:
        return scalar_quantization(image_array, interval=int(step), left=True).reshape(image_array.shape)

    quantized = _to_255_scale(image_array, image_scale)
    quantized //= int(step)
    quantized *= int(step)
    return _restore_scale(quantized, image_scale).reshape(image_array.shape)


def _to_chw_255(image: np.ndarray, scale: Optional[str] = None) -> Tuple[np.ndarray, str, str]:
    """Convert one image to CHW 0-255 data for spatial smoothing."""
    image_array = np.asarray(image, dtype=np.float32)
    layout = _layout_for_single_image(image_array)
    image_scale = scale or _image_scale(image_array)

    if layout == "hw":
        chw = image_array.reshape((1,) + image_array.shape)
    elif layout == "hwc":
        chw = np.transpose(image_array, (2, 0, 1))
    else:
        chw = image_array

    return _to_255_scale(chw, image_scale), layout, image_scale


def _restore_from_chw_255(chw_255: np.ndarray, layout: str, scale: str) -> np.ndarray:
    """Restore smoothed CHW 0-255 data to the original layout and scale."""
    restored = _restore_scale(chw_255, scale)

    if layout == "hw":
        restored = restored[0]
    elif layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    elif layout != "chw":
        raise ValueError("Unknown image layout: {0}".format(layout))

    return restored.astype(np.float32)


def _cross_smoothing_7x7_native_scale(image: np.ndarray, scale: Optional[str] = None) -> np.ndarray:
    """Apply 7x7 cross smoothing while preserving layout and scale."""
    chw_255, layout, image_scale = _to_chw_255(image, scale=scale)
    smoothed = spatial_smoothing_filter(chw_255, mask_type="cross", size=7)
    return _restore_from_chw_255(smoothed, layout=layout, scale=image_scale).reshape(image.shape)


def _article_final_single(image: np.ndarray) -> np.ndarray:
    """Apply the final detector filter to one image."""
    image_array = np.asarray(image, dtype=np.float32)
    scale = _image_scale(image_array)
    entropy = _entropy_for_image(image_array)
    step = _step_for_entropy(entropy)
    quantized = _quantize_native_scale(image_array, step=step, scale=scale)

    if entropy < 5.0:
        return quantized.astype(np.float32).reshape(image_array.shape)

    smoothed = _cross_smoothing_7x7_native_scale(quantized, scale=scale)
    use_quantized = np.abs(quantized - image_array) <= np.abs(smoothed - image_array)
    return np.where(use_quantized, quantized, smoothed).astype(np.float32).reshape(image_array.shape)


def article_final_detection_filter(image: np.ndarray) -> np.ndarray:
    """Apply the article's final entropy-aware detection filter."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        filtered = [_article_final_single(single) for single in image_array]
        return np.asarray(filtered, dtype=np.float32).reshape(image_array.shape)
    return _article_final_single(image_array)
