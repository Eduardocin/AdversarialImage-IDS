"""Experimental filters used by the ImageNet Table 7 reproduction."""

from __future__ import annotations

import numpy as np

from deepdetector.filters.quantization import scalar_quantization
from deepdetector.filters.spatial_smoothing import spatial_smoothing_filter


def _scalar_quantization_255(image: np.ndarray, quantization_step: int) -> np.ndarray:
    """Apply the existing normalized scalar quantizer to 0-255 CHW data."""
    image_255 = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    normalized = image_255 / 255.0
    quantized = scalar_quantization(
        normalized,
        interval=int(quantization_step),
        left=True,
    )
    return (quantized * 255.0).astype(np.float32)


def table7_filter(
    image: np.ndarray,
    mask_type: str,
    size: int,
    quantization_step: int = 43,
) -> np.ndarray:
    """Apply scalar quantization followed by a Table 7 spatial smoothing mask."""
    quantized = _scalar_quantization_255(
        image=image,
        quantization_step=quantization_step,
    )
    smoothed = spatial_smoothing_filter(
        image=quantized,
        mask_type=mask_type,
        size=size,
    )
    return np.clip(smoothed, 0.0, 255.0).astype(np.float32)
