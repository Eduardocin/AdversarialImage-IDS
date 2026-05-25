"""Experimental filters used by the ImageNet Table 7 reproduction."""

from __future__ import annotations

import numpy as np

from deepdetector.filters.spatial_smoothing import spatial_smoothing_filter


def table7_filter(
    image: np.ndarray,
    mask_type: str,
    size: int,
) -> np.ndarray:
    """Apply one Table 7 spatial smoothing mask to 0-255 CHW data."""
    smoothed = spatial_smoothing_filter(
        image=np.asarray(image, dtype=np.float32),
        mask_type=mask_type,
        size=size,
    )
    return np.clip(smoothed, 0.0, 255.0).astype(np.float32)
