"""Final adaptive detection filter for MNIST defense-aware evaluation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from deepdetector.filters.entropy import one_d_entropy
from deepdetector.filters.mean_filters import cross_mean_filter
from deepdetector.filters.quantization import scalar_quantization


def _single_final_adaptive_detection_filter(
    image: np.ndarray,
    *,
    low_threshold: float,
    medium_threshold: float,
    low_entropy_step: int,
    medium_entropy_step: int,
    high_entropy_step: int,
    spatial_radius: int,
) -> np.ndarray:
    image_array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    entropy = one_d_entropy(image_array)

    if entropy < float(low_threshold):
        return scalar_quantization(
            image_array,
            interval=int(low_entropy_step),
            left=True,
        ).reshape(image_array.shape)

    if entropy < float(medium_threshold):
        return scalar_quantization(
            image_array,
            interval=int(medium_entropy_step),
            left=True,
        ).reshape(image_array.shape)

    quantized = scalar_quantization(
        image_array,
        interval=int(high_entropy_step),
        left=True,
    ).reshape(image_array.shape)
    smoothed = cross_mean_filter(quantized, radius=int(spatial_radius)).reshape(
        image_array.shape
    )
    use_quantized = np.abs(quantized - image_array) < np.abs(smoothed - image_array)
    return np.where(use_quantized, quantized, smoothed).astype(np.float32).reshape(
        image_array.shape
    )


def final_adaptive_detection_filter(
    image: np.ndarray,
    *,
    low_threshold: float = 4.0,
    medium_threshold: float = 5.0,
    low_entropy_step: int = 128,
    medium_entropy_step: int = 64,
    high_entropy_step: int = 43,
    spatial_radius: int = 3,
) -> np.ndarray:
    """Apply the final adaptive detection filter on MNIST [0, 1] inputs."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        filtered = [
            _single_final_adaptive_detection_filter(
                single,
                low_threshold=low_threshold,
                medium_threshold=medium_threshold,
                low_entropy_step=low_entropy_step,
                medium_entropy_step=medium_entropy_step,
                high_entropy_step=high_entropy_step,
                spatial_radius=spatial_radius,
            )
            for single in image_array
        ]
        return np.asarray(filtered, dtype=np.float32).reshape(image_array.shape)

    return _single_final_adaptive_detection_filter(
        image_array,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
        low_entropy_step=low_entropy_step,
        medium_entropy_step=medium_entropy_step,
        high_entropy_step=high_entropy_step,
        spatial_radius=spatial_radius,
    ).reshape(image_array.shape)


def build_final_adaptive_detection_filter(config: Dict[str, Any]):
    """Build the defense-aware transform from experiment detector config."""
    thresholds = dict(config.get("entropy_thresholds", {}))
    quantization = dict(config.get("quantization", {}))
    spatial_filter = dict(config.get("spatial_filter", {}))

    low_threshold = float(thresholds.get("low", 4.0))
    medium_threshold = float(thresholds.get("medium", 5.0))
    low_entropy_step = int(quantization.get("low_entropy_step", 128))
    medium_entropy_step = int(quantization.get("medium_entropy_step", 64))
    high_entropy_step = int(quantization.get("high_entropy_step", 43))
    spatial_radius = int(spatial_filter.get("radius", 3))

    def transform(image: np.ndarray) -> np.ndarray:
        return final_adaptive_detection_filter(
            image,
            low_threshold=low_threshold,
            medium_threshold=medium_threshold,
            low_entropy_step=low_entropy_step,
            medium_entropy_step=medium_entropy_step,
            high_entropy_step=high_entropy_step,
            spatial_radius=spatial_radius,
        )

    return transform
