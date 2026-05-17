"""Adaptive noise reduction filters."""

from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.filters.entropy import one_d_entropy
from deepdetector.filters.mean_filters import (
    box_mean_filter,
    cross_mean_filter,
    diamond_mean_filter,
)
from deepdetector.filters.quantization import (
    find_border,
    nonuniform_quantization,
    normalize_image_range,
    scalar_quantization,
)
from deepdetector.filters.registry import FILTER_REGISTRY

__all__ = [
    "FILTER_REGISTRY",
    "box_mean_filter",
    "cross_mean_filter",
    "diamond_mean_filter",
    "entropy_based_quantization",
    "find_border",
    "nonuniform_quantization",
    "normalize_image_range",
    "one_d_entropy",
    "scalar_quantization",
]
