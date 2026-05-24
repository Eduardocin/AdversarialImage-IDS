"""Adaptive noise reduction filters."""

from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.filters.entropy import image_entropy_255_chw, one_d_entropy
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
from deepdetector.filters.spatial_smoothing import (
    apply_mask_mean_filter,
    build_box_mask,
    build_cross_mask,
    build_diamond_mask,
    spatial_smoothing_filter,
)
from deepdetector.filters.table7_filters import table7_filter

__all__ = [
    "FILTER_REGISTRY",
    "apply_mask_mean_filter",
    "box_mean_filter",
    "build_box_mask",
    "build_cross_mask",
    "build_diamond_mask",
    "cross_mean_filter",
    "diamond_mean_filter",
    "entropy_based_quantization",
    "find_border",
    "image_entropy_255_chw",
    "nonuniform_quantization",
    "normalize_image_range",
    "one_d_entropy",
    "scalar_quantization",
    "spatial_smoothing_filter",
    "table7_filter",
]
