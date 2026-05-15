"""Adaptive noise reduction filters."""

from deepdetector.filters.quantization import (
    find_border,
    nonuniform_quantization,
    normalize_image_range,
    scalar_quantization,
)

__all__ = [
    "find_border",
    "nonuniform_quantization",
    "normalize_image_range",
    "scalar_quantization",
]
