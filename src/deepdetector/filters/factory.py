"""Build configured image filters for experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from deepdetector.filters.mean_filters import (
    box_mean_filter,
    cross_mean_filter,
    diamond_mean_filter,
)
from deepdetector.filters.quantization import (
    nonuniform_quantization,
    nonuniform_quantization_legacy,
)


FilterFn = Callable[[np.ndarray], np.ndarray]


def _require_name(config: Dict[str, Any]) -> str:
    """Return a non-empty configured filter name."""
    name = str(config.get("name", "")).strip()
    if not name:
        raise ValueError("Filter config must define name.")
    return name


def _require_positive_int(config: Dict[str, Any], key: str) -> int:
    """Return a positive integer config value."""
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError("{0} must be a positive integer.".format(key))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("{0} must be a positive integer.".format(key))
    if parsed <= 0:
        raise ValueError("{0} must be a positive integer.".format(key))
    return parsed


def _metadata(
    filter_name: str,
    filter_type: str,
    mask_type: str,
    mask_size: int,
    radius: Any = None,
    kernel_size: Any = None,
) -> Dict[str, Any]:
    """Return normalized filter metadata for outputs."""
    return {
        "filter_name": filter_name,
        "filter_type": filter_type,
        "mask_type": mask_type,
        "mask_size": int(mask_size),
        "radius": radius,
        "kernel_size": kernel_size,
    }


def build_filter_from_config(config: Dict[str, Any]) -> Tuple[str, FilterFn, Dict[str, Any]]:
    """Build one filter from a YAML config."""
    filter_name = _require_name(config)
    filter_type = str(config.get("type", "")).strip()

    if filter_type == "scalar_quantization":
        from deepdetector.evaluation.article_reproduction import (
            interval_size,
            scalar_filter_for_intervals,
        )

        intervals = _require_positive_int(config, "intervals")
        quantization_interval = interval_size(intervals)
        metadata = {
            "filter_name": filter_name,
            "filter_type": filter_type,
            "intervals": intervals,
            "interval_size": quantization_interval,
        }
        return filter_name, scalar_filter_for_intervals(intervals), metadata

    if filter_type == "nonuniform_quantization":
        metadata = {
            "filter_name": filter_name,
            "filter_type": filter_type,
        }
        return filter_name, nonuniform_quantization, metadata

    if filter_type == "nonuniform_quantization_legacy":
        metadata = {
            "filter_name": filter_name,
            "filter_type": filter_type,
        }
        return filter_name, nonuniform_quantization_legacy, metadata

    if filter_type == "cross_mean":
        radius = _require_positive_int(config, "radius")
        metadata = _metadata(
            filter_name=filter_name,
            filter_type=filter_type,
            mask_type="cross",
            mask_size=2 * radius + 1,
            radius=radius,
            kernel_size=None,
        )
        return filter_name, lambda image: cross_mean_filter(image, radius=radius), metadata

    if filter_type == "diamond_mean":
        radius = _require_positive_int(config, "radius")
        metadata = _metadata(
            filter_name=filter_name,
            filter_type=filter_type,
            mask_type="diamond",
            mask_size=2 * radius + 1,
            radius=radius,
            kernel_size=None,
        )
        return filter_name, lambda image: diamond_mean_filter(image, radius=radius), metadata

    if filter_type == "box_mean":
        kernel_size = _require_positive_int(config, "kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        metadata = _metadata(
            filter_name=filter_name,
            filter_type=filter_type,
            mask_type="box",
            mask_size=kernel_size,
            radius=None,
            kernel_size=kernel_size,
        )
        return filter_name, lambda image: box_mean_filter(image, kernel_size=kernel_size), metadata

    if filter_type == "adaptive_quantization":
        from deepdetector.evaluation.article_reproduction import adaptive_quantization_filter

        metadata = {
            "filter_name": filter_name,
            "filter_type": filter_type,
        }
        return filter_name, adaptive_quantization_filter, metadata

    if filter_type in {"proposed_detection_filter", "proposed_filter"}:
        from deepdetector.evaluation.article_reproduction import proposed_detection_filter

        metadata = {
            "filter_name": filter_name,
            "filter_type": "proposed_detection_filter",
        }
        return filter_name, proposed_detection_filter, metadata

    raise ValueError("Unknown filter type: {0}".format(filter_type or "<missing>"))
