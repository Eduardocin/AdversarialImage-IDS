"""Shared helpers for ImageNet article reproduction runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.adversarial_loader import adversarial_images_for_run
from deepdetector.data.imagenet import (
    class_image_rows,
    read_rgb_image,
    resize_normalized_image,
)
from deepdetector.evaluation.imagenet_utils import (
    article_model_inputs,
    filter_clean_baseline_images,
    validate_imagenet_split_paths,
)
from deepdetector.io.csv_writer import write_pivot_csv as write_shared_pivot_csv
from deepdetector.io.paths import resolve_project_path
from deepdetector.io.result_writers import write_metrics_json
from deepdetector.models.imagenet_wrappers import (
    GoogLeNetCaffeWrapper,
    build_googlenet_caffe_model,
)


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a project-relative path."""
    return resolve_project_path(path_value)


def output_dir_from_config(config: Dict[str, Any]) -> Path:
    """Return the configured ImageNet output directory."""
    output_config = config.get("output", config.get("outputs", {}))
    output_dir = _resolve_path(output_config.get("dir") or output_config.get("results_dir"))
    if output_dir is None:
        raise ValueError("ImageNet runner config must define output.dir.")
    return output_dir


def status_path_from_config(
    config: Dict[str, Any],
    output_dir: Path,
    default_name: str = "imagenet_status.json",
) -> Path:
    """Return the configured ImageNet status JSON path."""
    output_config = config.get("output", config.get("outputs", {}))
    return output_dir / str(output_config.get("status_json", default_name))


def write_imagenet_status(
    config: Dict[str, Any],
    output_dir: Path,
    status: str,
    default_name: str = "imagenet_status.json",
    **fields: Any,
) -> Path:
    """Write a status JSON file for complete or partial ImageNet runs."""
    payload = {"status": status}
    payload.update(fields)
    path = status_path_from_config(config, output_dir, default_name=default_name)
    write_metrics_json(path, payload)
    return path


def remove_stale_standard_outputs(output_dir: Path) -> None:
    """Remove stale generic metric outputs from older ImageNet table runs."""
    for filename in ("metrics.csv", "metrics.json"):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def build_imagenet_caffe_model(config: Dict[str, Any]) -> GoogLeNetCaffeWrapper:
    """Instantiate the configured GoogLeNet Caffe wrapper."""
    return build_googlenet_caffe_model(config)


def load_imagenet_subset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load configured ImageNet class folders as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    class_configs = dataset_config.get("classes", [])
    if not class_configs:
        raise ValueError("ImageNet runner config must define dataset.classes.")

    validate_imagenet_split_paths(config)
    rows = class_image_rows(class_configs)
    if bool(dataset_config.get("shuffle", False)):
        seed = int(config.get("experiment", {}).get("seed", 20170830))
        rng = np.random.RandomState(seed)
        rows = [rows[int(index)] for index in rng.permutation(len(rows))]

    configured_n_samples = dataset_config.get("n_samples")
    if configured_n_samples not in (None, "", "all"):
        rows = rows[: int(configured_n_samples)]

    image_size = int(dataset_config.get("image_size", 224))
    images = []
    labels = []
    for path, label in rows:
        image = read_rgb_image(path)
        images.append(resize_normalized_image(image, image_size=image_size))
        labels.append(label)

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def prepare_clean_imagenet_baseline(
    model: object,
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Keep only images that the clean ImageNet model classifies correctly."""
    return filter_clean_baseline_images(model=model, images=images, labels=labels)


def prepare_imagenet_adversarial_images(
    config: Dict[str, Any],
    model: object,
    images: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert clean images to model input space and load or generate adversarials."""
    model_inputs = article_model_inputs(model, images)
    adversarial_images = adversarial_images_for_run(
        config=config,
        model=model,
        images=model_inputs,
        selected_indices=selected_indices,
    )
    return model_inputs, adversarial_images


def write_imagenet_pivot_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
) -> Path:
    """Write an ImageNet article-style pivot CSV."""
    return write_shared_pivot_csv(path=path, rows=rows, columns=columns)
