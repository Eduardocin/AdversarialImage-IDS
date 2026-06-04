"""Shared ImageNet evaluation helpers for article reproduction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import predict_caffe_label, preprocess_caffe_inputs
from deepdetector.io.paths import resolve_project_path
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


def article_model_inputs(model: GoogLeNetCaffeWrapper, images: np.ndarray) -> np.ndarray:
    """Return images in the Caffe input space used by the source article."""
    return preprocess_caffe_inputs(model, images)


def predict_label(model: GoogLeNetCaffeWrapper, image: np.ndarray) -> int:
    """Predict the top-1 label for one HWC image or preprocessed Caffe tensor."""
    return predict_caffe_label(model, image)


def label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to a Python int."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def epsilon_normalized(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in normalized [0, 1] image scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"]) / 255.0
    return float(attack_config.get("epsilon", attack_config.get("eps", 1.0 / 255.0)))


def epsilon_255(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in raw 0-255 image scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    return float(epsilon_normalized(config) * 255.0)


def filter_clean_baseline_images(
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Keep only images correctly classified before attack evaluation."""
    keep_indices = []
    for index, image in enumerate(images):
        clean_pred = predict_label(model, image)
        if clean_pred == label_to_int(labels[index]):
            keep_indices.append(index)

    selected = np.asarray(keep_indices, dtype=np.int64)
    summary = {
        "total_images": int(len(images)),
        "clean_correct": int(len(selected)),
        "skipped_wrong_baseline": int(len(images) - len(selected)),
    }
    return images[selected], labels[selected], selected, summary


def validate_imagenet_split_paths(
    config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """Reject class paths that point at a different ImageNet split."""
    dataset_config = config.get("dataset", {})
    split = str(dataset_config.get("split", "")).strip().lower()
    if not split:
        return
    if split == "training":
        split = "train"

    known_splits = {"train", "validation", "test"}
    for class_config in dataset_config.get("classes", []):
        class_path = resolve_project_path(class_config.get("path"), project_root=project_root)
        if class_path is None:
            continue

        path_parts = {part.lower() for part in class_path.parts}
        mismatched = sorted((known_splits - {split}) & path_parts)
        if mismatched:
            raise ValueError(
                "Configured ImageNet {0} split cannot use {1} path: {2}".format(
                    split,
                    mismatched[0],
                    class_path,
                )
            )
