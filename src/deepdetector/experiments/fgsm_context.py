"""Reusable FGSM evaluation context for filter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from deepdetector.attacks.fgsm import generate_fgsm_examples
from deepdetector.evaluation.article_reproduction import (
    create_restored_mnist_graph,
    load_mnist_test_slice,
    predict_labels,
)
from deepdetector.filters.entropy import one_d_entropy
from deepdetector.io.paths import resolve_project_path
from deepdetector.paths import MNIST_M1_CHECKPOINT_DIR


@dataclass
class FGSMEvaluationContext:
    """Prepared data and predictions reused by all candidate filters."""

    graph: Dict[str, Any]
    images: np.ndarray
    labels: np.ndarray
    adversarial_images: np.ndarray
    clean_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    metadata: Dict[str, Any]


def _entropy_threshold_min(config: Dict[str, Any]) -> float:
    """Return the configured lower entropy threshold."""
    threshold_config = config.get("entropy_threshold", {})
    if isinstance(threshold_config, dict):
        return float(threshold_config.get("min", 5.0))
    return float(threshold_config)


def _filter_high_entropy(
    images: np.ndarray,
    labels: np.ndarray,
    dataset_config: Dict[str, Any],
) -> tuple:
    """Return only images whose entropy is above the configured threshold."""
    if not bool(dataset_config.get("high_entropy_only", False)):
        return images, labels, {
            "high_entropy_only": False,
            "entropy_threshold_min": None,
            "num_loaded": int(len(images)),
            "num_after_entropy_filter": int(len(images)),
        }

    threshold_min = _entropy_threshold_min(dataset_config)
    keep = np.asarray(
        [one_d_entropy(image) > threshold_min for image in images],
        dtype=bool,
    )
    return images[keep], labels[keep], {
        "high_entropy_only": True,
        "entropy_threshold_min": threshold_min,
        "num_loaded": int(len(images)),
        "num_after_entropy_filter": int(np.sum(keep)),
    }


def prepare_fgsm_context(config: Dict[str, Any]) -> FGSMEvaluationContext:
    """Load data, generate FGSM examples once, and compute base predictions."""
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    attack_config = config.get("attack", {})
    evaluation_config = config.get("evaluation", {})
    dataset_name = str(dataset_config.get("name", "")).strip()
    if dataset_name != "mnist":
        raise ValueError(
            "filter_grid currently supports MNIST configs, got {0}.".format(dataset_name)
        )

    start = int(dataset_config.get("start", 0))
    end = int(dataset_config.get("end", start))
    images, labels = load_mnist_test_slice(start, end)
    images, labels, metadata = _filter_high_entropy(images, labels, dataset_config)

    checkpoint_dir = resolve_project_path(model_config.get("checkpoint_dir"))
    graph = create_restored_mnist_graph(str(checkpoint_dir or MNIST_M1_CHECKPOINT_DIR))
    batch_size = int(evaluation_config.get("batch_size", 256))

    adversarial_images = generate_fgsm_examples(
        sess=graph["sess"],
        model=graph["model"],
        x_placeholder=graph["x"],
        images=images,
        eps=float(attack_config.get("epsilon", attack_config.get("eps", 0.2))),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 1.0)),
    )
    clean_predictions = predict_labels(
        graph["sess"],
        graph["x"],
        graph["predictions"],
        images,
        batch_size=batch_size,
    )
    adversarial_predictions = predict_labels(
        graph["sess"],
        graph["x"],
        graph["predictions"],
        adversarial_images,
        batch_size=batch_size,
    )

    return FGSMEvaluationContext(
        graph=graph,
        images=images,
        labels=labels,
        adversarial_images=adversarial_images,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        metadata=metadata,
    )
