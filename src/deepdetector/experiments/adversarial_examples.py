"""Adversarial example materialization for configured experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


DEFAULT_ADVERSARIAL_CACHE_DIR = "artifacts/adversarial_examples"


@dataclass
class AdversarialExampleSet:
    """Clean/adversarial samples and baseline predictions for evaluation."""

    graph: Dict[str, Any]
    images: np.ndarray
    labels: np.ndarray
    adversarial_images: np.ndarray
    clean_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    metadata: Dict[str, Any]


def _checkpoint_dir(config: Dict[str, Any]) -> str:
    """Return the configured MNIST checkpoint directory."""
    checkpoint_dir = resolve_project_path(config.get("model", {}).get("checkpoint_dir"))
    return str(checkpoint_dir or MNIST_M1_CHECKPOINT_DIR)


def _json_key(data: Dict[str, Any]) -> str:
    """Return a deterministic JSON string for cache key material."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _cache_digest(data: Dict[str, Any]) -> str:
    """Return a short stable cache digest."""
    return hashlib.sha256(_json_key(data).encode("utf-8")).hexdigest()[:20]


def _cache_enabled(config: Dict[str, Any]) -> bool:
    """Return whether adversarial example cache is enabled."""
    return bool(config.get("attack", {}).get("cache", True))


def _cache_root(config: Dict[str, Any]) -> Path:
    """Return the configured adversarial cache root."""
    attack_config = config.get("attack", {})
    cache_dir = attack_config.get("cache_dir") or DEFAULT_ADVERSARIAL_CACHE_DIR
    resolved = resolve_project_path(cache_dir)
    return resolved or Path(str(cache_dir))


def _mnist_fgsm_cache_key(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return cache key material for one MNIST FGSM materialization."""
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    attack_config = config.get("attack", {})
    return {
        "version": 1,
        "kind": "mnist_fgsm",
        "dataset": {
            "name": "mnist",
            "split": dataset_config.get("split"),
            "start": int(dataset_config.get("start", 0)),
            "end": int(dataset_config.get("end", dataset_config.get("start", 0))),
            "high_entropy_only": bool(dataset_config.get("high_entropy_only", False)),
            "entropy_threshold": dataset_config.get("entropy_threshold"),
        },
        "model": {
            "name": model_config.get("name", "mnist_m1"),
            "checkpoint_dir": _checkpoint_dir(config),
        },
        "attack": {
            "name": attack_config.get("name", "fgsm"),
            "epsilon": float(attack_config.get("epsilon", attack_config.get("eps", 0.2))),
            "clip_min": float(attack_config.get("clip_min", 0.0)),
            "clip_max": float(attack_config.get("clip_max", 1.0)),
        },
    }


def _mnist_fgsm_cache_path(config: Dict[str, Any]) -> Path:
    """Return the cache path for one MNIST FGSM materialization."""
    attack_config = config.get("attack", {})
    explicit_path = attack_config.get("cache_path") or attack_config.get("adversarial_path")
    if explicit_path:
        resolved = resolve_project_path(explicit_path)
        return resolved or Path(str(explicit_path))

    digest = _cache_digest(_mnist_fgsm_cache_key(config))
    return _cache_root(config) / "mnist" / "fgsm" / "{0}.npz".format(digest)


def _cache_metadata_path(cache_path: Path) -> Path:
    """Return the sidecar metadata path for a cache archive."""
    return cache_path.with_suffix(".json")


def _read_cache_metadata(cache_path: Path) -> Dict[str, Any]:
    """Return cache sidecar metadata when available."""
    metadata_path = _cache_metadata_path(cache_path)
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache_metadata(cache_path: Path, metadata: Dict[str, Any]) -> None:
    """Write human-readable cache metadata."""
    metadata_path = _cache_metadata_path(cache_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_mnist_fgsm_cache(
    cache_path: Path,
    graph: Dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    metadata: Dict[str, Any],
) -> Optional[AdversarialExampleSet]:
    """Load a compatible cached MNIST FGSM materialization."""
    if not cache_path.is_file():
        return None
    try:
        with np.load(str(cache_path), allow_pickle=False) as archive:
            adversarial_images = np.asarray(archive["adversarial_images"], dtype=np.float32)
            clean_predictions = np.asarray(archive["clean_predictions"], dtype=np.int64)
            adversarial_predictions = np.asarray(
                archive["adversarial_predictions"],
                dtype=np.int64,
            )
    except (OSError, KeyError, ValueError):
        return None

    if adversarial_images.shape != images.shape:
        return None
    if len(clean_predictions) != len(images) or len(adversarial_predictions) != len(images):
        return None

    cached_metadata = dict(metadata)
    cached_metadata.update(_read_cache_metadata(cache_path))
    cached_metadata.update(
        {
            "cache_status": "hit",
            "cache_path": str(cache_path),
        }
    )
    return AdversarialExampleSet(
        graph=graph,
        images=images,
        labels=labels,
        adversarial_images=adversarial_images,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        metadata=cached_metadata,
    )


def _write_mnist_fgsm_cache(
    cache_path: Path,
    adversarial_set: AdversarialExampleSet,
    cache_key: Dict[str, Any],
) -> None:
    """Persist one MNIST FGSM materialization."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        adversarial_images=adversarial_set.adversarial_images,
        clean_predictions=adversarial_set.clean_predictions,
        adversarial_predictions=adversarial_set.adversarial_predictions,
    )
    metadata = dict(adversarial_set.metadata)
    metadata.update(
        {
            "cache_key": cache_key,
            "cache_path": str(cache_path),
        }
    )
    _write_cache_metadata(cache_path, metadata)


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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


def load_mnist_experiment_slice(
    dataset_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and optionally entropy-filter one configured MNIST slice."""
    dataset_name = str(dataset_config.get("name", "")).strip()
    if dataset_name != "mnist":
        raise ValueError(
            "MNIST FGSM materialization requires dataset.name=mnist, got {0}.".format(
                dataset_name
            )
        )

    start = int(dataset_config.get("start", 0))
    end = int(dataset_config.get("end", start))
    images, labels = load_mnist_test_slice(start, end)
    images, labels, metadata = _filter_high_entropy(images, labels, dataset_config)
    metadata.update({"start": start, "end": end})
    if dataset_config.get("slice_name") is not None:
        metadata["slice_name"] = dataset_config.get("slice_name")
    if dataset_config.get("split") is not None:
        metadata["split"] = dataset_config.get("split")
    return images, labels, metadata


def materialize_mnist_fgsm_examples(
    graph: Dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    attack_config: Dict[str, Any],
    evaluation_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AdversarialExampleSet:
    """Generate MNIST FGSM examples and baseline predictions once."""
    evaluation_config = evaluation_config or {}
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
    materialized_metadata = dict(metadata or {})
    materialized_metadata.update(
        {
            "attack_name": str(attack_config.get("name", "fgsm")),
            "epsilon": float(attack_config.get("epsilon", attack_config.get("eps", 0.2))),
            "clip_min": float(attack_config.get("clip_min", 0.0)),
            "clip_max": float(attack_config.get("clip_max", 1.0)),
            "num_examples": int(len(images)),
        }
    )
    return AdversarialExampleSet(
        graph=graph,
        images=images,
        labels=labels,
        adversarial_images=adversarial_images,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        metadata=materialized_metadata,
    )


def prepare_mnist_fgsm_adversarial_set(
    config: Dict[str, Any],
    graph: Optional[Dict[str, Any]] = None,
) -> AdversarialExampleSet:
    """Load data, restore or reuse a graph, and materialize MNIST FGSM samples."""
    images, labels, metadata = load_mnist_experiment_slice(config.get("dataset", {}))
    active_graph = graph or create_restored_mnist_graph(_checkpoint_dir(config))

    if _cache_enabled(config):
        cache_path = _mnist_fgsm_cache_path(config)
        cached = _load_mnist_fgsm_cache(
            cache_path=cache_path,
            graph=active_graph,
            images=images,
            labels=labels,
            metadata=metadata,
        )
        if cached is not None:
            return cached

    adversarial_set = materialize_mnist_fgsm_examples(
        graph=active_graph,
        images=images,
        labels=labels,
        attack_config=config.get("attack", {}),
        evaluation_config=config.get("evaluation", {}),
        metadata=metadata,
    )
    if _cache_enabled(config):
        cache_key = _mnist_fgsm_cache_key(config)
        cache_path = _mnist_fgsm_cache_path(config)
        adversarial_set.metadata.update(
            {
                "cache_status": "miss",
                "cache_path": str(cache_path),
            }
        )
        _write_mnist_fgsm_cache(cache_path, adversarial_set, cache_key)
    return adversarial_set
