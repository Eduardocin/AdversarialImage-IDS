"""Official combined Table 9 runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from deepdetector.evaluation.table6_imagenet import (
    DEFAULT_SPLIT_ORDER,
    TABLE6_OUTPUT_FIELDS,
    evaluate_imagenet_fgsm_filter,
    validate_table6_result,
)
from deepdetector.experiments.fgsm_split_runner import evaluate_mnist_fgsm_splits
from deepdetector.experiments.table6_runner import (
    _load_imagenet_fgsm_caches,
    _split_key,
    _write_missing_imagenet_fgsm_caches,
    aggregate_table6_rows,
    build_imagenet_model,
    load_imagenet_split_samples,
)
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json


def _output_dir(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output_dir = resolve_project_path(config.get("output", {}).get("dir"))
    if output_dir is None:
        raise ValueError("Table 9 must define output.dir.")
    return output_dir


def _epsilon_255(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in 0-255 Caffe scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    if "epsilon" in attack_config:
        return float(attack_config["epsilon"]) * 255.0
    return 1.0


def evaluate_mnist_table9(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate the final Table 9 detector on MNIST splits."""
    rows, _, _ = evaluate_mnist_fgsm_splits(config)
    return rows


def evaluate_imagenet_table9(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate the final Table 9 detector on ImageNet splits."""
    _, filter_fn, _ = build_filter_from_config(config.get("filter", {}))
    model = build_imagenet_model(config)
    samples_by_split = load_imagenet_split_samples(config)
    attack_config = config.get("attack", {})
    split_order = tuple(
        _split_key(split)
        for split in config.get("split_order", DEFAULT_SPLIT_ORDER)
    )
    cached_by_split, paths_by_split = _load_imagenet_fgsm_caches(
        config,
        split_order,
        table_label="Table 9",
    )
    result = evaluate_imagenet_fgsm_filter(
        model=model,
        samples_by_split=samples_by_split,
        filter_fn=filter_fn,
        epsilon_255=_epsilon_255(config),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 255.0)),
        split_order=split_order,
        adversarial_by_split=cached_by_split,
    )
    _write_missing_imagenet_fgsm_caches(
        result,
        cached_by_split,
        paths_by_split,
        table_label="Table 9",
    )
    validate_table6_result(result)
    return result.rows


def _json_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the compact Table 9 JSON payload."""
    payload: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        payload[str(row["split"])] = {
            "tp": int(row["TP"]),
            "fn": int(row["FN"]),
            "fp": int(row["FP"]),
            "recall_percent": float(row["recall_percent"]),
            "precision_percent": float(row["precision_percent"]),
            "f1_percent": float(row["f1_percent"]),
        }
    return payload


def run_table9_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run the official combined MNIST + ImageNet Table 9 experiment."""
    mnist_config = dict(config.get("mnist", {}))
    imagenet_config = dict(config.get("imagenet", {}))
    if not mnist_config or not imagenet_config:
        raise ValueError("Table 9 must define internal mnist and imagenet configs.")

    mnist_rows = evaluate_mnist_table9(mnist_config)
    imagenet_rows = evaluate_imagenet_table9(imagenet_config)
    rows = aggregate_table6_rows(
        mnist_rows=mnist_rows,
        imagenet_rows=imagenet_rows,
        split_order=config.get("split_order", DEFAULT_SPLIT_ORDER),
    )

    output_dir = ensure_dir(_output_dir(config))
    output_config = config.get("output", {})
    write_metrics_csv(
        output_dir / str(output_config.get("csv", "metrics.csv")),
        rows,
        TABLE6_OUTPUT_FIELDS,
    )
    write_metrics_json(
        output_dir / str(output_config.get("json", "metrics.json")),
        _json_payload(rows),
    )
    return rows
