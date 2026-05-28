"""Generic FGSM split runner for configured filter experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from deepdetector.evaluation.article_reproduction import (
    close_graph,
    create_restored_mnist_graph,
    evaluate_filter_on_images,
    load_mnist_test_slice,
)
from deepdetector.experiments.metadata import build_experiment_payload
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.io.result_writers import write_experiment_outputs
from deepdetector.paths import MNIST_M1_CHECKPOINT_DIR


SPLIT_METRIC_FIELDS: Sequence[str] = (
    "split",
    "TP",
    "FN",
    "FP",
    "recall_percent",
    "precision_percent",
    "f1_percent",
)


def _experiment_id(config: Dict[str, Any]) -> str:
    """Return the configured experiment identity."""
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("Config must define experiment_id.")
    return experiment_id


def _configured_slices(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Return configured dataset slices."""
    slices = config.get("dataset", {}).get("slices", [])
    if not slices:
        raise ValueError("Config must define dataset.slices.")
    for split_config in slices:
        yield split_config


def _output_dir(config: Dict[str, Any], experiment_id: str) -> Any:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = resolve_project_path(
        output_config.get("dir") or output_config.get("results_dir")
    )
    if output_dir is not None:
        return output_dir
    return resolve_project_path("results/experiments/{0}".format(experiment_id))


def _output_name(config: Dict[str, Any], key: str, default: str) -> str:
    """Return a configured output filename."""
    return str(config.get("output", {}).get(key, default))


def _checkpoint_dir(config: Dict[str, Any]) -> str:
    """Return the configured model checkpoint directory."""
    checkpoint_dir = resolve_project_path(config.get("model", {}).get("checkpoint_dir"))
    return str(checkpoint_dir or MNIST_M1_CHECKPOINT_DIR)


def run_fgsm_split_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run an FGSM filter evaluation experiment over configured dataset splits."""
    experiment_id = _experiment_id(config)
    dataset_name = str(config.get("dataset", {}).get("name", "")).strip()
    if dataset_name != "mnist":
        raise ValueError(
            "split_eval currently supports MNIST configs, got {0}.".format(dataset_name)
        )
    split_configs = list(_configured_slices(config))
    filter_name, filter_fn, filter_metadata = build_filter_from_config(config.get("filter", {}))
    attack_config = config.get("attack", {})
    evaluation_config = config.get("evaluation", {})
    rows: List[Dict[str, Any]] = []

    graph = create_restored_mnist_graph(_checkpoint_dir(config))
    try:
        for split_config in split_configs:
            split_name = str(split_config["name"])
            images, labels = load_mnist_test_slice(
                int(split_config["start"]),
                int(split_config["end"]),
            )
            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=float(attack_config.get("epsilon", attack_config.get("eps", 0.2))),
                filter_fn=filter_fn,
                batch_size=int(evaluation_config.get("batch_size", 256)),
                clip_min=float(attack_config.get("clip_min", 0.0)),
                clip_max=float(attack_config.get("clip_max", 1.0)),
                exclude_invalid_pairs=bool(
                    evaluation_config.get("exclude_invalid_pairs", False)
                ),
            )
            row = {"split": split_name}
            row.update(metrics)
            rows.append(row)
    finally:
        close_graph(graph)

    payload = build_experiment_payload(
        experiment_id=experiment_id,
        config=config,
        rows=rows,
        extra={"filter": dict(filter_metadata), "filter_name": filter_name},
    )
    output_dir = _output_dir(config, experiment_id)
    if output_dir is None:
        raise ValueError("Config must define an output directory.")
    write_experiment_outputs(
        output_dir=output_dir,
        rows=rows,
        csv_fields=SPLIT_METRIC_FIELDS,
        metadata=payload,
        csv_name=_output_name(config, "csv", "metrics.csv"),
        json_name=_output_name(config, "json", "metrics.json"),
    )
    return rows
