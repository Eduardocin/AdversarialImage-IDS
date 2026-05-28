"""Generic runner for configured filter-candidate experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from deepdetector.evaluation.article_reproduction import (
    close_graph,
    evaluate_filter_on_existing_adversarial,
)
from deepdetector.experiments.fgsm_context import prepare_fgsm_context
from deepdetector.experiments.metadata import build_experiment_payload
from deepdetector.filters.factory import build_filter_from_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.io.result_writers import write_experiment_outputs


COUNT_FIELDS = (
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
    output_config = config.get("output", {})
    return str(output_config.get(key, default))


def _best_filter_by_f1(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a small best-filter summary by f1_percent."""
    if not rows:
        return {}

    best = max(rows, key=lambda row: float(row.get("f1_percent", 0.0)))
    fields = (
        "filter_name",
        "filter_type",
        "intervals",
        "interval_size",
        "mask_type",
        "mask_size",
        "radius",
        "kernel_size",
        "f1_percent",
    )
    return {field: best.get(field) for field in fields}


def _csv_fields_for_rows(rows: Sequence[Dict[str, Any]]) -> Sequence[str]:
    """Return output columns based on filter metadata present in result rows."""
    if rows and all("intervals" in row and "interval_size" in row for row in rows):
        return (
            "filter_name",
            "filter_type",
            "intervals",
            "interval_size",
        ) + COUNT_FIELDS

    mask_fields = ("mask_type", "mask_size", "radius", "kernel_size")
    if rows and any(any(field in row for field in mask_fields) for row in rows):
        return (
            "filter_name",
            "filter_type",
            "mask_type",
            "mask_size",
            "radius",
            "kernel_size",
        ) + COUNT_FIELDS

    return ("filter_name", "filter_type") + COUNT_FIELDS


def run_filter_candidate_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run a filter candidate evaluation experiment."""
    experiment_id = _experiment_id(config)
    filter_configs = config.get("filters", [])
    if not filter_configs:
        raise ValueError("Config must define at least one filter candidate.")

    context = prepare_fgsm_context(config)
    rows: List[Dict[str, Any]] = []

    try:
        for filter_config in filter_configs:
            _, filter_fn, filter_metadata = build_filter_from_config(filter_config)
            metrics = evaluate_filter_on_existing_adversarial(
                graph=context.graph,
                images=context.images,
                labels=context.labels,
                adv_images=context.adversarial_images,
                clean_pred=context.clean_predictions,
                adv_pred=context.adversarial_predictions,
                filter_fn=filter_fn,
                batch_size=int(config.get("evaluation", {}).get("batch_size", 256)),
                exclude_invalid_pairs=bool(
                    config.get("evaluation", {}).get("exclude_invalid_pairs", False)
                ),
            )
            row = dict(filter_metadata)
            row.update(metrics)
            rows.append(row)
    finally:
        close_graph(context.graph)

    extra = {
        "num_filters": int(len(filter_configs)),
        "num_loaded": int(context.metadata.get("num_loaded", len(context.images))),
        "num_after_entropy_filter": int(
            context.metadata.get("num_after_entropy_filter", len(context.images))
        ),
        "high_entropy_only": bool(context.metadata.get("high_entropy_only", False)),
    }
    selection_stage = config.get("selection_stage")
    if selection_stage is not None:
        extra["selection_stage"] = selection_stage
    extra["best_filter_by_f1"] = _best_filter_by_f1(rows)

    payload = build_experiment_payload(
        experiment_id=experiment_id,
        config=config,
        rows=rows,
        extra=extra,
    )
    output_dir = _output_dir(config, experiment_id)
    if output_dir is None:
        raise ValueError("Config must define an output directory.")
    write_experiment_outputs(
        output_dir=output_dir,
        rows=rows,
        csv_fields=_csv_fields_for_rows(rows),
        metadata=payload,
        csv_name=_output_name(config, "csv", "metrics.csv"),
        json_name=_output_name(config, "json", "metrics.json"),
    )
    return rows
