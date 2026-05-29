"""Materialize official Table 10 model-group outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json


TABLE_10_SCHEMA: list[str] = [
    "no",
    "attack_model",
    "dataset",
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
]

TABLE_10_METRIC_FIELDS: list[str] = [
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
]


def build_pending_table_10_row(
    *,
    no: int,
    attack_model: str,
    dataset: str,
) -> dict[str, Any]:
    """Build a Table 10 row whose experiment metrics are not available yet."""
    row: dict[str, Any] = {
        "no": no,
        "attack_model": attack_model,
        "dataset": dataset,
    }
    for field in TABLE_10_METRIC_FIELDS:
        row[field] = None
    return row


def normalize_table_10_result(
    *,
    no: int,
    attack_model: str,
    dataset: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Convert a computed row result to the official Table 10 schema."""
    row = build_pending_table_10_row(
        no=no,
        attack_model=attack_model,
        dataset=dataset,
    )
    metrics = result.get("metrics", {})
    source = metrics if isinstance(metrics, dict) else result
    for field in TABLE_10_METRIC_FIELDS:
        if field in source:
            row[field] = source[field]
    return row


def _output_dir(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    configured_dir = output_config.get("dir") or config.get("output_dir")
    output_dir = resolve_project_path(configured_dir)
    if output_dir is None:
        raise ValueError("Table 10 group must define output.dir or output_dir.")
    return output_dir


def _row_result(row_config: dict[str, Any]) -> dict[str, Any]:
    metrics = row_config.get("metrics")
    if isinstance(metrics, dict):
        return {"metrics": metrics}
    return row_config


def save_table_10_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    dataset_group: str,
    model_group: str,
) -> dict[str, Path]:
    """Write the official CSV and JSON outputs for one Table 10 group."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / "metrics.csv", rows, TABLE_10_SCHEMA)
    json_path = write_metrics_json(
        output_path / "metrics.json",
        {
            "table": 10,
            "dataset_group": dataset_group,
            "model_group": model_group,
            "rows": rows,
        },
    )
    return {"csv": csv_path, "json": json_path}


def run_table_10_group(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize one configured Table 10 model group."""
    rows_config = list(config.get("rows", []))
    if not rows_config:
        raise ValueError("Table 10 group must define rows.")

    model_group = str(config.get("model_group", "")).strip()
    if not model_group:
        raise ValueError("Table 10 group must define model_group.")

    dataset_label = str(config.get("dataset_label", "")).strip()
    if not dataset_label:
        raise ValueError("Table 10 group must define dataset_label.")

    dataset_group = str(config.get("dataset", {}).get("name", "")).strip()
    if not dataset_group:
        raise ValueError("Table 10 group must define dataset.name.")

    rows: list[dict[str, Any]] = []
    for row_config in rows_config:
        no = int(row_config["no"])
        attack_model = str(row_config["attack_model"])
        status = str(row_config.get("status", "planned"))
        if status != "implemented":
            rows.append(
                build_pending_table_10_row(
                    no=no,
                    attack_model=attack_model,
                    dataset=dataset_label,
                )
            )
            continue

        rows.append(
            normalize_table_10_result(
                no=no,
                attack_model=attack_model,
                dataset=dataset_label,
                result=_row_result(row_config),
            )
        )

    save_table_10_outputs(
        rows=rows,
        output_dir=_output_dir(config),
        dataset_group=dataset_group,
        model_group=model_group,
    )
    return rows

