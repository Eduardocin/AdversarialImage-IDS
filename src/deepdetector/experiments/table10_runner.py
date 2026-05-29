"""Runner for Table 10 model-group output scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json


TABLE10_SCHEMA: Sequence[str] = (
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
)

TABLE10_METRIC_FIELDS: Sequence[str] = (
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
)


def _output_dir(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = resolve_project_path(output_config.get("dir"))
    if output_dir is None:
        raise ValueError("Config must define output.dir.")
    return output_dir


def _metric_value(row_config: Dict[str, Any], field: str) -> Any:
    """Return an optional Table 10 metric value."""
    if field in row_config:
        return row_config[field]
    metrics = row_config.get("metrics", {})
    if isinstance(metrics, dict) and field in metrics:
        return metrics[field]
    return None


def _table_row(config: Dict[str, Any], row_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return one Table 10 row in the official schema."""
    row: Dict[str, Any] = {
        "no": int(row_config["no"]),
        "attack_model": str(row_config["attack_model"]),
        "dataset": str(row_config.get("dataset", config.get("dataset_label", ""))),
    }
    for field in TABLE10_METRIC_FIELDS:
        row[field] = _metric_value(row_config, field)
    return row


def _row_statuses(rows_config: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return compact row execution statuses for the manifest."""
    statuses: List[Dict[str, Any]] = []
    for row_config in rows_config:
        entry: Dict[str, Any] = {
            "no": int(row_config["no"]),
            "status": str(row_config.get("status", "planned")),
        }
        if row_config.get("blocked_reason"):
            entry["blocked_reason"] = str(row_config["blocked_reason"])
        statuses.append(entry)
    return statuses


def _blocked_reason(rows_config: Sequence[Dict[str, Any]]) -> Any:
    """Return the shared blocked reason when all rows are blocked for one reason."""
    blocked_rows = [
        row for row in rows_config if str(row.get("status", "")) == "blocked"
    ]
    if len(blocked_rows) != len(rows_config) or not rows_config:
        return None
    reasons = {
        str(row.get("blocked_reason", "")).strip()
        for row in blocked_rows
        if str(row.get("blocked_reason", "")).strip()
    }
    if len(reasons) == 1:
        return next(iter(reasons))
    return None


def _manifest_payload(
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Return the Table 10 per-model manifest payload."""
    rows_config = config.get("rows", [])
    payload: Dict[str, Any] = {
        "table": 10,
        "experiment_id": config["experiment_id"],
        "kind": config["kind"],
        "model_group": config["model_group"],
        "dataset": config["dataset_label"],
        "rows": [int(row["no"]) for row in rows],
        "schema": list(TABLE10_SCHEMA),
        "row_statuses": _row_statuses(rows_config),
        "outputs": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
        },
    }
    reason = _blocked_reason(rows_config)
    if reason is not None:
        payload["status"] = "blocked"
        payload["blocked_reason"] = reason
    return payload


def run_table10_group_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Write the configured Table 10 model-group outputs."""
    rows_config = list(config.get("rows", []))
    if not rows_config:
        raise ValueError("Table 10 group must define rows.")
    if not config.get("model_group"):
        raise ValueError("Table 10 group must define model_group.")
    if not config.get("dataset_label"):
        raise ValueError("Table 10 group must define dataset_label.")

    output_dir = ensure_dir(_output_dir(config))
    rows = [_table_row(config, row_config) for row_config in rows_config]
    metrics_csv = write_metrics_csv(output_dir / "metrics.csv", rows, TABLE10_SCHEMA)
    metrics_payload = {
        "table": 10,
        "experiment_id": config["experiment_id"],
        "kind": config["kind"],
        "model_group": config["model_group"],
        "dataset": config["dataset_label"],
        "schema": list(TABLE10_SCHEMA),
        "metrics": rows,
    }
    metrics_json = write_metrics_json(output_dir / "metrics.json", metrics_payload)
    manifest_payload = _manifest_payload(config, rows, output_dir)
    manifest_json = write_metrics_json(output_dir / "manifest.json", manifest_payload)

    return {
        "status": manifest_payload.get("status", "completed"),
        "model_group": config["model_group"],
        "n_rows": len(rows),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "manifest_json": str(manifest_json),
    }
