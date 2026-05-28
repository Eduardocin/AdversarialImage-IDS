"""Standard experiment output writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from deepdetector.io.paths import ensure_dir


def write_metrics_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    """Write experiment metrics to CSV with stable column order."""
    csv_path = Path(path)
    if csv_path.parent:
        ensure_dir(csv_path.parent)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return csv_path


def write_metrics_json(
    path: Path,
    payload: Dict[str, Any],
) -> Path:
    """Write experiment metadata and metrics to JSON."""
    json_path = Path(path)
    if json_path.parent:
        ensure_dir(json_path.parent)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return json_path


def write_experiment_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    csv_fields: Sequence[str],
    metadata: Dict[str, Any],
    csv_name: str = "metrics.csv",
    json_name: str = "metrics.json",
) -> Dict[str, Path]:
    """Write the standard CSV and JSON outputs for an experiment."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / csv_name, rows, csv_fields)
    json_path = write_metrics_json(output_path / json_name, metadata)
    return {"csv": csv_path, "json": json_path}

