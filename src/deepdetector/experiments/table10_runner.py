"""Experiment-runner adapter for Table 10 model groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepdetector.evaluation.tables.table_10 import TABLE_10_SCHEMA, run_table_10_group


TABLE10_SCHEMA = TABLE_10_SCHEMA


def run_table10_group_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one Table 10 group and return a compact runner result."""
    rows = run_table_10_group(config)
    output_dir = Path(config["output"]["dir"])
    return {
        "status": "completed",
        "model_group": config["model_group"],
        "n_rows": len(rows),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "metrics_json": str(output_dir / "metrics.json"),
    }
