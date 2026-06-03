"""CSV writers for article reproduction outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PIVOT_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("Recall", "recall"),
    ("Precision", "precision"),
    ("F1 Score", "f1"),
)


def write_pivot_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    metric_rows: Sequence[tuple[str, str]] | None = None,
) -> Path:
    """Write article-style metric rows with one column per filter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_column = {
        "{0}_{1}x{1}".format(row["mask_type"], int(row["size"])): row
        for row in rows
    }
    selected_metrics = metric_rows or DEFAULT_PIVOT_METRIC_ROWS

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + list(columns))
        for display_name, field in selected_metrics:
            writer.writerow(
                [display_name]
                + [
                    "{0:.6f}".format(float(by_column[column][field]))
                    if column in by_column
                    else ""
                    for column in columns
                ]
            )
    return path
