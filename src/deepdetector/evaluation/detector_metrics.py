"""Metrics and report helpers for prediction-change detection."""

from __future__ import print_function

import csv
import os
from typing import Any, Dict, Iterable, List


def _as_bool(value: Any) -> bool:
    """Parse booleans that may have been read back from CSV-like records."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def compute_detector_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Compute detector counts over clean/adversarial pair records."""
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "TTP": 0}
    for record in records:
        detected = _as_bool(record.get("detected", False))
        false_positive = _as_bool(record.get("false_positive", False))
        corrected = _as_bool(record.get("corrected", False))

        if detected:
            counts["TP"] += 1
            if corrected:
                counts["TTP"] += 1
        else:
            counts["FN"] += 1

        if false_positive:
            counts["FP"] += 1
        else:
            counts["TN"] += 1
    return counts


def compute_precision_recall(counts: Dict[str, int]) -> Dict[str, float]:
    """Compute precision and recall from detector counts."""
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    tn = int(counts.get("TN", 0))
    ttp = int(counts.get("TTP", 0))

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    fpr_denominator = fp + tn

    precision = tp / float(precision_denominator) if precision_denominator else 0.0
    recall = tp / float(recall_denominator) if recall_denominator else 0.0
    false_positive_rate = fp / float(fpr_denominator) if fpr_denominator else 0.0
    correction_rate = ttp / float(tp) if tp else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "correction_rate": float(correction_rate),
    }


def save_results_csv(records: Iterable[Dict[str, Any]], path: str) -> str:
    """Save per-example detector records to CSV."""
    rows = list(records)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    preferred = [
        "filter_name",
        "epsilon",
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "filtered_clean_pred",
        "filtered_adv_pred",
        "detected",
        "false_positive",
        "corrected",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = preferred + extra

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _summary_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a single metrics dict or a mapping of metrics dicts."""
    if "TP" in metrics or "precision" in metrics:
        row = dict(metrics)
        row.setdefault("filter_name", "all")
        row.setdefault("evaluated_pairs", int(row.get("TP", 0)) + int(row.get("FN", 0)))
        return [row]

    rows = []
    for filter_name, values in sorted(metrics.items()):
        row = dict(values)
        row.setdefault("filter_name", filter_name)
        row.setdefault("evaluated_pairs", int(row.get("TP", 0)) + int(row.get("FN", 0)))
        rows.append(row)
    return rows


def save_summary_md(metrics: Dict[str, Any], path: str) -> str:
    """Save aggregate detector metrics as a Markdown table."""
    rows = _summary_rows(metrics)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    lines = [
        "# MNIST Prediction-Change Detector",
        "",
        "| filter | pairs | TP | FP | FN | TN | TTP | precision | recall | false_positive_rate | correction_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {filter_name} | {evaluated_pairs} | {TP} | {FP} | {FN} | {TN} | {TTP} | "
            "{precision:.6f} | {recall:.6f} | {false_positive_rate:.6f} | "
            "{correction_rate:.6f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "TP counts adversarial examples detected by prediction change after filtering.",
            "FP counts clean examples whose prediction changed after filtering.",
            "FN counts adversarial examples that kept the same prediction after filtering.",
            "TTP counts detected adversarial examples whose filtered prediction returned to the true label.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
