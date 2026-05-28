"""Metrics and report helpers for prediction-change detection."""

from __future__ import print_function

import csv
import os
from typing import Any, Dict, Iterable


def _as_bool(value: Any) -> bool:
    """Parse booleans that may come from Python values or CSV strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def compute_detector_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Compute TP/FP/FN/TN/TTP and discard counts from detector records."""
    counts = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "TTP": 0,
        "n_discarded_clean_error": 0,
        "n_discarded_attack_failed": 0,
    }

    for record in records:
        if _as_bool(record.get("discarded_clean_error", False)):
            counts["n_discarded_clean_error"] += 1
            continue
        if _as_bool(record.get("discarded_attack_failed", False)):
            counts["n_discarded_attack_failed"] += 1
            continue

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
        elif not detected:
            counts["TN"] += 1

    return counts


def compute_precision_recall(counts: Dict[str, int]) -> Dict[str, float]:
    """Compute precision, recall, F1 and TTP rate with zero-safe division."""
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    ttp = int(counts.get("TTP", 0))

    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    ttp_rate = ttp / float(tp) if tp else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ttp_rate": float(ttp_rate),
    }


def save_results_csv(records: Iterable[Dict[str, Any]], path: str) -> str:
    """Save one CSV row per evaluated clean/adversarial pair."""
    rows = list(records)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    preferred = [
        "filter_name",
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "filtered_clean_pred",
        "filtered_adv_pred",
        "detected",
        "corrected",
        "false_positive",
        "discarded_clean_error",
        "discarded_attack_failed",
        "discard_reason",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = preferred + extra

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


