"""Metrics and report helpers for prediction-change detection."""

from __future__ import print_function

import csv
import os
from typing import Any, Dict, Iterable, List


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


def _summary_rows(metrics: Dict[str, Any], filter_name: str) -> List[Dict[str, Any]]:
    """Normalize either one metrics dict or a mapping of filter metrics."""
    if "TP" in metrics:
        row = dict(metrics)
        row["filter_name"] = filter_name
        return [row]

    rows = []
    for name, values in metrics.items():
        row = dict(values)
        row["filter_name"] = name
        rows.append(row)
    return rows


def _summary_display_rows(metrics: Dict[str, Any], filter_name: str) -> List[Dict[str, Any]]:
    """Return summary rows with derived display fields."""
    rows = []
    for row in _summary_rows(metrics, filter_name):
        display_row = dict(row)
        n_discarded = int(display_row.get("n_discarded_clean_error", 0)) + int(
            display_row.get("n_discarded_attack_failed", 0)
        )
        n_total = int(display_row.get("n_total", 0))
        if not n_total:
            n_total = (
                int(display_row.get("TP", 0))
                + int(display_row.get("FN", 0))
                + n_discarded
            )

        display_row["n_total"] = n_total
        display_row["n_discarded"] = n_discarded
        display_row["n_evaluated"] = n_total - n_discarded
        rows.append(display_row)
    return rows


def save_summary_md(metrics: Dict[str, Any], filter_name: str, path: str) -> str:
    """Save a Markdown table comparing detector metrics by filter."""
    rows = _summary_display_rows(metrics, filter_name)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    lines = [
        "# MNIST Prediction-Change Detector",
        "",
        "| filtro | n_total | n_descartados | TP | FP | FN | TTP | precision | recall | f1 | ttp_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {filter_name} | {n_total} | {n_discarded} | {TP} | {FP} | {FN} | {TTP} | "
            "{precision:.6f} | {recall:.6f} | {f1:.6f} | {ttp_rate:.6f} |".format(
                filter_name=row["filter_name"],
                n_total=int(row["n_total"]),
                n_discarded=int(row["n_discarded"]),
                TP=int(row.get("TP", 0)),
                FP=int(row.get("FP", 0)),
                FN=int(row.get("FN", 0)),
                TTP=int(row.get("TTP", 0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                f1=float(row.get("f1", 0.0)),
                ttp_rate=float(row.get("ttp_rate", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "Descartes incluem erro limpo e ataque FGSM que nao mudou a classe verdadeira.",
            "TTP e o subconjunto dos TP em que a filtragem tambem corrige a classe para o rotulo verdadeiro.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
