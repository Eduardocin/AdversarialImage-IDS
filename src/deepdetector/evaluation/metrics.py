"""Small shared metric helpers."""

from __future__ import annotations

from typing import Tuple


def safe_precision_recall_f1(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Return recall, precision, and F1 with zero-safe division."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return float(recall), float(precision), float(f1)
