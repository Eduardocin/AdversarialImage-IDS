"""Small metric helpers shared by detector experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionCounts:
    tp: int
    fp: int
    fn: int
    tn: int


def classification_counts(y_true: np.ndarray, y_pred: np.ndarray) -> DetectionCounts:
    """Return binary detection counts for 0/1 labels."""

    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}.")

    return DetectionCounts(
        tp=int(np.sum(true & pred)),
        fp=int(np.sum(~true & pred)),
        fn=int(np.sum(true & ~pred)),
        tn=int(np.sum(~true & ~pred)),
    )


def precision_recall(counts: DetectionCounts) -> dict[str, float]:
    """Compute precision and recall with zero-safe denominators."""

    precision_denominator = counts.tp + counts.fp
    recall_denominator = counts.tp + counts.fn
    precision = counts.tp / precision_denominator if precision_denominator else 0.0
    recall = counts.tp / recall_denominator if recall_denominator else 0.0
    return {"precision": precision, "recall": recall}
