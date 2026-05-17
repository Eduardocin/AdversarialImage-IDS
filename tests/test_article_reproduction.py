from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.article_reproduction import (
    evaluate_filter_predictions,
    interval_size,
)


def test_interval_size_uses_article_mapping() -> None:
    """Check interval-count to interval-size mapping."""
    assert interval_size(2) == 128
    assert interval_size(6) == 43
    assert interval_size(10) == 26


def test_interval_size_rejects_unsupported_count() -> None:
    """Unsupported interval counts should fail explicitly."""
    with pytest.raises(ValueError):
        interval_size(11)


def test_evaluate_filter_predictions_matches_article_count_semantics() -> None:
    """Check #F, TP, FN, FP and RTP on a small deterministic example."""
    y_true = np.array([1, 2, 3, 4])
    clean_pred = np.array([1, 2, 3, 4])
    adv_pred = np.array([1, 0, 0, 0])
    filtered_clean_pred = np.array([1, 9, 3, 4])
    filtered_adv_pred = np.array([1, 2, 0, 4])

    metrics = evaluate_filter_predictions(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )

    assert metrics["F"] == 1
    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["RTP"] == 2
    assert metrics["recall_percent"] == pytest.approx(66.6666667)
    assert metrics["precision_percent"] == pytest.approx(66.6666667)
