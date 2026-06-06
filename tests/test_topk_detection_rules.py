from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.topk_detection import (  # noqa: E402
    detect_by_rule,
    detect_confidence_drop,
    detect_rank_displacement,
    detect_top1_change,
    detect_top1_in_topk,
    is_detected_topk,
)


def test_top1_change_detects_when_argmax_changes() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.1, 0.8, 0.1])
    assert detect_top1_change(before, after) is True


def test_top1_change_no_detection_when_argmax_stable() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.7, 0.2, 0.1])
    assert detect_top1_change(before, after) is False


def test_top1_in_topk_detects_when_original_class_expelled() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.1, 0.5, 0.4])
    assert detect_top1_in_topk(before, after, k=2) is True


def test_top1_in_topk_no_detection_when_original_class_remains() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.4, 0.5, 0.1])
    assert detect_top1_in_topk(before, after, k=2) is False


def test_top1_in_topk_k1_equivalent_to_top1_change() -> None:
    rng = np.random.default_rng(99)
    for _ in range(500):
        before = rng.dirichlet(np.ones(10))
        after = rng.dirichlet(np.ones(10))
        assert detect_top1_in_topk(before, after, k=1) == detect_top1_change(before, after)


def test_rank_displacement_at_boundary() -> None:
    before = np.array([0.7, 0.2, 0.1])
    after = np.array([0.1, 0.3, 0.6])
    assert detect_rank_displacement(before, after, threshold=2) is True
    assert detect_rank_displacement(before, after, threshold=3) is False


def test_rank_displacement_equivalence_with_top1_in_topk() -> None:
    rng = np.random.default_rng(42)
    for _ in range(2000):
        before = rng.dirichlet(np.ones(10))
        after = rng.dirichlet(np.ones(10))
        for k in [1, 2, 3, 5]:
            a = detect_top1_in_topk(before, after, k=k)
            b = detect_rank_displacement(before, after, threshold=k)
            assert a == b, "Mismatch at k={0}".format(k)


def test_confidence_drop_detects_via_position() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.1, 0.5, 0.4])
    assert detect_confidence_drop(before, after, k=2, delta=0.99) is True


def test_confidence_drop_detects_via_delta_only() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.6, 0.3, 0.1])
    assert detect_confidence_drop(before, after, k=2, delta=0.10) is True
    assert detect_confidence_drop(before, after, k=2, delta=0.30) is False


def test_confidence_drop_no_detection_below_threshold() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.78, 0.12, 0.1])
    assert detect_confidence_drop(before, after, k=2, delta=0.10) is False


def test_topk_overlap_k1_equivalent_to_top1_change() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        before = rng.dirichlet(np.ones(1000))
        after = rng.dirichlet(np.ones(1000))
        assert is_detected_topk(before, after, k=1) == detect_top1_change(before, after)


def test_detect_by_rule_unknown_raises() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.1, 0.8, 0.1])
    with pytest.raises(ValueError, match="Regra desconhecida"):
        detect_by_rule(before, after, rule="regra_inventada")


def test_detect_by_rule_dispatches_correctly() -> None:
    before = np.array([0.8, 0.1, 0.1])
    after = np.array([0.1, 0.5, 0.4])
    assert detect_by_rule(before, after, rule="top1_change") is True
    assert detect_by_rule(before, after, rule="top1_in_topk", k=2) is True
    assert detect_by_rule(before, after, rule="topk_overlap", k=2) is False
