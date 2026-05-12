"""Metrics for the MNIST DeepDetector replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from .detection import DetectionDecision

MetricValue = Union[float, int]


@dataclass
class DetectorCounts:
    """Counters reported by the original MNIST scripts."""

    test_number: int = 0
    original_classified_wrong_number: int = 0
    disturbed_failure_number: int = 0
    tp: int = 0
    fn: int = 0
    fp: int = 0
    ttp: int = 0

    def update_detection(self, decision: DetectionDecision) -> None:
        """Update detector counters from one evaluated adversarial sample."""

        self.test_number += 1
        if decision.adversarial_detected:
            self.tp += 1
            if decision.adversarial_restored:
                self.ttp += 1
        else:
            self.fn += 1

        if decision.false_positive:
            self.fp += 1

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    def as_dict(self) -> Dict[str, MetricValue]:
        """Return counters and metrics in a serializable shape."""

        return {
            "test_number": self.test_number,
            "original_classified_wrong_number": self.original_classified_wrong_number,
            "disturbed_failure_number": self.disturbed_failure_number,
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "ttp": self.ttp,
            "recall": self.recall,
            "precision": self.precision,
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Result object returned by MNIST experiment runners."""

    attack_name: str
    counts: DetectorCounts

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable experiment summary."""

        return {
            "attack_name": self.attack_name,
            "metrics": self.counts.as_dict(),
        }
