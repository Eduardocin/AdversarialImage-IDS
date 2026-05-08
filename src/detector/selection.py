"""Detector candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass

from src.detector.config import MnistDetectorConfig
from src.detector.deepdetector import MnistDetectorCounts


@dataclass(frozen=True)
class MnistDetectorCandidateResult:
    config: MnistDetectorConfig
    counts: MnistDetectorCounts

    @property
    def score(self) -> tuple[float, float, float, int]:
        """Rank by F1, then precision, recall, and lower false positives."""

        return (
            self.counts.f1,
            self.counts.precision,
            self.counts.recall,
            -self.counts.fp,
        )

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "metrics": self.counts.to_dict(),
        }


def select_best_mnist_detector(
    results: list[MnistDetectorCandidateResult],
) -> MnistDetectorCandidateResult:
    if not results:
        raise ValueError("Cannot select a detector from an empty result list.")
    return max(results, key=lambda result: result.score)
