"""Detection rules for MNIST adversarial examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionDecision:
    """Prediction comparison before and after noise reduction."""

    original_label: int
    adversarial_label: int
    filtered_original_label: int
    filtered_adversarial_label: int
    true_label: int

    @property
    def adversarial_detected(self) -> bool:
        return self.adversarial_label != self.filtered_adversarial_label

    @property
    def adversarial_restored(self) -> bool:
        return self.adversarial_detected and self.filtered_adversarial_label == self.true_label

    @property
    def false_positive(self) -> bool:
        return self.original_label != self.filtered_original_label


def is_detected(before_label: int, after_label: int) -> bool:
    """Return true when filtering changes the predicted class."""

    return before_label != after_label

