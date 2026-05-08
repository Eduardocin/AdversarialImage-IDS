"""Detector configuration objects and candidate grids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

MnistTransformName = Literal["uniform", "nonuniform", "box", "diamond", "cross"]

# Same scalar quantization grid used by OwenSec/DeepDetector's FGSM training scripts.
MNIST_QUANTIZATION_INTERVALS = [128, 85, 64, 51, 43, 37, 32, 28, 26]
MNIST_FILTER_KERNEL_SIZES = [3, 5, 7, 9]


@dataclass(frozen=True)
class MnistDetectorConfig:
    """A single MNIST detector preprocessing configuration."""

    transform: MnistTransformName
    interval: int | None = None
    left: bool = True
    kernel_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MnistDetectorConfig":
        return cls(
            transform=payload["transform"],
            interval=payload.get("interval"),
            left=payload.get("left", True),
            kernel_size=payload.get("kernel_size"),
        )


def default_mnist_detector_candidates() -> list[MnistDetectorConfig]:
    """Return the MNIST candidate grid used in the original training scripts."""

    candidates: list[MnistDetectorConfig] = [
        MnistDetectorConfig("uniform", interval=interval)
        for interval in MNIST_QUANTIZATION_INTERVALS
    ]
    candidates.append(MnistDetectorConfig("nonuniform"))
    candidates.extend(
        MnistDetectorConfig("box", kernel_size=kernel_size)
        for kernel_size in MNIST_FILTER_KERNEL_SIZES
    )
    candidates.extend(
        MnistDetectorConfig("diamond", kernel_size=kernel_size)
        for kernel_size in MNIST_FILTER_KERNEL_SIZES
    )
    candidates.extend(
        MnistDetectorConfig("cross", kernel_size=kernel_size)
        for kernel_size in MNIST_FILTER_KERNEL_SIZES
    )
    return candidates
