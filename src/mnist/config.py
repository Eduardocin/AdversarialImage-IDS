"""Configuration objects for the MNIST DeepDetector replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MnistPaths:
    """Canonical project paths for the MNIST experiment."""

    project_root: Path
    data_root: Path
    outputs_root: Path
    nn_robust_attacks_root: Path

    @classmethod
    def from_project_root(cls, project_root: Optional[Path] = None) -> "MnistPaths":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        return cls(
            project_root=root,
            data_root=root / "data" / "mnist",
            outputs_root=root / "outputs" / "mnist",
            nn_robust_attacks_root=root / "third_party" / "nn_robust_attacks",
        )

    @property
    def m1_checkpoint_dir(self) -> Path:
        return self.outputs_root / "m1"

    @property
    def m2_weights_path(self) -> Path:
        return self.nn_robust_attacks_root / "models" / "mnist"

    @property
    def carlini_data_dir(self) -> Path:
        return self.nn_robust_attacks_root / "data"


@dataclass(frozen=True)
class MnistSplitConfig:
    """Index ranges used by the original MNIST scripts."""

    train_start: int = 0
    train_end: int = 60000
    test_start: int = 0
    test_end: int = 10000
    detector_train_end: int = 4500
    validation_start: int = 4500
    validation_end: int = 5500
    detector_test_start: int = 5500


@dataclass(frozen=True)
class M1Config:
    """Training defaults for the CleverHans MNIST M1 classifier."""

    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 0.001
    filename: str = "mnist.ckpt"
    load_model: bool = True


@dataclass(frozen=True)
class M2Config:
    """Training defaults for Carlini's MNIST M2 classifier."""

    epochs: int = 50
    batch_size: int = 128
    keras_verbose: int = 0


@dataclass(frozen=True)
class AttackConfig:
    """Attack defaults preserved from the DeepDetector MNIST references."""

    fgsm_train_eps: float = 0.2
    fgsm_test_eps: float = 0.3
    cw_l2_max_iterations: int = 2000
    cw_l2_confidence: float = 0.0
    cw_l2_binary_search_steps: int = 5
    cw_l2_initial_const: float = 1.0
    cw_l2_learning_rate: float = 1e-1
    cw_l2_targeted: bool = False
    cw_linf_max_iterations: int = 1000
    cw_linf_targeted: bool = False


@dataclass(frozen=True)
class DetectionConfig:
    """Adaptive noise reduction thresholds used by the MNIST detector."""

    low_entropy_threshold: float = 4.0
    mid_entropy_threshold: float = 5.0
    low_entropy_interval: int = 128
    mid_entropy_interval: int = 64
    high_entropy_interval: int = 43
    cross_start: int = 3
    cross_end: int = 25
    cross_coefficient: int = 13


@dataclass(frozen=True)
class MnistExperimentConfig:
    """Top-level MNIST experiment configuration."""

    paths: MnistPaths = field(default_factory=MnistPaths.from_project_root)
    splits: MnistSplitConfig = field(default_factory=MnistSplitConfig)
    m1: M1Config = field(default_factory=M1Config)
    m2: M2Config = field(default_factory=M2Config)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    source_samples: int = 1000
