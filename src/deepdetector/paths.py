"""Shared project paths for generated artifacts and experiment results."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
ADVERSARIAL_ARTIFACTS_DIR = ARTIFACTS_DIR / "adversarial_examples"

MNIST_RESULTS_DIR = RESULTS_DIR / "mnist"
MNIST_MODEL_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "mnist"
MNIST_ADVERSARIAL_ARTIFACTS_DIR = ADVERSARIAL_ARTIFACTS_DIR / "mnist"

MNIST_M1_CHECKPOINT_DIR = (
    MNIST_MODEL_ARTIFACTS_DIR / "m1" / "clean_baseline" / "checkpoints"
)
MNIST_M2_CHECKPOINT_DIR = (
    MNIST_MODEL_ARTIFACTS_DIR / "m2" / "clean_baseline" / "checkpoints"
)

MNIST_M1_ADVERSARIAL_DIR = MNIST_ADVERSARIAL_ARTIFACTS_DIR / "m1"
MNIST_M2_ADVERSARIAL_DIR = MNIST_ADVERSARIAL_ARTIFACTS_DIR / "m2"

MNIST_M2_RESULTS_DIR = MNIST_RESULTS_DIR / "m2_cw"
