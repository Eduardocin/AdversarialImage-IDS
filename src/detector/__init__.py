"""Detector transformations and feature extraction utilities."""

from src.detector.config import (
    MNIST_FILTER_KERNEL_SIZES,
    MNIST_QUANTIZATION_INTERVALS,
    MnistDetectorConfig,
    default_mnist_detector_candidates,
)
from src.detector.features import one_d_entropy_mnist, one_d_entropy_uint8_chw
from src.detector.selection import (
    MnistDetectorCandidateResult,
    select_best_mnist_detector,
)
from src.detector.deepdetector import (
    MnistDetectorCounts,
    PredictionChangeResult,
    detect_prediction_change,
    evaluate_mnist_fgsm_detector,
    transform_mnist_batch,
)
from src.detector.transformations import (
    box_mean_filter_chw,
    choose_closer_filter,
    cross_mean_filter_chw,
    diamond_mean_filter_chw,
    find_nonuniform_border_mnist,
    nonuniform_quantization_mnist,
    normalize_values,
    scalar_quantization_mnist,
    scalar_quantization_uint8,
)

__all__ = [
    "box_mean_filter_chw",
    "choose_closer_filter",
    "cross_mean_filter_chw",
    "detect_prediction_change",
    "diamond_mean_filter_chw",
    "default_mnist_detector_candidates",
    "evaluate_mnist_fgsm_detector",
    "find_nonuniform_border_mnist",
    "MNIST_FILTER_KERNEL_SIZES",
    "MNIST_QUANTIZATION_INTERVALS",
    "MnistDetectorCandidateResult",
    "MnistDetectorConfig",
    "MnistDetectorCounts",
    "nonuniform_quantization_mnist",
    "normalize_values",
    "one_d_entropy_mnist",
    "one_d_entropy_uint8_chw",
    "PredictionChangeResult",
    "scalar_quantization_mnist",
    "scalar_quantization_uint8",
    "select_best_mnist_detector",
    "transform_mnist_batch",
]
