"""Experiment orchestration for the MNIST DeepDetector flow."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .attacks import generate_cw_l2, generate_cw_linf, generate_fgsm
from .config import MnistExperimentConfig
from .data import detector_test_slice, generate_untargeted_data
from .detection import DetectionDecision
from .evaluation import DetectorCounts, ExperimentResult
from .models import predict_label
from .noise_reduction import adaptive_reduce


def _evaluate_pairs(
    model: Any,
    originals: np.ndarray,
    labels: np.ndarray,
    adversarial: np.ndarray,
    *,
    offset: float,
    config: MnistExperimentConfig,
) -> DetectorCounts:
    counts = DetectorCounts()
    for index in range(len(adversarial)):
        true_label = int(np.argmax(labels[index]))
        original_label = predict_label(model, originals[index])
        if original_label != true_label:
            counts.original_classified_wrong_number += 1
            continue

        adversarial_label = predict_label(model, adversarial[index])
        if adversarial_label == original_label:
            counts.disturbed_failure_number += 1
            continue

        filtered_original = adaptive_reduce(
            originals[index].reshape(28, 28),
            config.detection,
            offset=offset,
        )
        filtered_adversarial = adaptive_reduce(
            adversarial[index].reshape(28, 28),
            config.detection,
            offset=offset,
        )
        decision = DetectionDecision(
            original_label=original_label,
            adversarial_label=adversarial_label,
            filtered_original_label=predict_label(model, filtered_original),
            filtered_adversarial_label=predict_label(model, filtered_adversarial),
            true_label=true_label,
        )
        counts.update_detection(decision)
    return counts


def run_fgsm_experiment(
    sess: Any,
    x: Any,
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: Optional[MnistExperimentConfig] = None,
) -> ExperimentResult:
    """Run the M1/FGSM detector flow using CleverHans."""

    cfg = config or MnistExperimentConfig()
    originals, labels = detector_test_slice(test_images, test_labels, cfg.splits)
    adversarial = generate_fgsm(sess, x, model, originals, cfg.attacks.fgsm_test_eps)
    counts = _evaluate_pairs(model, originals, labels, adversarial, offset=0.0, config=cfg)
    return ExperimentResult(attack_name="fgsm", counts=counts)


def _run_cw_experiment(
    attack_name: str,
    attack_fn: Callable[..., np.ndarray],
    sess: Any,
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: MnistExperimentConfig,
) -> ExperimentResult:
    originals, labels = generate_untargeted_data(
        test_images,
        test_labels,
        samples=config.source_samples,
        start=config.splits.detector_test_start,
    )
    adversarial = attack_fn(
        sess,
        model,
        originals,
        labels,
        config.paths.nn_robust_attacks_root,
        config.attacks,
    )
    counts = _evaluate_pairs(model, originals, labels, adversarial, offset=0.5, config=config)
    return ExperimentResult(attack_name=attack_name, counts=counts)


def run_cw_l2_experiment(
    sess: Any,
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: Optional[MnistExperimentConfig] = None,
) -> ExperimentResult:
    """Run the M2/C&W L2 detector flow using Carlini's implementation."""

    cfg = config or MnistExperimentConfig()
    return _run_cw_experiment("cw_l2", generate_cw_l2, sess, model, test_images, test_labels, cfg)


def run_cw_linf_experiment(
    sess: Any,
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config: Optional[MnistExperimentConfig] = None,
) -> ExperimentResult:
    """Run the M2/C&W Linf detector flow using Carlini's implementation."""

    cfg = config or MnistExperimentConfig()
    return _run_cw_experiment("cw_linf", generate_cw_linf, sess, model, test_images, test_labels, cfg)
