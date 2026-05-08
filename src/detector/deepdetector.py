"""Prediction-change detector for the modern MNIST reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attacks.fgsm import fgsm_attack
from src.detector.config import MnistDetectorConfig, MnistTransformName
from src.detector.transformations import (
    box_mean_filter_chw,
    cross_mean_filter_chw,
    diamond_mean_filter_chw,
    nonuniform_quantization_mnist,
    scalar_quantization_mnist,
)

MNIST_DIAMOND_KERNELS = {
    3: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32),
    5: np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    ),
    7: np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    9: np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
}


@dataclass(frozen=True)
class PredictionChangeResult:
    original_predictions: object
    transformed_predictions: object
    is_adversarial: object
    transformed_images: object


@dataclass(frozen=True)
class MnistDetectorCounts:
    evaluated: int
    original_wrong: int
    attack_failed: int
    tp: int
    fp: int
    fn: int
    ttp: int

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "evaluated": self.evaluated,
            "original_wrong": self.original_wrong,
            "attack_failed": self.attack_failed,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "ttp": self.ttp,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _config_from_transform_args(
    transform: MnistTransformName | MnistDetectorConfig,
    *,
    interval: int = 128,
    left: bool = True,
    kernel_size: int | None = None,
) -> MnistDetectorConfig:
    if isinstance(transform, MnistDetectorConfig):
        return transform
    return MnistDetectorConfig(
        transform=transform,
        interval=interval,
        left=left,
        kernel_size=kernel_size,
    )


def _filter_bounds(kernel_size: int) -> tuple[int, int, int]:
    if kernel_size not in {3, 5, 7, 9}:
        raise ValueError("MNIST filter kernel_size must be one of: 3, 5, 7, 9.")

    start = (kernel_size - 1) // 2
    end = 28 - start
    return start, end, kernel_size * kernel_size


def transform_mnist_batch(
    images,
    transform: MnistTransformName | MnistDetectorConfig,
    *,
    interval: int = 128,
    left: bool = True,
    kernel_size: int | None = None,
):
    """Apply a legacy MNIST detector transform to a NCHW torch batch."""

    import torch

    config = _config_from_transform_args(
        transform,
        interval=interval,
        left=left,
        kernel_size=kernel_size,
    )
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected MNIST NCHW batch with shape (N, 1, 28, 28), got {images.shape}."
        )

    image_array = images.detach().cpu().numpy()
    if config.transform == "uniform":
        if config.interval is None:
            raise ValueError("Uniform quantization requires an interval.")
        transformed = scalar_quantization_mnist(
            image_array,
            interval=config.interval,
            left=config.left,
        )
    elif config.transform == "nonuniform":
        transformed = np.empty_like(image_array, dtype=np.float32)
        for index in range(image_array.shape[0]):
            transformed[index, 0] = nonuniform_quantization_mnist(image_array[index, 0])
    elif config.transform == "box":
        if config.kernel_size is None:
            raise ValueError("Box filter requires a kernel_size.")
        start, end, coefficient = _filter_bounds(config.kernel_size)
        transformed = np.empty_like(image_array, dtype=np.float32)
        for index in range(image_array.shape[0]):
            transformed[index] = box_mean_filter_chw(
                image_array[index],
                start,
                end,
                coefficient,
            )
    elif config.transform == "diamond":
        if config.kernel_size is None:
            raise ValueError("Diamond filter requires a kernel_size.")
        start, end, _ = _filter_bounds(config.kernel_size)
        kernel = MNIST_DIAMOND_KERNELS[config.kernel_size]
        transformed = np.empty_like(image_array, dtype=np.float32)
        for index in range(image_array.shape[0]):
            transformed[index] = diamond_mean_filter_chw(
                image_array[index],
                kernel,
                start,
                end,
                int(kernel.sum()),
            )
    elif config.transform == "cross":
        if config.kernel_size is None:
            raise ValueError("Cross filter requires a kernel_size.")
        start, end, _ = _filter_bounds(config.kernel_size)
        coefficient = 1 + 4 * start
        transformed = np.empty_like(image_array, dtype=np.float32)
        for index in range(image_array.shape[0]):
            transformed[index] = cross_mean_filter_chw(
                image_array[index],
                start,
                end,
                coefficient,
            )
    else:
        raise ValueError(f"Unsupported MNIST transform: {config.transform}.")

    return torch.as_tensor(transformed, dtype=images.dtype, device=images.device)


def detect_prediction_change(
    model,
    images,
    transform: MnistTransformName | MnistDetectorConfig,
    *,
    interval: int = 128,
    left: bool = True,
    kernel_size: int | None = None,
) -> PredictionChangeResult:
    """Detect adversarial samples by comparing predictions before and after transform."""

    import torch

    was_training = model.training
    model.eval()
    with torch.no_grad():
        original_predictions = model(images).argmax(dim=1)
        transformed_images = transform_mnist_batch(
            images,
            transform,
            interval=interval,
            left=left,
            kernel_size=kernel_size,
        )
        transformed_predictions = model(transformed_images).argmax(dim=1)
        is_adversarial = transformed_predictions != original_predictions
    if was_training:
        model.train()

    return PredictionChangeResult(
        original_predictions=original_predictions,
        transformed_predictions=transformed_predictions,
        is_adversarial=is_adversarial,
        transformed_images=transformed_images,
    )


def evaluate_mnist_fgsm_detector(
    model,
    dataloader,
    *,
    epsilon: float = 0.2,
    transform: MnistTransformName | MnistDetectorConfig = "uniform",
    interval: int = 128,
    left: bool = True,
    kernel_size: int | None = None,
    device=None,
) -> MnistDetectorCounts:
    """Evaluate the MNIST detector with generated FGSM adversarial examples."""

    import torch

    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    original_wrong = 0
    attack_failed = 0
    evaluated = 0
    tp = fp = fn = ttp = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)
        clean_correct_mask = clean_predictions == labels
        original_wrong += int((~clean_correct_mask).sum().item())
        if not bool(clean_correct_mask.any()):
            continue

        clean_images = images[clean_correct_mask]
        clean_labels = labels[clean_correct_mask]
        clean_predictions = clean_predictions[clean_correct_mask]
        adversarial_images = fgsm_attack(
            model,
            clean_images,
            clean_labels,
            epsilon=epsilon,
            clip_min=0.0,
            clip_max=1.0,
        )

        with torch.no_grad():
            adversarial_predictions = model(adversarial_images).argmax(dim=1)
        attack_success_mask = adversarial_predictions != clean_predictions
        attack_failed += int((~attack_success_mask).sum().item())
        if not bool(attack_success_mask.any()):
            continue

        clean_images = clean_images[attack_success_mask]
        clean_labels = clean_labels[attack_success_mask]
        clean_predictions = clean_predictions[attack_success_mask]
        adversarial_images = adversarial_images[attack_success_mask]
        adversarial_predictions = adversarial_predictions[attack_success_mask]
        evaluated += int(adversarial_images.shape[0])

        clean_detection = detect_prediction_change(
            model,
            clean_images,
            transform,
            interval=interval,
            left=left,
            kernel_size=kernel_size,
        )
        adversarial_detection = detect_prediction_change(
            model,
            adversarial_images,
            transform,
            interval=interval,
            left=left,
            kernel_size=kernel_size,
        )

        fp += int(clean_detection.is_adversarial.sum().item())
        detected_mask = adversarial_detection.transformed_predictions != adversarial_predictions
        tp += int(detected_mask.sum().item())
        fn += int((~detected_mask).sum().item())
        ttp += int(
            (
                detected_mask
                & (adversarial_detection.transformed_predictions == clean_labels)
            )
            .sum()
            .item()
        )

    if was_training:
        model.train()

    return MnistDetectorCounts(
        evaluated=evaluated,
        original_wrong=original_wrong,
        attack_failed=attack_failed,
        tp=tp,
        fp=fp,
        fn=fn,
        ttp=ttp,
    )
