"""Evaluation helpers for ImageNet Table 7 spatial smoothing filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deepdetector.attacks.fgsm_imagenet import (
    generate_fgsm_caffe_image,
    uses_caffe_scale,
)
from deepdetector.evaluation.imagenet_common import (
    apply_spatial_filter_to_model_input,
    image_to_chw_255,
    iter_detection_dataset,
    label_to_int,
    predict_one_caffe,
    safe_precision_recall_f1,
)
from deepdetector.filters.entropy import image_entropy_255_chw


@dataclass
class Table7FilterResult:
    """Detection counts and metrics for one Table 7 filter candidate."""

    mask_type: str
    size: int
    tp: int
    fn: int
    fp: int
    recall: float
    precision: float
    f1: float
    total_images: int
    clean_correct: int
    attack_success: int
    n_high_entropy_clean: int
    n_high_entropy_adversarial: int
    skipped_low_entropy_clean: int
    disturbed_failure: int
    skipped_wrong_baseline: int


_label_to_int = label_to_int
_predict_one = predict_one_caffe
_iter_dataset = iter_detection_dataset
_image_to_chw_255 = image_to_chw_255
_metrics = safe_precision_recall_f1


def _entropy_for_image(image: np.ndarray) -> float:
    """Compute Table 7 entropy for a model input image."""
    chw_255, _, _ = _image_to_chw_255(image)
    return image_entropy_255_chw(chw_255)


def _apply_table7_filter_to_model_input(
    image: np.ndarray,
    mask_type: str,
    size: int,
) -> np.ndarray:
    """Apply the CHW 0-255 Table 7 filter and restore the model input format."""
    return apply_spatial_filter_to_model_input(
        image=image,
        mask_type=mask_type,
        size=size,
    )


def _generate_adversarial_image(
    model: Any,
    image: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Generate one FGSM adversarial example when model attack handles exist."""
    if hasattr(model, "gradient"):
        class_id = _predict_one(model, image)
        image_array = np.asarray(image, dtype=np.float32)
        epsilon_255 = float(epsilon) * 255.0
        return generate_fgsm_caffe_image(
            model=model,
            image=image_array,
            class_id=class_id,
            epsilon_255=epsilon_255,
            clip_min=0.0,
            clip_max=255.0 if uses_caffe_scale(image_array) else 1.0,
        )

    if hasattr(model, "generate_fgsm"):
        generated = model.generate_fgsm(
            np.asarray(image, dtype=np.float32).reshape((1,) + image.shape),
            epsilon,
        )
        return np.asarray(generated, dtype=np.float32).reshape(image.shape)

    raise ValueError(
        "FGSM generation requires dataset adversarial images or a Caffe model "
        "with a gradient(image, class_id) method."
    )


def evaluate_table7_filter(
    model: Any,
    dataset: Any,
    mask_type: str,
    size: int,
    epsilon: float,
    entropy_threshold: float = 5.0,
) -> Table7FilterResult:
    """Evaluate one spatial smoothing candidate for ImageNet Table 7."""
    tp = 0
    fn = 0
    fp = 0
    total_images = 0
    clean_correct = 0
    attack_success = 0
    n_high_entropy_clean = 0
    n_high_entropy_adversarial = 0
    skipped_low_entropy_clean = 0
    disturbed_failure = 0
    skipped_wrong_baseline = 0

    for clean_image, label, provided_adversarial in _iter_dataset(dataset):
        total_images += 1
        true_label = _label_to_int(label)
        clean_pred = _predict_one(model, clean_image)
        if clean_pred != true_label:
            skipped_wrong_baseline += 1
            continue
        clean_correct += 1

        adversarial_image = provided_adversarial
        if adversarial_image is None:
            adversarial_image = _generate_adversarial_image(
                model=model,
                image=clean_image,
                epsilon=epsilon,
            )

        adv_pred = _predict_one(model, adversarial_image)
        if adv_pred == clean_pred:
            disturbed_failure += 1
            continue

        attack_success += 1
        clean_entropy = _entropy_for_image(clean_image)
        adversarial_entropy = _entropy_for_image(adversarial_image)
        threshold = float(entropy_threshold)

        if clean_entropy <= threshold:
            skipped_low_entropy_clean += 1
            continue

        n_high_entropy_clean += 1
        if adversarial_entropy > threshold:
            n_high_entropy_adversarial += 1

        filtered_clean = _apply_table7_filter_to_model_input(
            image=clean_image,
            mask_type=mask_type,
            size=size,
        )
        filtered_clean_pred = _predict_one(model, filtered_clean)
        if filtered_clean_pred != clean_pred:
            fp += 1

        filtered_adv = _apply_table7_filter_to_model_input(
            image=adversarial_image,
            mask_type=mask_type,
            size=size,
        )
        filtered_adv_pred = _predict_one(model, filtered_adv)
        if filtered_adv_pred != adv_pred:
            tp += 1
        else:
            fn += 1

    recall, precision, f1 = _metrics(tp=tp, fn=fn, fp=fp)
    return Table7FilterResult(
        mask_type=str(mask_type),
        size=int(size),
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        recall=recall,
        precision=precision,
        f1=f1,
        total_images=int(total_images),
        clean_correct=int(clean_correct),
        attack_success=int(attack_success),
        n_high_entropy_clean=int(n_high_entropy_clean),
        n_high_entropy_adversarial=int(n_high_entropy_adversarial),
        skipped_low_entropy_clean=int(skipped_low_entropy_clean),
        disturbed_failure=int(disturbed_failure),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
    )
