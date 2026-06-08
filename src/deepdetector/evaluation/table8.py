"""Evaluation helpers for ImageNet Table 8 validation spatial smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deepdetector.evaluation.imagenet_common import (
    apply_spatial_filter_to_model_input,
    iter_detection_dataset,
    label_to_int,
    predict_one_caffe,
    safe_precision_recall_f1,
)


@dataclass
class Table8FilterResult:
    """Detection counts and metrics for one Table 8 filter candidate."""

    mask_type: str
    size: int
    tp: int
    fn: int
    fp: int
    recall: float
    precision: float
    f1: float
    attack_success: int
    disturbed_failure: int
    skipped_wrong_baseline: int


_label_to_int = label_to_int
_predict_one = predict_one_caffe
_iter_dataset = iter_detection_dataset
_metrics = safe_precision_recall_f1


def _apply_table8_filter_to_model_input(
    image: np.ndarray,
    mask_type: str,
    size: int,
) -> np.ndarray:
    """Apply the shared CHW 0-255 spatial smoothing filter to model input."""
    return apply_spatial_filter_to_model_input(
        image=image,
        mask_type=mask_type,
        size=size,
    )


def evaluate_table8_filter(
    model: Any,
    dataset: Any,
    mask_type: str,
    size: int,
) -> Table8FilterResult:
    """Evaluate one fixed spatial smoothing candidate for ImageNet Table 8."""
    tp = 0
    fn = 0
    fp = 0
    attack_success = 0
    disturbed_failure = 0
    skipped_wrong_baseline = 0

    for clean_image, label, adversarial_image in _iter_dataset(dataset):
        if adversarial_image is None:
            raise ValueError("Table 8 evaluation requires adversarial images.")

        true_label = _label_to_int(label)
        clean_pred = _predict_one(model, clean_image)
        if clean_pred != true_label:
            skipped_wrong_baseline += 1
            continue

        adv_pred = _predict_one(model, adversarial_image)
        if adv_pred == clean_pred:
            disturbed_failure += 1
            continue

        attack_success += 1

        filtered_clean = _apply_table8_filter_to_model_input(
            image=clean_image,
            mask_type=mask_type,
            size=size,
        )
        filtered_clean_pred = _predict_one(model, filtered_clean)
        if filtered_clean_pred != clean_pred:
            fp += 1

        filtered_adv = _apply_table8_filter_to_model_input(
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
    return Table8FilterResult(
        mask_type=str(mask_type),
        size=int(size),
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        recall=recall,
        precision=precision,
        f1=f1,
        attack_success=int(attack_success),
        disturbed_failure=int(disturbed_failure),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
    )
