"""Evaluation helpers for ImageNet Table 4 scalar quantization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import (
    fgsm_changed_pixels,
    fgsm_linf_255,
    generate_fgsm_caffe_image,
    predict_caffe_label,
    preprocess_caffe_inputs,
    uses_caffe_scale,
)
from deepdetector.evaluation.article_reproduction import interval_size, scalar_filter_for_intervals
from deepdetector.filters.entropy import image_entropy_255_chw, one_d_entropy


logger = logging.getLogger(__name__)


TABLE4_INTERVALS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)
ZERO_ATTACK_SUCCESS_MESSAGE = (
    "FGSM did not generate any successful adversarial example.\n"
    "Check epsilon scale, preprocessing, gradient sign, model wrapper, or labels."
)
TABLE4_INTERVAL_LABELS: Tuple[str, ...] = tuple(str(value) for value in TABLE4_INTERVALS)
TABLE4_OUTPUT_HEADER: Tuple[str, ...] = ("Dataset", "Metric") + TABLE4_INTERVAL_LABELS
TABLE4_METRICS: Tuple[Tuple[str, str], ...] = (
    ("recall", "Recall"),
    ("precision", "Precision"),
    ("f1", "F1 Score"),
)
TABLE4_DATASET_LABEL = "ImageNet"
DIAGNOSTIC_FIELDS: Tuple[str, ...] = (
    "image_id",
    "class_name",
    "true_label",
    "clean_pred",
    "adv_pred",
    "clean_correct",
    "was_skipped",
    "skip_reason",
    "attack_success",
    "disturbed_failure",
    "entropy_clean",
    "entropy_adv",
    "fgsm_linf_255",
    "fgsm_changed_pixels",
)


@dataclass(frozen=True)
class Table4Sample:
    """One ImageNet sample used by the Table 4 reproduction."""

    image: np.ndarray
    true_label: int
    image_id: str
    class_name: str


@dataclass(frozen=True)
class _ValidAttack:
    clean_image: np.ndarray
    adversarial_image: np.ndarray
    clean_pred: int
    adv_pred: int


@dataclass(frozen=True)
class Table4Evaluation:
    """Rows and diagnostics produced by the Table 4 evaluation."""

    rows: List[dict[str, Any]]
    diagnostics: List[dict[str, Any]]
    n_clean_total: int
    n_clean_correct: int
    n_attack_success: int
    disturbed_failure: int
    skipped_wrong_baseline: int


def _predict_one(model: Any, image: np.ndarray) -> int:
    """Predict a single image with a model exposing ``predict_label``."""
    return predict_caffe_label(model, image)


def _article_model_input(model: Any, image: np.ndarray) -> np.ndarray:
    """Return the image in the same Caffe input space used by the article."""
    return preprocess_caffe_inputs(model, image)[0]


def _metrics(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Return recall, precision, and F1 with zero-safe division."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return float(recall), float(precision), float(f1)


def _progress_every(total: int) -> int:
    """Return an interval for progress logging based on total count."""
    if total <= 0:
        return 1
    return 20


def _uses_caffe_scale(image: np.ndarray) -> bool:
    """Return whether image data appears to be Caffe 0-255 input data."""
    return uses_caffe_scale(image)


def _entropy_for_image(image: np.ndarray) -> float:
    """Compute entropy using the correct scale for normalized or Caffe images."""
    image_array = np.asarray(image, dtype=np.float32)
    if _uses_caffe_scale(image_array):
        if image_array.ndim == 3 and image_array.shape[0] == 3:
            return image_entropy_255_chw(image_array)
        if image_array.ndim == 3 and image_array.shape[-1] == 3:
            return image_entropy_255_chw(np.transpose(image_array, (2, 0, 1)))
    return one_d_entropy(image_array)


def _quantize_for_intervals(image: np.ndarray, intervals: int) -> np.ndarray:
    """Apply scalar quantization in the image's native scale."""
    step = interval_size(intervals)
    image_array = np.asarray(image, dtype=np.float32)
    if _uses_caffe_scale(image_array):
        quantized = np.clip(image_array, 0.0, 255.0)
        quantized //= step
        quantized *= step
        return quantized.astype(np.float32)
    return scalar_filter_for_intervals(intervals)(image_array)


def _fgsm_linf_255(clean_image: np.ndarray, adversarial_image: np.ndarray) -> float:
    """Return Linf perturbation size in 0-255 pixel units."""
    return fgsm_linf_255(clean_image, adversarial_image)


def _fgsm_changed_pixels(clean_image: np.ndarray, adversarial_image: np.ndarray) -> int:
    """Return number of elements changed by FGSM."""
    return fgsm_changed_pixels(clean_image, adversarial_image)


def generate_fgsm_from_gradient(
    model: Any,
    image: np.ndarray,
    class_id: int,
    epsilon_255: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """Generate one normalized ImageNet FGSM image using a model input gradient."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must have shape (H, W, C).")

    effective_clip_max = 255.0 if clip_max <= 1.0 and _uses_caffe_scale(image_array) else clip_max
    return generate_fgsm_caffe_image(
        model=model,
        image=image_array,
        class_id=class_id,
        epsilon_255=epsilon_255,
        clip_min=clip_min,
        clip_max=effective_clip_max,
    )


def _evaluate_interval(model: Any, attacks: Sequence[_ValidAttack], intervals: int) -> Tuple[int, int, int]:
    """Return TP, FN, and FP counts for one scalar quantization interval count."""
    tp = 0
    fn = 0
    fp = 0

    for attack in attacks:
        filtered_clean = _quantize_for_intervals(attack.clean_image, intervals)
        filtered_adv = _quantize_for_intervals(attack.adversarial_image, intervals)
        filtered_clean_pred = _predict_one(model, filtered_clean)
        filtered_adv_pred = _predict_one(model, filtered_adv)

        if filtered_clean_pred != attack.clean_pred:
            fp += 1
        if filtered_adv_pred != attack.adv_pred:
            tp += 1
        else:
            fn += 1

    return int(tp), int(fn), int(fp)


def evaluate_table4_imagenet(
    model: Any,
    samples: Sequence[Table4Sample],
    intervals: Iterable[int] = TABLE4_INTERVALS,
    epsilon_255: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Table4Evaluation:
    """Evaluate ImageNet Table 4 detection counts for scalar quantization intervals."""
    interval_values = [int(value) for value in intervals]
    n_clean_total = len(samples)
    n_clean_correct = 0
    n_attack_success = 0
    disturbed_failure = 0
    skipped_wrong_baseline = 0
    diagnostics: List[dict[str, Any]] = []
    valid_attacks: List[_ValidAttack] = []

    progress_every = _progress_every(n_clean_total)

    for sample_index, sample in enumerate(samples, start=1):
        clean_image = _article_model_input(model, sample.image)
        clean_pred = _predict_one(model, clean_image)
        clean_correct = clean_pred == int(sample.true_label)
        adv_pred: Optional[int] = None
        entropy_adv: Optional[float] = None
        fgsm_linf_255: Optional[float] = None
        fgsm_changed_pixels: Optional[int] = None
        attack_success = False
        is_disturbed_failure = False

        if not clean_correct:
            skipped_wrong_baseline += 1
        else:
            n_clean_correct += 1
            adversarial_image = generate_fgsm_from_gradient(
                model=model,
                image=clean_image,
                class_id=clean_pred,
                epsilon_255=epsilon_255,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            fgsm_linf_255 = _fgsm_linf_255(clean_image, adversarial_image)
            fgsm_changed_pixels = _fgsm_changed_pixels(clean_image, adversarial_image)
            adv_pred = _predict_one(model, adversarial_image)
            entropy_adv = _entropy_for_image(adversarial_image)

            if adv_pred == clean_pred:
                disturbed_failure += 1
                is_disturbed_failure = True
            else:
                n_attack_success += 1
                attack_success = True
                valid_attacks.append(
                    _ValidAttack(
                        clean_image=clean_image,
                        adversarial_image=adversarial_image,
                        clean_pred=clean_pred,
                        adv_pred=adv_pred,
                    )
                )

        diagnostics.append(
            {
                "image_id": sample.image_id,
                "class_name": sample.class_name,
                "true_label": int(sample.true_label),
                "clean_pred": int(clean_pred),
                "adv_pred": "" if adv_pred is None else int(adv_pred),
                "clean_correct": bool(clean_correct),
                "was_skipped": bool(not clean_correct),
                "skip_reason": "none" if clean_correct else "wrong_clean_prediction",
                "attack_success": bool(attack_success),
                "disturbed_failure": bool(is_disturbed_failure),
                "entropy_clean": float(_entropy_for_image(clean_image)),
                "entropy_adv": "" if entropy_adv is None else float(entropy_adv),
                "fgsm_linf_255": "" if fgsm_linf_255 is None else float(fgsm_linf_255),
                "fgsm_changed_pixels": "" if fgsm_changed_pixels is None else int(fgsm_changed_pixels),
            }
        )

        if (
            sample_index == 1
            or sample_index == n_clean_total
            or sample_index % progress_every == 0
        ):
            logger.info(
                "Table 4 ImageNet FGSM progress %d/%d | clean_correct=%d attack_success=%d disturbed_failure=%d skipped=%d",
                sample_index,
                n_clean_total,
                n_clean_correct,
                n_attack_success,
                disturbed_failure,
                skipped_wrong_baseline,
            )

    rows: List[dict[str, Any]] = []
    logger.info(
        "Table 4 ImageNet evaluating intervals: %s",
        ", ".join(str(value) for value in interval_values),
    )
    for interval_count in interval_values:
        interval_size(interval_count)
        tp, fn, fp = _evaluate_interval(
            model=model,
            attacks=valid_attacks,
            intervals=interval_count,
        )
        recall, precision, f1 = _metrics(tp=tp, fn=fn, fp=fp)
        logger.info(
            "Table 4 ImageNet interval %d -> tp=%d fn=%d fp=%d recall=%.4f precision=%.4f f1=%.4f",
            interval_count,
            tp,
            fn,
            fp,
            recall,
            precision,
            f1,
        )
        rows.append(
            {
                "intervals": interval_count,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "n_clean_total": int(n_clean_total),
                "n_clean_correct": int(n_clean_correct),
                "n_attack_success": int(n_attack_success),
                "disturbed_failure": int(disturbed_failure),
                "skipped_wrong_baseline": int(skipped_wrong_baseline),
            }
        )

    return Table4Evaluation(
        rows=rows,
        diagnostics=diagnostics,
        n_clean_total=int(n_clean_total),
        n_clean_correct=int(n_clean_correct),
        n_attack_success=int(n_attack_success),
        disturbed_failure=int(disturbed_failure),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
    )


def validate_attack_success(result: Table4Evaluation) -> None:
    """Fail explicitly when FGSM did not create any successful adversarial example."""
    if result.n_attack_success == 0:
        raise RuntimeError(ZERO_ATTACK_SUCCESS_MESSAGE)


def _table4_wide_rows(result: Table4Evaluation) -> List[List[Any]]:
    """Return the article-style Table 4 rows for CSV output."""
    rows_by_interval = {int(row["intervals"]): row for row in result.rows}
    wide_rows: List[List[Any]] = []
    for metric_key, metric_label in TABLE4_METRICS:
        row: List[Any] = [TABLE4_DATASET_LABEL, metric_label]
        for interval in TABLE4_INTERVALS:
            row.append(rows_by_interval.get(int(interval), {}).get(metric_key, ""))
        wide_rows.append(row)
    return wide_rows


def write_table4_outputs(
    output_dir: Path,
    result: Table4Evaluation,
    csv_name: str = "table_4_imagenet.csv",
    diagnostics_name: str = "table_4_imagenet_diagnostics.csv",
) -> Tuple[Path, Path]:
    """Write Table 4 result and diagnostic CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    diagnostics_path = output_dir / diagnostics_name

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE4_OUTPUT_HEADER)
        for row in _table4_wide_rows(result):
            writer.writerow(row)

    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in result.diagnostics:
            writer.writerow({field: row.get(field, "") for field in DIAGNOSTIC_FIELDS})

    return csv_path, diagnostics_path
