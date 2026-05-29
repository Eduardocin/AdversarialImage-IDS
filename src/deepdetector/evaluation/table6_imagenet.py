"""Evaluation helpers for ImageNet Table 6 adaptive quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import (
    predict_caffe_label,
    preprocess_caffe_inputs,
    uses_caffe_scale,
)
from deepdetector.evaluation.table4_imagenet import (
    ZERO_ATTACK_SUCCESS_MESSAGE,
    Table4Sample,
    generate_fgsm_from_gradient,
)
from deepdetector.filters.entropy import image_entropy_255_chw, one_d_entropy
from deepdetector.filters.quantization import scalar_quantization


logger = logging.getLogger(__name__)


TABLE6_OUTPUT_FIELDS: Tuple[str, ...] = (
    "split",
    "TP",
    "FN",
    "FP",
    "recall_percent",
    "precision_percent",
    "f1_percent",
)
DEFAULT_SPLIT_ORDER: Tuple[str, ...] = ("train", "validation")
NO_CLEAN_CORRECT_MESSAGE = "No clean-correct ImageNet samples were available for Table 6."


@dataclass(frozen=True)
class Table6SplitSummary:
    """Counters and metrics for one Table 6 ImageNet split."""

    split: str
    total_images: int
    clean_correct: int
    skipped_wrong_baseline: int
    fgsm_success: int
    disturbed_failure: int
    tp: int
    fn: int
    fp: int
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class Table6Evaluation:
    """Rows, diagnostics, and summaries produced by Table 6 evaluation."""

    rows: List[dict[str, Any]]
    diagnostics: List[dict[str, Any]]
    summaries: List[Table6SplitSummary]
    adversarial_by_split: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_clean_total(self) -> int:
        """Return the total number of configured images."""
        return int(sum(summary.total_images for summary in self.summaries))

    @property
    def n_clean_correct(self) -> int:
        """Return the total number of clean-correct images."""
        return int(sum(summary.clean_correct for summary in self.summaries))

    @property
    def n_attack_success(self) -> int:
        """Return the total number of successful FGSM examples."""
        return int(sum(summary.fgsm_success for summary in self.summaries))

    @property
    def n_valid_detections(self) -> int:
        """Return the total number of valid adversarial pairs evaluated."""
        return int(sum(summary.tp + summary.fn for summary in self.summaries))


def adaptive_quantization_step(entropy: float) -> int:
    """Return the Table 6 scalar quantization step for one entropy value."""
    if float(entropy) < 4.0:
        return 128
    if float(entropy) < 5.0:
        return 64
    return 43


def _metrics(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Return recall, precision, and F1 with zero-safe division."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return float(recall), float(precision), float(f1)


def _predict_one(model: Any, image: np.ndarray) -> int:
    """Predict a single image through the Caffe article path."""
    return predict_caffe_label(model, image)


def _article_model_input(model: Any, image: np.ndarray) -> np.ndarray:
    """Return one image in Caffe input space."""
    return preprocess_caffe_inputs(model, image)[0]


def _entropy_for_image(image: np.ndarray) -> float:
    """Compute entropy for normalized images or Caffe ``CHW/0..255`` tensors."""
    image_array = np.asarray(image, dtype=np.float32)
    if uses_caffe_scale(image_array):
        if image_array.ndim == 3 and image_array.shape[0] == 3:
            return image_entropy_255_chw(image_array)
        if image_array.ndim == 3 and image_array.shape[-1] == 3:
            return image_entropy_255_chw(np.transpose(image_array, (2, 0, 1)))
    return one_d_entropy(image_array)


def _quantize_with_step(image: np.ndarray, step: int) -> np.ndarray:
    """Apply scalar quantization with ``step`` in the image's native scale."""
    image_array = np.asarray(image, dtype=np.float32)
    if uses_caffe_scale(image_array):
        quantized = np.clip(image_array, 0.0, 255.0)
        quantized //= int(step)
        quantized *= int(step)
        return quantized.astype(np.float32)
    return scalar_quantization(image_array, interval=int(step), left=True)


def adaptive_quantize_image(image: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Apply Table 6 entropy-defined adaptive quantization."""
    entropy = _entropy_for_image(image)
    step = adaptive_quantization_step(entropy)
    return _quantize_with_step(image, step), float(entropy), int(step)


def _blank_diagnostic(
    split: str,
    sample: Table4Sample,
    clean_pred: int,
    clean_correct: bool,
    skip_reason: str,
) -> dict[str, Any]:
    """Return a diagnostic row for samples skipped before detection."""
    return {
        "split": split,
        "image_id": sample.image_id,
        "class_name": sample.class_name,
        "true_label": int(sample.true_label),
        "clean_pred": int(clean_pred),
        "filtered_clean_pred": "",
        "adv_pred": "",
        "filtered_adv_pred": "",
        "clean_correct": bool(clean_correct),
        "attack_success": False,
        "entropy_clean": "",
        "entropy_adv": "",
        "clean_quantization_step": "",
        "adv_quantization_step": "",
        "is_fp": False,
        "is_tp": False,
        "is_fn": False,
        "skip_reason": skip_reason,
    }


def _evaluate_split(
    model: Any,
    split: str,
    samples: Sequence[Table4Sample],
    epsilon_255: float,
    clip_min: float,
    clip_max: float,
    cached_adversarial_images: Optional[np.ndarray] = None,
) -> Tuple[Table6SplitSummary, List[dict[str, Any]], np.ndarray]:
    """Evaluate adaptive quantization for one split."""
    total_images = len(samples)
    clean_correct_count = 0
    skipped_wrong_baseline = 0
    fgsm_success = 0
    disturbed_failure = 0
    tp = 0
    fn = 0
    fp = 0
    diagnostics: List[dict[str, Any]] = []
    cached_adv_index = 0
    using_cache = cached_adversarial_images is not None
    adversarial_for_cache: List[np.ndarray] = []

    for sample_index, sample in enumerate(samples, start=1):
        clean_image = _article_model_input(model, sample.image)
        clean_pred = _predict_one(model, clean_image)
        clean_correct = clean_pred == int(sample.true_label)

        if not clean_correct:
            skipped_wrong_baseline += 1
            diagnostics.append(
                _blank_diagnostic(
                    split=split,
                    sample=sample,
                    clean_pred=clean_pred,
                    clean_correct=False,
                    skip_reason="wrong_clean_prediction",
                )
            )
            continue

        clean_correct_count += 1
        filtered_clean, entropy_clean, clean_step = adaptive_quantize_image(clean_image)
        filtered_clean_pred = _predict_one(model, filtered_clean)

        if using_cache and cached_adv_index < len(cached_adversarial_images):
            adversarial_image = np.asarray(
                cached_adversarial_images[cached_adv_index],
                dtype=np.float32,
            )
            cached_adv_index += 1
            if adversarial_image.shape != clean_image.shape:
                raise ValueError(
                    "Cached Table 6 ImageNet adversarial shape {0} does not match clean image shape {1} for split {2}.".format(
                        adversarial_image.shape,
                        clean_image.shape,
                        split,
                    )
                )
        elif using_cache:
            raise ValueError(
                "Cached Table 6 ImageNet adversarials for split {0} ended after {1} clean-correct samples. "
                "The cache is incompatible with the current dataset/model configuration.".format(
                    split,
                    cached_adv_index,
                )
            )
        else:
            adversarial_image = generate_fgsm_from_gradient(
                model=model,
                image=clean_image,
                class_id=clean_pred,
                epsilon_255=epsilon_255,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        adversarial_for_cache.append(np.asarray(adversarial_image, dtype=np.float32))
        adv_pred = _predict_one(model, adversarial_image)

        if adv_pred == clean_pred:
            disturbed_failure += 1
            diagnostics.append(
                {
                    "split": split,
                    "image_id": sample.image_id,
                    "class_name": sample.class_name,
                    "true_label": int(sample.true_label),
                    "clean_pred": int(clean_pred),
                    "filtered_clean_pred": int(filtered_clean_pred),
                    "adv_pred": int(adv_pred),
                    "filtered_adv_pred": "",
                    "clean_correct": True,
                    "attack_success": False,
                    "entropy_clean": float(entropy_clean),
                    "entropy_adv": "",
                    "clean_quantization_step": int(clean_step),
                    "adv_quantization_step": "",
                    "is_fp": False,
                    "is_tp": False,
                    "is_fn": False,
                    "skip_reason": "fgsm_failed_to_change_prediction",
                }
            )
            continue

        fgsm_success += 1
        filtered_adv, entropy_adv, adv_step = adaptive_quantize_image(adversarial_image)
        filtered_adv_pred = _predict_one(model, filtered_adv)

        is_fp = bool(filtered_clean_pred != clean_pred)
        is_tp = bool(filtered_adv_pred != adv_pred)
        is_fn = bool(not is_tp)
        fp += int(is_fp)
        tp += int(is_tp)
        fn += int(is_fn)

        diagnostics.append(
            {
                "split": split,
                "image_id": sample.image_id,
                "class_name": sample.class_name,
                "true_label": int(sample.true_label),
                "clean_pred": int(clean_pred),
                "filtered_clean_pred": int(filtered_clean_pred),
                "adv_pred": int(adv_pred),
                "filtered_adv_pred": int(filtered_adv_pred),
                "clean_correct": True,
                "attack_success": True,
                "entropy_clean": float(entropy_clean),
                "entropy_adv": float(entropy_adv),
                "clean_quantization_step": int(clean_step),
                "adv_quantization_step": int(adv_step),
                "is_fp": is_fp,
                "is_tp": is_tp,
                "is_fn": is_fn,
                "skip_reason": "none",
            }
        )

        if sample_index == 1 or sample_index == total_images or sample_index % 20 == 0:
            logger.info(
                "Table 6 ImageNet %s progress %d/%d | adversarial_source=%s clean_correct=%d attack_success=%d disturbed_failure=%d skipped=%d",
                split,
                sample_index,
                total_images,
                "cache" if using_cache else "generated",
                clean_correct_count,
                fgsm_success,
                disturbed_failure,
                skipped_wrong_baseline,
            )

    recall, precision, f1 = _metrics(tp=tp, fn=fn, fp=fp)
    summary = Table6SplitSummary(
        split=split,
        total_images=int(total_images),
        clean_correct=int(clean_correct_count),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
        fgsm_success=int(fgsm_success),
        disturbed_failure=int(disturbed_failure),
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        recall=recall,
        precision=precision,
        f1=f1,
    )
    if using_cache and cached_adv_index != len(cached_adversarial_images):
        logger.warning(
            "Table 6 ImageNet cache for split %s had %d unused adversarial examples.",
            split,
            len(cached_adversarial_images) - cached_adv_index,
        )
    adversarial_array = np.asarray(adversarial_for_cache, dtype=np.float32)
    return summary, diagnostics, adversarial_array


def evaluate_table6_imagenet(
    model: Any,
    samples_by_split: Mapping[str, Sequence[Table4Sample]],
    epsilon_255: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
    split_order: Sequence[str] = DEFAULT_SPLIT_ORDER,
    adversarial_by_split: Optional[Mapping[str, np.ndarray]] = None,
) -> Table6Evaluation:
    """Evaluate ImageNet Table 6 adaptive quantization for configured splits."""
    rows: List[dict[str, Any]] = []
    diagnostics: List[dict[str, Any]] = []
    summaries: List[Table6SplitSummary] = []
    output_adversarial_by_split: Dict[str, np.ndarray] = {}
    cached_by_split = adversarial_by_split or {}

    for split in split_order:
        split_samples = list(samples_by_split.get(split, []))
        if not split_samples:
            raise ValueError("Configured split has no images: {0}".format(split))

        summary, split_diagnostics, split_adversarial = _evaluate_split(
            model=model,
            split=str(split),
            samples=split_samples,
            epsilon_255=epsilon_255,
            clip_min=clip_min,
            clip_max=clip_max,
            cached_adversarial_images=cached_by_split.get(str(split)),
        )
        summaries.append(summary)
        diagnostics.extend(split_diagnostics)
        output_adversarial_by_split[str(split)] = split_adversarial
        rows.append(
            {
                "split": str(split),
                "TP": summary.tp,
                "FN": summary.fn,
                "FP": summary.fp,
                "recall_percent": float(summary.recall * 100.0),
                "precision_percent": float(summary.precision * 100.0),
                "f1_percent": float(summary.f1 * 100.0),
            }
        )

    return Table6Evaluation(
        rows=rows,
        diagnostics=diagnostics,
        summaries=summaries,
        adversarial_by_split=output_adversarial_by_split,
    )


def validate_table6_result(result: Table6Evaluation) -> None:
    """Fail explicitly when the Table 6 run did not evaluate valid attacks."""
    if result.n_clean_correct == 0:
        raise RuntimeError(NO_CLEAN_CORRECT_MESSAGE)
    if result.n_attack_success == 0 or result.n_valid_detections == 0:
        raise RuntimeError(ZERO_ATTACK_SUCCESS_MESSAGE)
