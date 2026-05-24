"""Evaluation helpers for ImageNet Table 7 spatial smoothing filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import (
    generate_fgsm_caffe_image,
    predict_caffe_label,
    uses_caffe_scale,
)
from deepdetector.filters.entropy import image_entropy_255_chw
from deepdetector.filters.table7_filters import table7_filter


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
    n_high_entropy_clean: int
    n_high_entropy_adversarial: int
    disturbed_failure: int
    skipped_wrong_baseline: int


def _label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to a Python int."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def _predict_one(model: Any, image: np.ndarray) -> int:
    """Predict one image with a model exposing ``predict_label``."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 3:
        batch = image_array.reshape((1,) + image_array.shape)
    elif image_array.ndim == 4 and image_array.shape[0] == 1:
        batch = image_array
    else:
        raise ValueError("image must have shape (H, W, C), (C, H, W), or (1, ...).")

    return predict_caffe_label(model, batch[0])


def _iter_dataset(dataset: Any) -> Iterator[Tuple[np.ndarray, Any, Optional[np.ndarray]]]:
    """Yield ``(image, label, adversarial_image)`` rows from common dataset forms."""
    if isinstance(dataset, tuple):
        if len(dataset) not in (2, 3):
            raise ValueError("dataset tuple must be (images, labels) or (images, labels, adv_images).")
        images = np.asarray(dataset[0], dtype=np.float32)
        labels = np.asarray(dataset[1])
        adv_images = None if len(dataset) == 2 else np.asarray(dataset[2], dtype=np.float32)

        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        if adv_images is not None and len(adv_images) != len(images):
            raise ValueError("adversarial images must have the same length as images.")

        for index, image in enumerate(images):
            adversarial = None if adv_images is None else adv_images[index]
            yield image, labels[index], adversarial
        return

    for item in dataset:
        if isinstance(item, dict):
            yield (
                np.asarray(item["image"], dtype=np.float32),
                item["label"],
                (
                    None
                    if item.get("adversarial_image") is None
                    else np.asarray(item.get("adversarial_image"), dtype=np.float32)
                ),
            )
            continue

        if len(item) == 2:
            image, label = item
            yield np.asarray(image, dtype=np.float32), label, None
            continue
        if len(item) == 3:
            image, label, adversarial = item
            yield (
                np.asarray(image, dtype=np.float32),
                label,
                np.asarray(adversarial, dtype=np.float32),
            )
            continue

        raise ValueError("dataset items must have 2 or 3 values.")


def _image_to_chw_255(image: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """Convert HWC/CHW image data to CHW 0-255 values."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must be a single 3D image.")

    if image_array.shape[-1] in (1, 3):
        layout = "hwc"
        chw = np.transpose(image_array, (2, 0, 1))
    elif image_array.shape[0] in (1, 3):
        layout = "chw"
        chw = image_array
    else:
        raise ValueError("image must use HWC or CHW layout with 1 or 3 channels.")

    scale = 255.0 if float(np.nanmax(chw)) <= 1.0 and float(np.nanmin(chw)) >= 0.0 else 1.0
    return (chw * scale).astype(np.float32), layout, scale


def _restore_from_chw_255(chw_255: np.ndarray, layout: str, scale: float) -> np.ndarray:
    """Restore filtered CHW 0-255 data to the model input layout and range."""
    restored = np.asarray(chw_255, dtype=np.float32)
    if scale == 255.0:
        restored = restored / 255.0

    if layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    elif layout != "chw":
        raise ValueError("Unknown image layout: {0}".format(layout))

    return restored.astype(np.float32)


def _entropy_for_image(image: np.ndarray) -> float:
    """Compute Table 7 entropy for a model input image."""
    chw_255, _, _ = _image_to_chw_255(image)
    return image_entropy_255_chw(chw_255)


def _apply_table7_filter_to_model_input(
    image: np.ndarray,
    mask_type: str,
    size: int,
    quantization_step: int,
) -> np.ndarray:
    """Apply the CHW 0-255 Table 7 filter and restore the model input format."""
    chw_255, layout, scale = _image_to_chw_255(image)
    filtered = table7_filter(
        image=chw_255,
        mask_type=mask_type,
        size=size,
        quantization_step=quantization_step,
    )
    return _restore_from_chw_255(filtered, layout=layout, scale=scale)


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


def _metrics(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Return recall, precision and F1 with zero-safe division."""
    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * recall * precision / float(recall + precision) if recall + precision else 0.0
    return float(recall), float(precision), float(f1)


def evaluate_table7_filter(
    model: Any,
    dataset: Any,
    mask_type: str,
    size: int,
    epsilon: float,
    entropy_threshold: float = 5.0,
    quantization_step: int = 43,
) -> Table7FilterResult:
    """Evaluate one spatial smoothing candidate for ImageNet Table 7."""
    tp = 0
    fn = 0
    fp = 0
    n_high_entropy_clean = 0
    n_high_entropy_adversarial = 0
    disturbed_failure = 0
    skipped_wrong_baseline = 0

    for clean_image, label, provided_adversarial in _iter_dataset(dataset):
        true_label = _label_to_int(label)
        clean_pred = _predict_one(model, clean_image)
        if clean_pred != true_label:
            skipped_wrong_baseline += 1
            continue

        if _entropy_for_image(clean_image) > float(entropy_threshold):
            n_high_entropy_clean += 1
            filtered_clean = _apply_table7_filter_to_model_input(
                image=clean_image,
                mask_type=mask_type,
                size=size,
                quantization_step=quantization_step,
            )
            filtered_clean_pred = _predict_one(model, filtered_clean)
            if filtered_clean_pred != clean_pred:
                fp += 1

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

        if _entropy_for_image(adversarial_image) <= float(entropy_threshold):
            continue

        n_high_entropy_adversarial += 1
        filtered_adv = _apply_table7_filter_to_model_input(
            image=adversarial_image,
            mask_type=mask_type,
            size=size,
            quantization_step=quantization_step,
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
        n_high_entropy_clean=int(n_high_entropy_clean),
        n_high_entropy_adversarial=int(n_high_entropy_adversarial),
        disturbed_failure=int(disturbed_failure),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
    )
