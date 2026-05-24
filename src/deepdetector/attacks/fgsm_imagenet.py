"""Caffe FGSM generation for ImageNet reproduction workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageNetFGSMResult:
    """FGSM outputs and counters for ImageNet/Caffe attacks."""

    clean_images: np.ndarray
    adversarial_images: np.ndarray
    labels: Optional[np.ndarray]
    clean_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    selected_indices: np.ndarray
    diagnostics: List[dict[str, Any]]
    n_total: int
    n_clean_correct: int
    n_attack_success: int
    disturbed_failure: int
    skipped_wrong_baseline: int


def _as_image_batch(images: np.ndarray) -> np.ndarray:
    """Return a float32 image batch from one image or a batch."""
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim == 3:
        image_array = image_array.reshape((1,) + image_array.shape)
    if image_array.ndim != 4:
        raise ValueError("images must have shape (N,H,W,C), (N,C,H,W), or one 3D image.")
    return image_array


def _label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to a Python int."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def is_caffe_input(image: np.ndarray) -> bool:
    """Return whether image data looks like preprocessed Caffe input."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        return bool(image_array.shape[1] == 3 and (image_array.shape[-1] != 3 or uses_caffe_scale(image_array)))
    return bool(
        image_array.ndim == 3
        and image_array.shape[0] == 3
        and (image_array.shape[-1] != 3 or uses_caffe_scale(image_array))
    )


def uses_caffe_scale(image: np.ndarray) -> bool:
    """Return whether image values appear to be in Caffe 0-255 scale."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0:
        return False
    return bool(float(np.nanmax(image_array)) > 1.0 or float(np.nanmin(image_array)) < 0.0)


def preprocess_caffe_inputs(model: Any, images: np.ndarray) -> np.ndarray:
    """Return images in the article's Caffe input space: NCHW/BGR/0-255."""
    image_batch = _as_image_batch(images)
    if is_caffe_input(image_batch):
        return np.asarray(image_batch, dtype=np.float32)
    if not hasattr(model, "preprocess"):
        return np.asarray(image_batch, dtype=np.float32)
    return np.asarray(model.preprocess(image_batch), dtype=np.float32)


def predict_caffe_label(model: Any, image: np.ndarray) -> int:
    """Predict one ImageNet label from a Caffe input tensor or normalized image."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must have shape (H,W,C) or (C,H,W).")

    batch = image_array.reshape((1,) + image_array.shape)
    if is_caffe_input(batch) and hasattr(model, "predict_preprocessed_label"):
        label = model.predict_preprocessed_label(batch)
    else:
        label = model.predict_label(batch)
    return int(np.asarray(label).reshape(-1)[0])


def generate_fgsm_caffe_image(
    model: Any,
    image: np.ndarray,
    class_id: int,
    epsilon_255: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> np.ndarray:
    """Generate one FGSM image using the article's Caffe gradient rule."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError("image must have shape (C,H,W) or (H,W,C).")
    if not hasattr(model, "gradient"):
        raise ValueError("ImageNet FGSM reproduction requires a model.gradient method.")

    gradient = np.asarray(model.gradient(image_array, int(class_id)), dtype=np.float32)
    if gradient.shape != image_array.shape:
        raise ValueError("Gradient shape does not match image shape.")

    if uses_caffe_scale(image_array):
        epsilon = float(epsilon_255)
        effective_clip_min = float(clip_min)
        effective_clip_max = float(clip_max)
    else:
        epsilon = float(epsilon_255) / 255.0
        effective_clip_min = 0.0 if clip_min == 0.0 else float(clip_min)
        effective_clip_max = 1.0 if clip_max >= 255.0 else float(clip_max)

    adversarial = image_array + epsilon * np.sign(gradient)
    return np.clip(adversarial, effective_clip_min, effective_clip_max).astype(np.float32)


def fgsm_linf_255(clean_image: np.ndarray, adversarial_image: np.ndarray) -> float:
    """Return Linf perturbation size in 0-255 units."""
    clean = np.asarray(clean_image, dtype=np.float32)
    adversarial = np.asarray(adversarial_image, dtype=np.float32)
    delta = np.abs(adversarial - clean)
    if not uses_caffe_scale(clean):
        delta = delta * 255.0
    return float(np.max(delta)) if delta.size else 0.0


def fgsm_changed_pixels(clean_image: np.ndarray, adversarial_image: np.ndarray) -> int:
    """Return the number of elements changed by FGSM."""
    delta = np.abs(np.asarray(adversarial_image, dtype=np.float32) - np.asarray(clean_image, dtype=np.float32))
    return int(np.count_nonzero(delta > 1e-6))


def generate_fgsm_imagenet(
    model: Any,
    images: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    epsilon_255: float = 1.0,
    skip_wrong_baseline: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> ImageNetFGSMResult:
    """Generate ImageNet FGSM examples with the source article's Caffe rule.

    The main ImageNet reproduction path intentionally does not use TensorFlow
    or CleverHans. Images are converted to Caffe input space before prediction,
    gradient computation, and perturbation.
    """
    clean_images = preprocess_caffe_inputs(model, images)
    labels_array = None if labels is None else np.asarray(labels)
    if labels_array is not None and len(labels_array) != len(clean_images):
        raise ValueError("labels must have the same length as images.")

    selected_indices = []
    adversarial_images = []
    clean_predictions = []
    adversarial_predictions = []
    diagnostics: List[dict[str, Any]] = []
    n_attack_success = 0
    disturbed_failure = 0
    skipped_wrong_baseline = 0

    for index, clean_image in enumerate(clean_images):
        clean_pred = predict_caffe_label(model, clean_image)
        true_label = None if labels_array is None else _label_to_int(labels_array[index])
        clean_correct = true_label is None or clean_pred == true_label

        if skip_wrong_baseline and not clean_correct:
            skipped_wrong_baseline += 1
            diagnostics.append(
                {
                    "sample_index": int(index),
                    "true_label": "" if true_label is None else int(true_label),
                    "clean_pred": int(clean_pred),
                    "adv_pred": "",
                    "clean_correct": False,
                    "was_skipped": True,
                    "skip_reason": "wrong_clean_prediction",
                    "attack_success": False,
                    "disturbed_failure": False,
                    "fgsm_linf_255": "",
                    "fgsm_changed_pixels": "",
                }
            )
            continue

        adversarial_image = generate_fgsm_caffe_image(
            model=model,
            image=clean_image,
            class_id=clean_pred,
            epsilon_255=epsilon_255,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        adv_pred = predict_caffe_label(model, adversarial_image)
        is_disturbed_failure = bool(adv_pred == clean_pred)
        if is_disturbed_failure:
            disturbed_failure += 1
        else:
            n_attack_success += 1

        selected_indices.append(index)
        adversarial_images.append(adversarial_image)
        clean_predictions.append(clean_pred)
        adversarial_predictions.append(adv_pred)
        diagnostics.append(
            {
                "sample_index": int(index),
                "true_label": "" if true_label is None else int(true_label),
                "clean_pred": int(clean_pred),
                "adv_pred": int(adv_pred),
                "clean_correct": bool(clean_correct),
                "was_skipped": False,
                "skip_reason": "none",
                "attack_success": bool(not is_disturbed_failure),
                "disturbed_failure": bool(is_disturbed_failure),
                "fgsm_linf_255": fgsm_linf_255(clean_image, adversarial_image),
                "fgsm_changed_pixels": fgsm_changed_pixels(clean_image, adversarial_image),
            }
        )

    selected_indices_array = np.asarray(selected_indices, dtype=np.int64)
    if adversarial_images:
        adversarial_array = np.asarray(adversarial_images, dtype=np.float32)
        selected_clean = clean_images[selected_indices_array]
    else:
        adversarial_array = np.empty((0,) + tuple(clean_images.shape[1:]), dtype=np.float32)
        selected_clean = np.empty_like(adversarial_array, dtype=np.float32)

    selected_labels = None if labels_array is None else labels_array[selected_indices_array]

    return ImageNetFGSMResult(
        clean_images=selected_clean.astype(np.float32),
        adversarial_images=adversarial_array.astype(np.float32),
        labels=selected_labels,
        clean_predictions=np.asarray(clean_predictions, dtype=np.int32),
        adversarial_predictions=np.asarray(adversarial_predictions, dtype=np.int32),
        selected_indices=selected_indices_array,
        diagnostics=diagnostics,
        n_total=int(len(clean_images)),
        n_clean_correct=int(len(selected_indices_array)),
        n_attack_success=int(n_attack_success),
        disturbed_failure=int(disturbed_failure),
        skipped_wrong_baseline=int(skipped_wrong_baseline),
    )
