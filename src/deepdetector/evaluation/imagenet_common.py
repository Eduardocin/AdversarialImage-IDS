"""Shared ImageNet evaluation helpers for Table 7 and Table 8."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import predict_caffe_label
from deepdetector.evaluation.metrics import safe_precision_recall_f1
from deepdetector.filters.table7_filters import table7_filter


def label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to a Python int."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def predict_one_caffe(model: Any, image: np.ndarray) -> int:
    """Predict one image with a Caffe-style ImageNet model wrapper."""
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 3:
        batch = image_array.reshape((1,) + image_array.shape)
    elif image_array.ndim == 4 and image_array.shape[0] == 1:
        batch = image_array
    else:
        raise ValueError("image must have shape (H, W, C), (C, H, W), or (1, ...).")

    return predict_caffe_label(model, batch[0])


def iter_detection_dataset(
    dataset: Any,
) -> Iterator[Tuple[np.ndarray, Any, Optional[np.ndarray]]]:
    """Yield `(image, label, adversarial_image)` rows from common dataset forms."""
    if isinstance(dataset, tuple):
        if len(dataset) not in (2, 3):
            raise ValueError("dataset tuple must be (images, labels) or (images, labels, adv_images).")
        images = np.asarray(dataset[0], dtype=np.float32)
        labels = np.asarray(dataset[1])
        adv_images = None if len(dataset) == 2 else np.asarray(dataset[2], dtype=np.float32)

        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        n_rows = len(images) if adv_images is None else min(len(images), len(adv_images))

        for index in range(n_rows):
            image = images[index]
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


def image_to_chw_255(image: np.ndarray) -> Tuple[np.ndarray, str, float]:
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


def restore_from_chw_255(chw_255: np.ndarray, layout: str, scale: float) -> np.ndarray:
    """Restore filtered CHW 0-255 data to the model input layout and range."""
    restored = np.asarray(chw_255, dtype=np.float32)
    if scale == 255.0:
        restored = restored / 255.0

    if layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    elif layout != "chw":
        raise ValueError("Unknown image layout: {0}".format(layout))

    return restored.astype(np.float32)


def apply_spatial_filter_to_model_input(
    image: np.ndarray,
    mask_type: str,
    size: int,
) -> np.ndarray:
    """Apply the shared CHW 0-255 spatial smoothing filter to model input."""
    chw_255, layout, scale = image_to_chw_255(image)
    filtered = table7_filter(
        image=chw_255,
        mask_type=mask_type,
        size=size,
    )
    return restore_from_chw_255(filtered, layout=layout, scale=scale)
