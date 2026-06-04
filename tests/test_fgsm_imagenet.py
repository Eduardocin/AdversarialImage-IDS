from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import (  # noqa: E402
    generate_fgsm_caffe_image,
    generate_fgsm_imagenet,
    preprocess_caffe_inputs,
)


class CaffeStyleModel:
    """Small model exposing the Caffe-style API used by ImageNet FGSM."""

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        batch = np.asarray(images, dtype=np.float32)
        return np.transpose(batch[:, :, :, ::-1] * 255.0, (0, 3, 1, 2))

    def predict_preprocessed_label(self, images: np.ndarray) -> np.ndarray:
        labels = []
        for image in images:
            labels.append(1 if float(np.mean(image)) < 200.0 else 2)
        return np.asarray(labels, dtype=np.int32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.ones_like(image, dtype=np.float32)


class DualGradientCaffeStyleModel(CaffeStyleModel):
    """Track whether FGSM uses prediction or attack gradients."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def prediction_gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        self.calls.append("prediction")
        return np.ones_like(image, dtype=np.float32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        self.calls.append("attack")
        return -np.ones_like(image, dtype=np.float32)


def test_generate_fgsm_caffe_image_uses_raw_epsilon_255() -> None:
    """The article path applies epsilon as a raw 0-255 pixel step."""
    image = np.full((3, 2, 2), 128.0, dtype=np.float32)

    adv = generate_fgsm_caffe_image(
        model=CaffeStyleModel(),
        image=image,
        class_id=1,
        epsilon_255=1.0,
    )

    np.testing.assert_array_equal(adv, np.full((3, 2, 2), 129.0, dtype=np.float32))


def test_generate_fgsm_caffe_image_prefers_prediction_gradient() -> None:
    """FGSM should follow the original softmax/prob gradient path."""
    model = DualGradientCaffeStyleModel()
    image = np.full((3, 2, 2), 128.0, dtype=np.float32)

    adv = generate_fgsm_caffe_image(
        model=model,
        image=image,
        class_id=1,
        epsilon_255=1.0,
    )

    assert model.calls == ["prediction"]
    np.testing.assert_array_equal(adv, np.full((3, 2, 2), 129.0, dtype=np.float32))


def test_generate_fgsm_imagenet_filters_wrong_clean_baseline() -> None:
    """Images with clean_pred != true_label should not be attacked."""
    images = np.asarray(
        [
            np.full((2, 2, 3), 0.5, dtype=np.float32),
            np.full((2, 2, 3), 0.9, dtype=np.float32),
        ]
    )
    labels = np.asarray([1, 1], dtype=np.int32)

    result = generate_fgsm_imagenet(
        model=CaffeStyleModel(),
        images=images,
        labels=labels,
        epsilon_255=1.0,
    )

    assert result.n_total == 2
    assert result.n_clean_correct == 1
    assert result.skipped_wrong_baseline == 1
    assert result.selected_indices.tolist() == [0]
    assert result.clean_images.shape == (1, 3, 2, 2)
    assert result.adversarial_images.shape == (1, 3, 2, 2)
    assert result.diagnostics[1]["skip_reason"] == "wrong_clean_prediction"


def test_generate_fgsm_imagenet_logs_progress(caplog) -> None:
    """ImageNet FGSM generation should expose progress for long runs."""
    images = np.asarray(
        [
            np.full((2, 2, 3), 0.5, dtype=np.float32),
            np.full((2, 2, 3), 0.6, dtype=np.float32),
        ]
    )

    with caplog.at_level(logging.INFO, logger="deepdetector.attacks.fgsm_imagenet"):
        generate_fgsm_imagenet(
            model=CaffeStyleModel(),
            images=images,
            labels=None,
            epsilon_255=1.0,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("ImageNet FGSM generation started" in message for message in messages)
    assert any("ImageNet FGSM progress 1/2" in message for message in messages)
    assert any("ImageNet FGSM generation completed" in message for message in messages)


def test_main_fgsm_path_does_not_import_tensorflow(monkeypatch) -> None:
    """The Caffe reproduction path should not require TensorFlow imports."""
    imported = []

    original_import = __import__

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    preprocess_caffe_inputs(CaffeStyleModel(), np.zeros((1, 2, 2, 3), dtype=np.float32))

    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imported)
