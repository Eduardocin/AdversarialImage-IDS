from __future__ import annotations

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
