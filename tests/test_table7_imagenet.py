from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts.article_reproduction.table_7_imagenet import (
    filter_clean_baseline_images,
    generate_adversarial_images,
    load_adversarial_images,
)


class SequenceModel:
    """Return one configured prediction per call."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.index = 0

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        label = self.labels[self.index]
        self.index += 1
        return np.full((len(images),), label, dtype=np.int32)


class GradientModel:
    """Small model exposing the Caffe-style methods used by the article path."""

    def predict_preprocessed_label(self, images: np.ndarray) -> np.ndarray:
        return np.ones((len(images),), dtype=np.int32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.ones_like(image, dtype=np.float32)


def test_table7_filters_wrong_clean_predictions_before_attack_inputs() -> None:
    """Only clean-correct images should remain for Table 7 attack evaluation."""
    images = np.asarray(
        [
            np.full((2, 2, 3), 0.1, dtype=np.float32),
            np.full((2, 2, 3), 0.2, dtype=np.float32),
            np.full((2, 2, 3), 0.3, dtype=np.float32),
        ]
    )
    labels = np.asarray([1, 1, 953], dtype=np.int32)

    filtered_images, filtered_labels, selected_indices, summary = filter_clean_baseline_images(
        model=SequenceModel([1, 9, 953]),
        images=images,
        labels=labels,
    )

    assert selected_indices.tolist() == [0, 2]
    assert filtered_images.shape == (2, 2, 2, 3)
    assert filtered_labels.tolist() == [1, 953]
    assert summary == {
        "total_images": 3,
        "clean_correct": 2,
        "skipped_wrong_baseline": 1,
    }


def test_table7_generated_fgsm_uses_article_caffe_scale() -> None:
    """Preprocessed Caffe tensors should receive a raw 0-255 FGSM step."""
    images = np.full((1, 3, 2, 2), 128.0, dtype=np.float32)

    adv = generate_adversarial_images(
        config={"attack": {"epsilon_255": 1.0}},
        model=GradientModel(),
        images=images,
    )

    np.testing.assert_array_equal(adv, np.full((1, 3, 2, 2), 129.0, dtype=np.float32))


def test_table7_loads_full_adversarial_array_with_selected_clean_indices(tmp_path) -> None:
    """A pre-generated full adversarial array should be aligned after clean filtering."""
    full_adv = np.arange(3 * 2 * 2 * 3, dtype=np.float32).reshape((3, 2, 2, 3))
    adv_path = tmp_path / "adv.npy"
    np.save(str(adv_path), full_adv)

    loaded = load_adversarial_images(
        adv_path,
        expected_shape=(2, 2, 2, 3),
        selected_indices=np.asarray([0, 2], dtype=np.int64),
    )

    assert loaded.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(loaded, full_adv[[0, 2]])
