from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation import table7 as table7_module
from deepdetector.evaluation.table7 import Table7FilterResult
from deepdetector.experiments import table7_imagenet_runner
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


class MarkerModel:
    """Predict from a marker value stored in the first Caffe tensor element."""

    def predict_preprocessed_label(self, images: np.ndarray) -> np.ndarray:
        marker_to_label = {10: 1, 11: 1, 20: 2, 4: 4}
        return np.asarray(
            [
                marker_to_label.get(
                    int(round(float(image[0, 0, 0]))),
                    int(round(float(image[0, 0, 0]))),
                )
                for image in images
            ],
            dtype=np.int32,
        )


def _marker_image(marker: int) -> np.ndarray:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[0, 0, 0] = float(marker)
    return image


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


def test_table7_counts_fp_only_for_valid_high_entropy_adversarial_pairs(monkeypatch) -> None:
    """FP must use the same selected adversarial high-entropy population as TP/FN."""
    dataset = (
        np.asarray([_marker_image(10), _marker_image(11)], dtype=np.float32),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([_marker_image(20), _marker_image(4)], dtype=np.float32),
    )

    def fake_entropy(image: np.ndarray) -> float:
        marker = int(round(float(image[0, 0, 0])))
        return 4.0 if marker == 4 else 6.0

    def fake_filter(
        image: np.ndarray,
        mask_type: str,
        size: int,
    ) -> np.ndarray:
        del mask_type, size
        marker = int(round(float(image[0, 0, 0])))
        marker_after_filter = {10: 9, 20: 3, 11: 9}.get(marker, marker)
        filtered = np.array(image, copy=True)
        filtered[0, 0, 0] = float(marker_after_filter)
        return filtered

    monkeypatch.setattr(table7_module, "_entropy_for_image", fake_entropy)
    monkeypatch.setattr(table7_module, "_apply_table7_filter_to_model_input", fake_filter)

    result = table7_module.evaluate_table7_filter(
        model=MarkerModel(),
        dataset=dataset,
        mask_type="box",
        size=3,
        epsilon=1.0 / 255.0,
        entropy_threshold=5.0,
    )

    assert result.n_high_entropy_adversarial == 1
    assert result.tp == 1
    assert result.fn == 0
    assert result.fp == 1
    assert result.precision == 0.5


def test_run_table7_experiment_writes_pivot_and_status(monkeypatch, tmp_path) -> None:
    """The official Table 7 experiment should write the ImageNet pivot output."""
    calls = []
    (tmp_path / "metrics.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "metrics.json").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        table7_imagenet_runner,
        "build_imagenet_table7_model",
        lambda config: MarkerModel(),
    )
    monkeypatch.setattr(
        table7_imagenet_runner,
        "load_imagenet_table7_subset",
        lambda config: (
            np.asarray([_marker_image(10)], dtype=np.float32),
            np.asarray([1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        table7_imagenet_runner,
        "filter_clean_baseline_images",
        lambda model, images, labels: (
            images,
            labels,
            np.asarray([0], dtype=np.int64),
            {"total_images": 1, "clean_correct": 1, "skipped_wrong_baseline": 0},
        ),
    )
    monkeypatch.setattr(
        table7_imagenet_runner,
        "_article_model_inputs",
        lambda model, images: images,
    )
    monkeypatch.setattr(
        table7_imagenet_runner,
        "adversarial_images_for_run",
        lambda config, model, images, selected_indices=None: images + 1.0,
    )

    def fake_evaluate(**kwargs):
        calls.append((kwargs["mask_type"], kwargs["size"]))
        return Table7FilterResult(
            mask_type=kwargs["mask_type"],
            size=kwargs["size"],
            tp=1,
            fn=1,
            fp=0,
            recall=0.5,
            precision=1.0,
            f1=2.0 / 3.0,
            n_high_entropy_clean=1,
            n_high_entropy_adversarial=1,
            disturbed_failure=0,
            skipped_wrong_baseline=0,
        )

    monkeypatch.setattr(table7_imagenet_runner, "evaluate_table7_filter", fake_evaluate)

    result = table7_imagenet_runner.run_table7_imagenet_experiment(
        {
            "experiment_id": "table_7",
            "kind": "imagenet_table_7",
            "dataset": {"name": "imagenet", "classes": []},
            "model": {"name": "googlenet_caffe"},
            "attack": {"epsilon_255": 1.0},
            "filter": {"mask_types": ["cross"], "sizes": [3], "entropy_threshold": 5.0},
            "output": {
                "dir": str(tmp_path),
                "pivot_csv": "table_7_imagnet.csv",
                "status_json": "table_7_status.json",
            },
        }
    )

    assert result["status"] == "completo"
    assert calls == [("cross", 3)]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "table_7_imagnet.csv",
        "table_7_status.json",
    ]
    assert (tmp_path / "table_7_imagnet.csv").read_text(encoding="utf-8").splitlines() == [
        "metric,cross_3x3,cross_5x5,cross_7x7,cross_9x9,diamond_3x3,diamond_5x5,diamond_7x7,diamond_9x9,box_3x3,box_5x5,box_7x7,box_9x9",
        "Recall,0.500000,,,,,,,,,,,",
        "Precision,1.000000,,,,,,,,,,,",
        "F1 Score,0.666667,,,,,,,,,,,",
    ]
