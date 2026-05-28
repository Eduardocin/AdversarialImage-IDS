from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts.article_reproduction.table_4_imagenet import load_subset_samples
from deepdetector.evaluation.table4_imagenet import (
    TABLE4_OUTPUT_HEADER,
    ZERO_ATTACK_SUCCESS_MESSAGE,
    Table4Sample,
    _quantize_for_intervals,
    evaluate_table4_imagenet,
    generate_fgsm_from_gradient,
    validate_attack_success,
    write_table4_outputs,
)


class ThresholdModel:
    """Small deterministic model for Table 4 counting tests."""

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        labels = []
        for image in images:
            mean_value = float(np.mean(image))
            if mean_value < 0.1:
                labels.append(0)
            elif mean_value < 0.6:
                labels.append(1)
            else:
                labels.append(2)
        return np.asarray(labels, dtype=np.int32)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.ones_like(image, dtype=np.float32)


class NoAttackModel(ThresholdModel):
    """Model with a zero gradient so FGSM cannot change the prediction."""

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.zeros_like(image, dtype=np.float32)


def _sample(value: float, true_label: int = 1, image_id: str = "img") -> Table4Sample:
    image = np.full((2, 2, 3), value, dtype=np.float32)
    return Table4Sample(
        image=image,
        true_label=true_label,
        image_id=image_id,
        class_name="goldfish",
    )


def test_evaluate_table4_imagenet_writes_nine_interval_rows() -> None:
    """Table 4 should evaluate every scalar interval from 2 to 10."""
    result = evaluate_table4_imagenet(
        model=ThresholdModel(),
        samples=[_sample(0.49)],
        epsilon_255=60.0,
    )

    assert [row["intervals"] for row in result.rows] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert any(row["intervals"] == 6 for row in result.rows)
    assert result.n_clean_total == 1
    assert result.n_clean_correct == 1
    assert result.n_attack_success == 1

    first_row = result.rows[0]
    assert first_row["tp"] == 1
    assert first_row["fn"] == 0
    assert first_row["fp"] == 1
    assert first_row["recall"] == pytest.approx(1.0)
    assert first_row["precision"] == pytest.approx(0.5)
    assert first_row["f1"] == pytest.approx(2.0 / 3.0)


def test_table4_fgsm_uses_article_caffe_scale_for_preprocessed_images() -> None:
    """In Caffe 0-255 space, epsilon_255 should be applied as a raw pixel step."""
    image = np.full((3, 2, 2), 128.0, dtype=np.float32)
    adv = generate_fgsm_from_gradient(
        model=ThresholdModel(),
        image=image,
        class_id=1,
        epsilon_255=1.0,
        clip_min=0.0,
        clip_max=1.0,
    )

    np.testing.assert_array_equal(adv, np.full((3, 2, 2), 129.0, dtype=np.float32))


def test_evaluate_table4_imagenet_counts_wrong_baseline_skips() -> None:
    """Wrong clean predictions should be skipped before FGSM."""
    result = evaluate_table4_imagenet(
        model=ThresholdModel(),
        samples=[
            _sample(0.49, true_label=1, image_id="correct"),
            _sample(0.49, true_label=5, image_id="wrong"),
        ],
        epsilon_255=60.0,
    )

    assert result.n_clean_total == 2
    assert result.n_clean_correct == 1
    assert result.n_attack_success == 1
    assert result.skipped_wrong_baseline == 1
    assert result.disturbed_failure == 0


def test_table4_quantization_preserves_caffe_scale_for_preprocessed_images() -> None:
    """Scalar quantization should operate in 0-255 space for Caffe tensors."""
    image = np.asarray([[[129.0, 255.0], [64.0, 0.0]]] * 3, dtype=np.float32)

    quantized = _quantize_for_intervals(image, intervals=2)

    np.testing.assert_array_equal(
        quantized,
        np.asarray([[[128.0, 128.0], [0.0, 0.0]]] * 3, dtype=np.float32),
    )


def test_table4_validation_fails_when_fgsm_has_zero_successes() -> None:
    """The required sanity failure should be explicit when no attack succeeds."""
    result = evaluate_table4_imagenet(
        model=NoAttackModel(),
        samples=[_sample(0.49)],
        epsilon_255=60.0,
    )

    assert result.n_attack_success == 0
    assert result.disturbed_failure == 1
    with pytest.raises(RuntimeError) as excinfo:
        validate_attack_success(result)

    assert str(excinfo.value) == ZERO_ATTACK_SUCCESS_MESSAGE


def test_write_table4_outputs_uses_specified_csv_columns(tmp_path) -> None:
    """The output CSV file should keep the spec field order."""
    result = evaluate_table4_imagenet(
        model=ThresholdModel(),
        samples=[_sample(0.49)],
        epsilon_255=60.0,
    )

    csv_path = write_table4_outputs(tmp_path, result)

    assert csv_path.name == "table_4_imagenet.csv"
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == ",".join(TABLE4_OUTPUT_HEADER)
    assert len(csv_lines) == 4
    assert csv_lines[1].split(",")[1] == "Recall"
    assert csv_lines[2].split(",")[1] == "Precision"
    assert csv_lines[3].split(",")[1] == "F1 Score"
    assert not (tmp_path / "table_4_imagenet_diagnostics.csv").exists()


def test_table4_script_loads_local_png_class_folders(tmp_path) -> None:
    """The CLI loader should read PNG images from class directories."""
    Image = pytest.importorskip("PIL.Image")

    class_dir = tmp_path / "goldfish"
    class_dir.mkdir()
    image = np.full((3, 3, 3), 128, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(str(class_dir / "sample.png"))

    config = {
        "dataset": {
            "images_dir": str(tmp_path),
            "image_size": 4,
            "class_indices": {"goldfish": 1},
        }
    }

    samples = load_subset_samples(config)

    assert len(samples) == 1
    assert samples[0].image.shape == (4, 4, 3)
    assert samples[0].image.dtype == np.float32
    assert samples[0].true_label == 1
    assert samples[0].class_name == "goldfish"
    assert samples[0].image_id == "sample"
