from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.table4_imagenet import (  # noqa: E402
    ZERO_ATTACK_SUCCESS_MESSAGE,
    Table4Sample,
)
from deepdetector.evaluation.table6_imagenet import (  # noqa: E402
    TABLE6_DIAGNOSTIC_FIELDS,
    TABLE6_OUTPUT_FIELDS,
    adaptive_quantization_step,
    adaptive_quantize_image,
    evaluate_table6_imagenet,
    validate_table6_result,
    write_table6_outputs,
)
from scripts.article_reproduction.table_6_imagenet import load_split_samples  # noqa: E402


class MeanCaffeModel:
    """Small deterministic Caffe-style model for Table 6 tests."""

    def predict_preprocessed_label(self, images: np.ndarray) -> np.ndarray:
        labels = []
        for image in images:
            mean_value = float(np.mean(image))
            if mean_value < 80.0:
                labels.append(0)
            elif mean_value < 150.0:
                labels.append(1)
            else:
                labels.append(2)
        return np.asarray(labels, dtype=np.int32)

    def predict_label(self, images: np.ndarray) -> np.ndarray:
        return self.predict_preprocessed_label(images)

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.ones_like(image, dtype=np.float32)


class NoAttackModel(MeanCaffeModel):
    """Model with a zero gradient so FGSM cannot change predictions."""

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        return np.zeros_like(image, dtype=np.float32)


def _sample(value: float, true_label: int = 1, image_id: str = "img") -> Table4Sample:
    image = np.full((3, 2, 2), value, dtype=np.float32)
    return Table4Sample(
        image=image,
        true_label=true_label,
        image_id=image_id,
        class_name="goldfish",
    )


def test_table6_config_documents_imagenet_parameters() -> None:
    """The ImageNet Table 6 config should record the spec parameters."""
    config_path = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_6.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["experiment"]["name"] == "imagenet_table_6_adaptive_quantization"
    assert config["dataset"]["name"] == "imagenet"
    assert config["dataset"]["splits"]["train"] == [
        {"name": "goldfish", "label": 1, "path": "data/imagenet/train/goldfish"},
        {"name": "pineapple", "label": 953, "path": "data/imagenet/train/pineapple"},
        {
            "name": "digital_clock",
            "label": 530,
            "path": "data/imagenet/train/digital_clock",
        },
    ]
    assert config["dataset"]["splits"]["validation"] == [
        {"name": "jellyfish", "label": 107, "path": "data/imagenet/validation/jellyfish"}
    ]
    assert config["model"]["name"] == "googlenet_caffe"
    assert config["model"]["reference"] == "BVLC GoogLeNet"
    assert config["model"]["mean_file"] is None
    assert config["attack"]["name"] == "fgsm"
    assert config["attack"]["epsilon_255"] == 1.0
    assert config["quantization"]["method"] == "entropy_defined_adaptive_quantization"
    assert config["quantization"]["entropy_thresholds"] == {"low": 4.0, "medium": 5.0}
    assert config["quantization"]["interval_sizes"] == {
        "low_entropy": 128,
        "medium_entropy": 64,
        "high_entropy": 43,
    }
    assert config["output"]["csv"] == "table_6_imagenet.csv"
    assert config["output"]["diagnostics_csv"] == "table_6_imagenet_diagnostics.csv"


def test_adaptive_quantization_step_uses_table6_thresholds() -> None:
    """Entropy thresholds should map to the configured scalar steps."""
    assert adaptive_quantization_step(3.99) == 128
    assert adaptive_quantization_step(4.0) == 64
    assert adaptive_quantization_step(4.99) == 64
    assert adaptive_quantization_step(5.0) == 43


def test_adaptive_quantization_preserves_caffe_scale() -> None:
    """Caffe tensors should be quantized directly in 0-255 space."""
    image = np.asarray([[[129.0, 255.0], [64.0, 0.0]]] * 3, dtype=np.float32)

    quantized, entropy, step = adaptive_quantize_image(image)

    assert step == 128
    assert entropy >= 0.0
    np.testing.assert_array_equal(
        quantized,
        np.asarray([[[128.0, 128.0], [0.0, 0.0]]] * 3, dtype=np.float32),
    )


def test_evaluate_table6_counts_metrics_and_skip_reasons() -> None:
    """Table 6 should count only clean-correct and attack-successful pairs."""
    samples_by_split = {
        "train": [
            _sample(149.0, true_label=1, image_id="success"),
            _sample(100.0, true_label=1, image_id="fgsm_failed"),
            _sample(149.0, true_label=9, image_id="wrong_clean"),
        ],
        "validation": [_sample(149.0, true_label=1, image_id="valid_success")],
    }

    result = evaluate_table6_imagenet(
        model=MeanCaffeModel(),
        samples_by_split=samples_by_split,
        epsilon_255=2.0,
    )

    assert result.n_clean_total == 4
    assert result.n_clean_correct == 3
    assert result.n_attack_success == 2
    assert result.n_valid_detections == 2

    train_summary = result.summaries[0]
    assert train_summary.total_images == 3
    assert train_summary.clean_correct == 2
    assert train_summary.skipped_wrong_baseline == 1
    assert train_summary.fgsm_success == 1
    assert train_summary.disturbed_failure == 1
    assert train_summary.tp == 1
    assert train_summary.fn == 0
    assert train_summary.fp == 0
    assert train_summary.recall == pytest.approx(1.0)
    assert train_summary.precision == pytest.approx(1.0)
    assert train_summary.f1 == pytest.approx(1.0)

    skip_reasons = {row["image_id"]: row["skip_reason"] for row in result.diagnostics}
    assert skip_reasons["success"] == "none"
    assert skip_reasons["fgsm_failed"] == "fgsm_failed_to_change_prediction"
    assert skip_reasons["wrong_clean"] == "wrong_clean_prediction"

    failed_row = next(row for row in result.diagnostics if row["image_id"] == "fgsm_failed")
    assert failed_row["is_fp"] is False
    assert failed_row["is_tp"] is False
    assert failed_row["is_fn"] is False


def test_table6_validation_fails_when_fgsm_has_zero_successes() -> None:
    """The required sanity failure should be explicit when no attack succeeds."""
    result = evaluate_table6_imagenet(
        model=NoAttackModel(),
        samples_by_split={
            "train": [_sample(149.0, true_label=1)],
            "validation": [_sample(149.0, true_label=1)],
        },
        epsilon_255=2.0,
    )

    assert result.n_attack_success == 0
    with pytest.raises(RuntimeError) as excinfo:
        validate_table6_result(result)

    assert str(excinfo.value) == ZERO_ATTACK_SUCCESS_MESSAGE


def test_write_table6_outputs_uses_specified_csv_columns(tmp_path) -> None:
    """The output CSV files should keep the spec field order."""
    result = evaluate_table6_imagenet(
        model=MeanCaffeModel(),
        samples_by_split={
            "train": [_sample(149.0, true_label=1)],
            "validation": [_sample(149.0, true_label=1)],
        },
        epsilon_255=2.0,
    )

    csv_path, diagnostics_path = write_table6_outputs(tmp_path, result)

    assert csv_path.name == "table_6_imagenet.csv"
    assert diagnostics_path.name == "table_6_imagenet_diagnostics.csv"
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == ",".join(TABLE6_OUTPUT_FIELDS)
    assert len(csv_lines) == 3
    assert csv_lines[1].split(",")[0] == "train"
    assert csv_lines[2].split(",")[0] == "validation"
    assert diagnostics_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        TABLE6_DIAGNOSTIC_FIELDS
    )


def test_table6_script_loads_local_jpeg_and_png_split_folders(tmp_path) -> None:
    """The CLI loader should read supported local ImageNet image extensions."""
    Image = pytest.importorskip("PIL.Image")

    train_dir = tmp_path / "train" / "goldfish"
    validation_dir = tmp_path / "validation" / "jellyfish"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    image = np.full((3, 3, 3), 128, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(str(train_dir / "train_sample.JPEG"))
    Image.fromarray(image, mode="RGB").save(str(validation_dir / "validation_sample.png"))
    (train_dir / "ignored.txt").write_text("not an image", encoding="utf-8")

    config = {
        "dataset": {
            "image_size": 4,
            "splits": {
                "train": [{"name": "goldfish", "label": 1, "path": str(train_dir)}],
                "validation": [
                    {"name": "jellyfish", "label": 107, "path": str(validation_dir)}
                ],
            },
        }
    }

    samples_by_split = load_split_samples(config)

    assert list(samples_by_split) == ["train", "validation"]
    assert len(samples_by_split["train"]) == 1
    assert len(samples_by_split["validation"]) == 1
    assert samples_by_split["train"][0].image.shape == (4, 4, 3)
    assert samples_by_split["train"][0].true_label == 1
    assert samples_by_split["validation"][0].true_label == 107
