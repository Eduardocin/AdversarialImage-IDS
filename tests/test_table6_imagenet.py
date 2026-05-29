from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from deepdetector.io.paths import get_project_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.table4_imagenet import (  # noqa: E402
    ZERO_ATTACK_SUCCESS_MESSAGE,
    Table4Sample,
)
from deepdetector.evaluation.table6_imagenet import (  # noqa: E402
    TABLE6_OUTPUT_FIELDS,
    adaptive_quantization_step,
    adaptive_quantize_image,
    evaluate_table6_imagenet,
    validate_table6_result,
)
from deepdetector.experiments import table6_runner  # noqa: E402
from deepdetector.experiments.table6_runner import (  # noqa: E402
    aggregate_table6_rows,
    imagenet_fgsm_cache_path,
    load_imagenet_split_samples,
    run_table6_experiment,
)


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


class ExplodingGradientModel(MeanCaffeModel):
    """Model that proves cached ImageNet adversarials skip FGSM generation."""

    def gradient(self, image: np.ndarray, class_id: int) -> np.ndarray:
        raise AssertionError("cache miss")


def _sample(value: float, true_label: int = 1, image_id: str = "img") -> Table4Sample:
    image = np.full((3, 2, 2), value, dtype=np.float32)
    return Table4Sample(
        image=image,
        true_label=true_label,
        image_id=image_id,
        class_name="goldfish",
    )


def test_table6_config_documents_imagenet_parameters() -> None:
    """The official Table 6 config should embed ImageNet as an internal dataset."""
    all_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    config = all_config["experiments"]["table_6"]["imagenet"]

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
    assert config["attack"]["clip_max"] == 255.0


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


def test_table6_imagenet_reuses_cached_adversarial_examples() -> None:
    """Cached ImageNet Table 6 attacks should avoid gradient generation."""
    cached_adv = np.asarray([np.full((3, 2, 2), 151.0, dtype=np.float32)])

    result = evaluate_table6_imagenet(
        model=ExplodingGradientModel(),
        samples_by_split={
            "train": [
                _sample(149.0, true_label=1, image_id="cached_success"),
                _sample(149.0, true_label=9, image_id="wrong_clean"),
            ],
            "validation": [_sample(149.0, true_label=1, image_id="valid_success")],
        },
        epsilon_255=2.0,
        adversarial_by_split={
            "train": cached_adv,
            "validation": cached_adv,
        },
    )

    assert result.n_clean_correct == 2
    assert result.n_attack_success == 2
    assert result.summaries[0].skipped_wrong_baseline == 1
    assert result.adversarial_by_split["train"].shape == (1, 3, 2, 2)


def test_table6_imagenet_rejects_short_adversarial_cache() -> None:
    """A present but incompatible cache should not silently fall back to FGSM."""
    with pytest.raises(ValueError, match="cache is incompatible"):
        evaluate_table6_imagenet(
            model=ExplodingGradientModel(),
            samples_by_split={
                "train": [_sample(149.0, true_label=1, image_id="missing_cached_adv")],
                "validation": [_sample(149.0, true_label=1, image_id="valid_success")],
            },
            epsilon_255=2.0,
            adversarial_by_split={
                "train": np.empty((0, 3, 2, 2), dtype=np.float32),
                "validation": np.asarray([np.full((3, 2, 2), 151.0, dtype=np.float32)]),
            },
        )


def test_table6_imagenet_cache_path_defaults_to_artifacts() -> None:
    """The Table 6 ImageNet cache should reuse the shared artifacts path."""
    cache_path = imagenet_fgsm_cache_path(
        {
            "attack": {
                "name": "fgsm",
                "epsilon_255": 1.0,
            },
        },
        "validation",
    )

    assert cache_path == (
        PROJECT_ROOT
        / "artifacts"
        / "adversarial_examples"
        / "imagenet"
        / "fgsm"
        / "validation"
        / "adversarial_examples.npy"
    )


def test_table6_imagenet_runner_generates_missing_cache_then_reuses(
    monkeypatch, tmp_path
) -> None:
    """Missing ImageNet caches should be generated once and reused on rerun."""
    samples_by_split = {
        "train": [_sample(149.0, true_label=1, image_id="train_cached")],
        "validation": [_sample(149.0, true_label=1, image_id="validation_cached")],
    }
    config = {
        "split_order": ["train", "validation"],
        "dataset": {"name": "imagenet"},
        "model": {"name": "googlenet_caffe"},
        "attack": {
            "name": "fgsm",
            "epsilon_255": 2.0,
            "clip_min": 0.0,
            "clip_max": 255.0,
            "cache_dir": str(tmp_path / "adversarial_examples"),
        },
    }

    monkeypatch.setattr(
        table6_runner,
        "load_imagenet_split_samples",
        lambda _: samples_by_split,
    )
    monkeypatch.setattr(
        table6_runner,
        "build_imagenet_model",
        lambda _: MeanCaffeModel(),
    )

    first_rows = table6_runner.evaluate_imagenet_table6(config)

    train_cache = imagenet_fgsm_cache_path(config, "train")
    validation_cache = imagenet_fgsm_cache_path(config, "validation")
    assert train_cache.is_file()
    assert validation_cache.is_file()
    assert np.load(str(train_cache)).shape == (1, 3, 2, 2)

    monkeypatch.setattr(
        table6_runner,
        "build_imagenet_model",
        lambda _: ExplodingGradientModel(),
    )

    second_rows = table6_runner.evaluate_imagenet_table6(config)

    assert second_rows == first_rows


def test_aggregate_table6_rows_sums_counts_before_calculating_metrics() -> None:
    """Combined Table 6 metrics should be calculated after summing counters."""
    rows = aggregate_table6_rows(
        mnist_rows=[
            {"split": "train", "TP": 1, "FN": 1, "FP": 0},
            {"split": "validation", "TP": 2, "FN": 0, "FP": 1},
        ],
        imagenet_rows=[
            {"split": "train", "TP": 3, "FN": 1, "FP": 2},
            {"split": "validation", "TP": 1, "FN": 3, "FP": 0},
        ],
    )

    assert rows[0]["split"] == "train"
    assert rows[0]["TP"] == 4
    assert rows[0]["FN"] == 2
    assert rows[0]["FP"] == 2
    assert rows[0]["recall_percent"] == pytest.approx(66.6666667)
    assert rows[0]["precision_percent"] == pytest.approx(66.6666667)
    assert rows[0]["f1_percent"] == pytest.approx(66.6666667)
    assert rows[1]["split"] == "validation"
    assert rows[1]["TP"] == 3
    assert rows[1]["FN"] == 3
    assert rows[1]["FP"] == 1


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

    samples_by_split = load_imagenet_split_samples(config)

    assert list(samples_by_split) == ["train", "validation"]
    assert len(samples_by_split["train"]) == 1
    assert len(samples_by_split["validation"]) == 1
    assert samples_by_split["train"][0].image.shape == (4, 4, 3)
    assert samples_by_split["train"][0].true_label == 1
    assert samples_by_split["validation"][0].true_label == 107


def test_run_table6_experiment_writes_only_official_outputs(monkeypatch, tmp_path) -> None:
    """The official Table 6 runner should write only metrics CSV and JSON."""
    monkeypatch.setattr(
        table6_runner,
        "evaluate_mnist_table6",
        lambda config: [
            {"split": "train", "TP": 1, "FN": 1, "FP": 0},
            {"split": "validation", "TP": 2, "FN": 0, "FP": 1},
        ],
    )
    monkeypatch.setattr(
        table6_runner,
        "evaluate_imagenet_table6",
        lambda config: [
            {"split": "train", "TP": 3, "FN": 1, "FP": 2},
            {"split": "validation", "TP": 1, "FN": 3, "FP": 0},
        ],
    )

    rows = run_table6_experiment(
        {
            "experiment_id": "table_6",
            "kind": "table_6",
            "mnist": {"dataset": {"name": "mnist"}},
            "imagenet": {"dataset": {"name": "imagenet"}},
            "split_order": ["train", "validation"],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert [row["split"] for row in rows] == ["train", "validation"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == (
        ",".join(TABLE6_OUTPUT_FIELDS)
    )
    payload = yaml.safe_load((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["train"]["tp"] == 4
    assert payload["train"]["fn"] == 2
    assert payload["train"]["fp"] == 2


def test_legacy_table6_imagenet_public_files_were_removed() -> None:
    """Table 6 should not expose a separate ImageNet experiment path."""
    assert not (
        PROJECT_ROOT / "scripts" / "article_reproduction" / "table_6_imagenet.py"
    ).exists()
    assert not (
        PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_6.yaml"
    ).exists()
