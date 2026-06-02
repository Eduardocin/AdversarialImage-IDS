import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.adaptive_cw_l2 import (  # noqa: E402
    generate_adaptive_cw_l2_attack,
)
from deepdetector.attacks.registry import ATTACK_REGISTRY  # noqa: E402
from deepdetector.evaluation import defense_aware as defense_aware_module  # noqa: E402
from deepdetector.evaluation.defense_aware import (  # noqa: E402
    DEFENSE_AWARE_SCHEMA,
    evaluate_defense_aware_arrays,
    rows_to_metrics_json,
    save_defense_aware_outputs,
)
from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from deepdetector.filters.adaptive_noise_reduction import (  # noqa: E402
    build_final_adaptive_detection_filter,
    final_adaptive_detection_filter,
)


def _consolidated_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )


def test_defense_aware_config_matches_spec() -> None:
    config = _consolidated_config()
    experiment = config["experiments"]["defense_aware"]

    assert experiment["kind"] == "defense_aware"
    assert experiment["output_dir"] == "results/experiments/defense_aware"
    assert experiment["seed"] == 42
    assert experiment["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 9000,
        "end": 10000,
        "value_range": {"min": 0.0, "max": 1.0},
    }
    assert "samples" not in experiment["dataset"]
    assert experiment["model"]["name"] == "mnist_m2"
    assert "attack" not in experiment
    assert set(experiment["attacks"]) == {"defense_unaware", "defense_aware"}
    assert experiment["attacks"]["defense_unaware"]["type"] == "cw_l2"
    assert experiment["attacks"]["defense_aware"]["type"] == "adaptive_cw_l2"
    assert experiment["detector"]["type"] == "final_adaptive_detection_filter"
    assert experiment["detector"]["spatial_filter"] == {
        "type": "cross_mean",
        "radius": 3,
    }


def test_build_experiment_config_preserves_two_internal_attacks() -> None:
    config = experiment_runner.build_experiment_config(
        "defense_aware",
        _consolidated_config(),
    )

    assert config["kind"] == "defense_aware"
    assert "attack" in config
    assert config["attack"] == {}
    assert set(config["attacks"]) == {"defense_unaware", "defense_aware"}
    assert config["detector"]["spatial_filter"]["radius"] == 3


def test_defense_aware_dispatches_through_central_runner(monkeypatch) -> None:
    calls = []

    def fake_run(config):
        calls.append((config["experiment_id"], config["kind"]))
        return [{"attack": "defense_unaware"}]

    monkeypatch.setattr(experiment_runner, "run_defense_aware", fake_run)

    result = experiment_runner.run_experiment("defense_aware", _consolidated_config())

    assert result == [{"attack": "defense_unaware"}]
    assert calls == [("defense_aware", "defense_aware")]


def test_final_adaptive_detection_filter_preserves_unit_range() -> None:
    image = np.asarray([[-0.25, 0.0, 0.5, 1.0, 1.25]], dtype=np.float32)

    filtered = final_adaptive_detection_filter(image)

    assert filtered.shape == image.shape
    assert float(np.min(filtered)) >= 0.0
    assert float(np.max(filtered)) <= 1.0


def test_detector_builder_uses_cross_mean_radius(monkeypatch) -> None:
    calls = []

    def fake_entropy(image):
        return 6.0

    def fake_quantization(image, interval, left=True):
        return np.full_like(image, 0.4, dtype=np.float32)

    def fake_cross(image, radius=1):
        calls.append(radius)
        return np.full_like(image, 0.6, dtype=np.float32)

    monkeypatch.setattr(
        "deepdetector.filters.adaptive_noise_reduction.one_d_entropy",
        fake_entropy,
    )
    monkeypatch.setattr(
        "deepdetector.filters.adaptive_noise_reduction.scalar_quantization",
        fake_quantization,
    )
    monkeypatch.setattr(
        "deepdetector.filters.adaptive_noise_reduction.cross_mean_filter",
        fake_cross,
    )

    transform = build_final_adaptive_detection_filter(
        {
            "entropy_thresholds": {"low": 4.0, "medium": 5.0},
            "quantization": {
                "low_entropy_step": 128,
                "medium_entropy_step": 64,
                "high_entropy_step": 43,
            },
            "spatial_filter": {"type": "cross_mean", "radius": 3},
        }
    )
    filtered = transform(np.full((3, 3, 1), 0.5, dtype=np.float32))

    assert calls == [3]
    assert filtered.shape == (3, 3, 1)


def test_adaptive_cw_l2_is_registered() -> None:
    assert "adaptive_cw_l2" in ATTACK_REGISTRY


def test_adaptive_cw_l2_accepts_only_undetected_misclassifications() -> None:
    images = np.asarray([[[[0.0]]], [[[1.0]]], [[[2.0]]]], dtype=np.float32)
    labels = np.asarray([0, 0, 0], dtype=np.int32)
    candidates = np.asarray([[[[1.0]]], [[[2.0]]], [[[-1.0]]]], dtype=np.float32)

    def base_attack_fn(**kwargs):
        return candidates

    def predict_fn(batch):
        values = batch.reshape((len(batch), -1))[:, 0]
        return np.asarray([0 if value in (0.0, -1.0) else int(value) for value in values])

    def transform_fn(image):
        if float(image.reshape(-1)[0]) == 2.0:
            return np.asarray([[[1.0]]], dtype=np.float32)
        return image

    result = generate_adaptive_cw_l2_attack(
        model=object(),
        images=images,
        labels=labels,
        transform_fn=transform_fn,
        predict_fn=predict_fn,
        base_attack_fn=base_attack_fn,
    )

    np.testing.assert_array_equal(result[0], candidates[0])
    np.testing.assert_array_equal(result[1], images[1])
    np.testing.assert_array_equal(result[2], images[2])


def test_evaluate_defense_aware_arrays_counts_success_failure_and_l2() -> None:
    images = np.asarray([[[[0.0]]], [[[10.0]]], [[[2.0]]]], dtype=np.float32)
    labels = np.asarray([0, 1, 2], dtype=np.int32)

    def predict_fn(batch):
        values = batch.reshape((len(batch), -1))[:, 0]
        labels_by_value = {
            0.0: 0,
            1.0: 1,
            2.0: 2,
            3.0: 3,
            4.0: 4,
            5.0: 5,
            6.0: 6,
            10.0: 9,
        }
        return np.asarray([labels_by_value[float(value)] for value in values])

    def transform_fn(image):
        value = float(np.asarray(image).reshape(-1)[0])
        if value == 1.0:
            return np.asarray([[[0.0]]], dtype=np.float32)
        return image

    def blind_attack(clean_image, true_label, clean_pred):
        if true_label == 0:
            return np.asarray([[[1.0]]], dtype=np.float32)
        return clean_image

    def adaptive_attack(clean_image, true_label, clean_pred):
        if true_label == 0:
            return np.asarray([[[3.0]]], dtype=np.float32)
        return np.asarray([[[4.0]]], dtype=np.float32)

    rows = evaluate_defense_aware_arrays(
        images=images,
        labels=labels,
        predict_fn=predict_fn,
        defense_unaware_attack_fn=blind_attack,
        defense_aware_attack_fn=adaptive_attack,
        transform_fn=transform_fn,
    )

    assert rows[0] == {
        "attack": "defense_unaware",
        "total_valid": 2,
        "success": 1,
        "detected": 1,
        "undetected": 0,
        "failures": 1,
        "attack_success_rate_percent": 50.0,
        "detection_rate_percent": 100.0,
        "evasion_rate_percent": 0.0,
        "failure_rate_percent": 50.0,
        "mean_l2": 1.0,
    }
    assert rows[1] == {
        "attack": "defense_aware",
        "total_valid": 2,
        "success": 2,
        "detected": 0,
        "undetected": 2,
        "failures": 0,
        "attack_success_rate_percent": 100.0,
        "detection_rate_percent": 0.0,
        "evasion_rate_percent": 100.0,
        "failure_rate_percent": 0.0,
        "mean_l2": 2.5,
    }


def test_rows_to_metrics_json_omits_metadata() -> None:
    row = {
        "attack": "defense_unaware",
        "total_valid": 1,
        "success": 1,
        "detected": 0,
        "undetected": 1,
        "failures": 0,
        "attack_success_rate_percent": 100.0,
        "detection_rate_percent": 0.0,
        "evasion_rate_percent": 100.0,
        "failure_rate_percent": 0.0,
        "mean_l2": 1.0,
    }

    payload = rows_to_metrics_json([row])

    assert payload == {
        "defense_unaware": {
            "total_valid": 1,
            "success": 1,
            "detected": 0,
            "undetected": 1,
            "failures": 0,
            "attack_success_rate_percent": 100.0,
            "detection_rate_percent": 0.0,
            "evasion_rate_percent": 100.0,
            "failure_rate_percent": 0.0,
            "mean_l2": 1.0,
        }
    }


def test_save_defense_aware_outputs_writes_only_official_artifacts(tmp_path) -> None:
    rows = [
        {
            "attack": "defense_unaware",
            "total_valid": 1,
            "success": 1,
            "detected": 0,
            "undetected": 1,
            "failures": 0,
            "attack_success_rate_percent": 100.0,
            "detection_rate_percent": 0.0,
            "evasion_rate_percent": 100.0,
            "failure_rate_percent": 0.0,
            "mean_l2": 1.0,
        },
        {
            "attack": "defense_aware",
            "total_valid": 1,
            "success": 0,
            "detected": 0,
            "undetected": 0,
            "failures": 1,
            "attack_success_rate_percent": 0.0,
            "detection_rate_percent": 0.0,
            "evasion_rate_percent": 0.0,
            "failure_rate_percent": 100.0,
            "mean_l2": 0.0,
        },
    ]

    save_defense_aware_outputs(rows=rows, output_dir=tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    assert csv_rows[0] == DEFENSE_AWARE_SCHEMA

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload) == {"defense_unaware", "defense_aware"}
    assert "metadata" not in payload
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "diagnostic.json").exists()
    assert not (tmp_path / "adversarial_examples").exists()


def test_run_defense_aware_evaluation_uses_shared_transform_in_adaptive_attack(
    monkeypatch,
) -> None:
    images = np.asarray([[[[0.0]]]], dtype=np.float32)
    labels = np.asarray([0], dtype=np.int32)
    captured = {}

    monkeypatch.setattr(
        defense_aware_module,
        "_load_images",
        lambda config: (images, labels),
    )
    monkeypatch.setattr(
        defense_aware_module,
        "create_restored_mnist_m2_graph",
        lambda train_dir: {"model": object(), "sess": object(), "x": object(), "predictions": object()},
    )
    monkeypatch.setattr(
        defense_aware_module,
        "_predict_fn",
        lambda graph, batch_size: lambda batch: np.asarray(
            [int(round(float(value))) for value in batch.reshape((len(batch), -1))[:, 0]]
        ),
    )

    def fake_build_filter(config):
        def transform(image):
            return image

        captured["transform"] = transform
        return transform

    def fake_cw(**kwargs):
        return np.asarray([[[[1.0]]]], dtype=np.float32)

    def fake_adaptive(**kwargs):
        captured["adaptive_transform"] = kwargs["transform_fn"]
        return np.asarray([[[[1.0]]]], dtype=np.float32)

    monkeypatch.setattr(
        defense_aware_module,
        "build_final_adaptive_detection_filter",
        fake_build_filter,
    )
    monkeypatch.setattr(defense_aware_module, "generate_cw_l2_attack", fake_cw)
    monkeypatch.setattr(
        defense_aware_module,
        "generate_adaptive_cw_l2_attack",
        fake_adaptive,
    )

    rows = defense_aware_module.run_defense_aware_evaluation(
        {
            "seed": 42,
            "dataset": {"name": "mnist", "start": 9000, "end": 10000},
            "model": {},
            "evaluation": {"batch_size": 1},
            "attacks": {
                "defense_unaware": {"type": "cw_l2"},
                "defense_aware": {"type": "adaptive_cw_l2"},
            },
            "detector": {"type": "final_adaptive_detection_filter"},
        }
    )

    assert captured["adaptive_transform"] is captured["transform"]
    assert rows[0]["success"] == 1
    assert rows[1]["success"] == 1
