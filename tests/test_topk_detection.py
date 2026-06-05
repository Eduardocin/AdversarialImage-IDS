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

from deepdetector.evaluation.topk_detection import (  # noqa: E402
    METRIC_FIELDS,
    RULE_METRIC_FIELDS,
    SELECTION_FIELDS,
    evaluate_topk_detection,
    is_detected_topk,
    top_k_indices,
    write_topk_detection_outputs,
)
from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from deepdetector.experiments import topk_detection as topk_runner  # noqa: E402


class FakeModel:
    """Tiny score lookup model keyed by the first scalar in the image."""

    def __init__(self, scores_by_key):
        self.scores_by_key = {
            int(key): np.asarray(scores, dtype=np.float32)
            for key, scores in scores_by_key.items()
        }

    def predict(self, image):
        key = int(np.asarray(image).reshape(-1)[0])
        return self.scores_by_key[key]


class DualGradientTopKModel:
    """Model where prediction gradients are zero and attack gradients are usable."""

    def prediction_gradient(self, image, class_id):
        return np.zeros_like(image, dtype=np.float32)

    def gradient(self, image, class_id):
        return np.ones_like(image, dtype=np.float32)


def _image(key: int) -> np.ndarray:
    return np.asarray([key], dtype=np.float32)


def _rgb_image(key: int) -> np.ndarray:
    return np.full((2, 2, 3), float(key), dtype=np.float32)


def _config(max_samples_per_class=50):
    return {
        "attack": {"name": "fgsm", "epsilon": 1.0 / 255.0, "clip_max": 255.0},
        "ambiguity_selection": {
            "max_margin": 0.15,
            "min_top1_confidence": 0.20,
            "max_top1_confidence": 0.60,
            "max_samples_per_class": max_samples_per_class,
        },
        "topk_detection": {
            "baseline_k": 1,
            "candidate_k_values": [2, 3, 5],
        },
    }


def _rules_config():
    config = _config()
    config["dataset"] = {"class_indices": {"cab": 468}}
    config["topk_detection"]["candidate_k_values"] = [2]
    config["topk_detection"]["rules"] = [
        {"name": "topk_overlap"},
        {"name": "top1_in_topk"},
        {"name": "rank_displacement", "thresholds": [1, 2]},
        {"name": "confidence_drop", "ks": [2], "deltas": [0.10]},
    ]
    return config


def test_top_k_detection_rule_and_baseline_equivalence() -> None:
    """k=1 should match the original argmax-change detector."""
    before = np.asarray([0.4, 0.3, 0.2, 0.1])
    after_same_top2 = np.asarray([0.1, 0.5, 0.3, 0.1])
    after_disjoint_top2 = np.asarray([0.1, 0.2, 0.4, 0.3])

    assert top_k_indices(before, 2) == [0, 1]
    assert is_detected_topk(before, after_same_top2, k=1) is True
    assert is_detected_topk(before, after_same_top2, k=2) is False
    assert is_detected_topk(before, after_disjoint_top2, k=2) is True
    assert is_detected_topk(before, after_same_top2, k=1) == (
        int(np.argmax(before)) != int(np.argmax(after_same_top2))
    )


def test_topk_fgsm_uses_attack_gradient_not_zero_prediction_gradient() -> None:
    """Top-k FGSM should use deploy_removeSoftmax gradients via model.gradient."""
    image = np.full((3, 2, 2), 128.0, dtype=np.float32)

    adversarial = topk_runner._generate_fgsm_googlenet(
        model=DualGradientTopKModel(),
        image=image,
        class_id=1,
        attack_config={"epsilon_255": 1.0, "clip_min": 0.0, "clip_max": 255.0},
    )

    np.testing.assert_array_equal(
        adversarial,
        np.full((3, 2, 2), 129.0, dtype=np.float32),
    )


def test_topk_evaluation_uses_and_ambiguity_and_does_not_fill_selection() -> None:
    """Only samples matching margin and confidence are selected."""
    model = FakeModel(
        {
            1: [0.40, 0.35, 0.10, 0.06, 0.05, 0.04],
            2: [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
            3: [0.50, 0.20, 0.10, 0.08, 0.07, 0.05],
            101: [0.35, 0.40, 0.10, 0.06, 0.05, 0.04],
            201: [0.35, 0.05, 0.40, 0.06, 0.05, 0.09],
            301: [0.40, 0.05, 0.35, 0.06, 0.05, 0.09],
        }
    )

    def transform(image):
        return image + 100

    def attack_generator(model, image, class_id, attack_config):
        assert class_id == 0
        assert attack_config["name"] == "fgsm"
        return image + 200

    result = evaluate_topk_detection(
        samples_by_class={"cab": [_image(1), _image(2), _image(3)]},
        model=model,
        transform=transform,
        attack_generator=attack_generator,
        config=_config(max_samples_per_class=3),
    )

    selection = result.selection_json["cab"]
    assert selection["total_loaded"] == 3
    assert selection["ambiguous_candidates"] == 1
    assert selection["selected_for_experiment"] == 1
    assert selection["mean_top1_confidence"] == pytest.approx(0.40)
    assert selection["mean_top1_top2_margin"] == pytest.approx(0.05)
    assert result.selection_json["global"]["selected_for_experiment"] == 1

    cab_k1 = result.metrics_json["cab"]["1"]
    cab_k2 = result.metrics_json["cab"]["2"]
    global_k1 = result.metrics_json["global"]["1"]

    assert cab_k1["selected"] == 1
    assert cab_k1["test_number"] == 1
    assert cab_k1["disturbed_failure"] == 0
    assert cab_k1["FP"] == 1
    assert cab_k1["TN"] == 0
    assert cab_k1["TP"] == 1
    assert cab_k1["FN"] == 0
    assert cab_k1["recall_percent"] == pytest.approx(100.0)
    assert cab_k1["precision_percent"] == pytest.approx(50.0)
    assert cab_k2["FP"] == 0
    assert cab_k2["TN"] == 1
    assert cab_k2["TP"] == 0
    assert cab_k2["FN"] == 1
    assert cab_k2["false_positive_reduction_percent"] == pytest.approx(100.0)
    assert global_k1["FP"] == 1
    assert global_k1["TP"] == 1


def test_topk_outputs_write_only_official_artifacts(tmp_path) -> None:
    """The writer should materialize exactly selection and metrics CSV/JSON."""
    model = FakeModel(
        {
            1: [0.40, 0.35, 0.10, 0.06, 0.05, 0.04],
            101: [0.35, 0.40, 0.10, 0.06, 0.05, 0.04],
            201: [0.35, 0.05, 0.40, 0.06, 0.05, 0.09],
            301: [0.40, 0.05, 0.35, 0.06, 0.05, 0.09],
        }
    )
    result = evaluate_topk_detection(
        samples_by_class={"cab": [_rgb_image(1)]},
        model=model,
        transform=lambda image: image + 100,
        attack_generator=lambda model, image, class_id, attack_config: image + 200,
        config=_config(),
    )
    stale_dir = tmp_path / "ambiguous_images" / "cab"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.png").write_bytes(b"old")

    outputs = write_topk_detection_outputs(output_dir=tmp_path, result=result)

    assert set(outputs) == {
        "ambiguous_images",
        "selection_csv",
        "selection_json",
        "metrics_csv",
        "metrics_json",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "ambiguous_images",
        "metrics.csv",
        "metrics.json",
        "selection.csv",
        "selection.json",
    ]
    assert sorted(path.name for path in (tmp_path / "ambiguous_images" / "cab").iterdir()) == [
        "000001.png"
    ]

    with (tmp_path / "selection.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(SELECTION_FIELDS)
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(METRIC_FIELDS)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["global"]["1"]["selected"] == 1


def test_topk_configured_rules_use_official_outputs_and_audit(tmp_path) -> None:
    """Configured rules should use metrics.csv/json without standalone diagnostics."""
    model = FakeModel(
        {
            1: [0.40, 0.35, 0.10, 0.06, 0.05, 0.04],
            101: [0.35, 0.40, 0.10, 0.06, 0.05, 0.04],
            201: [0.35, 0.05, 0.40, 0.06, 0.05, 0.09],
            301: [0.40, 0.05, 0.35, 0.06, 0.05, 0.09],
        }
    )
    result = evaluate_topk_detection(
        samples_by_class={"cab": [_rgb_image(1)]},
        model=model,
        transform=lambda image: image + 100,
        attack_generator=lambda model, image, class_id, attack_config: image + 200,
        config=_rules_config(),
    )

    assert result.metrics_json["metrics"] == result.metric_rows
    assert result.metrics_json["sample_audit"]
    assert {
        "rank_of_top1_before_in_after",
        "confidence_of_top1_before_in_after",
    }.issubset(result.metrics_json["sample_audit"][0])
    assert {
        (row["class"], row["rule"], row["k"], row["threshold"], row["delta"])
        for row in result.metric_rows
    } == {
        ("cab", "topk_overlap", 1, None, None),
        ("cab", "topk_overlap", 2, None, None),
        ("cab", "top1_in_topk", 1, None, None),
        ("cab", "top1_in_topk", 2, None, None),
        ("cab", "rank_displacement", None, 1, None),
        ("cab", "rank_displacement", None, 2, None),
        ("cab", "confidence_drop", 2, None, 0.10),
        ("global", "topk_overlap", 1, None, None),
        ("global", "topk_overlap", 2, None, None),
        ("global", "top1_in_topk", 1, None, None),
        ("global", "top1_in_topk", 2, None, None),
        ("global", "rank_displacement", None, 1, None),
        ("global", "rank_displacement", None, 2, None),
        ("global", "confidence_drop", 2, None, 0.10),
    }

    outputs = write_topk_detection_outputs(output_dir=tmp_path, result=result)

    assert set(outputs) == {
        "ambiguous_images",
        "selection_csv",
        "selection_json",
        "metrics_csv",
        "metrics_json",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "ambiguous_images",
        "metrics.csv",
        "metrics.json",
        "selection.csv",
        "selection.json",
    ]
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(RULE_METRIC_FIELDS)
    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "sample_audit" in metrics_payload
    assert metrics_payload["metrics"][0]["rule"] == "topk_overlap"


def test_topk_detection_config_matches_spec() -> None:
    """The consolidated config should expose the single top-k public experiment."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    experiment = config["experiments"]["topk_detection"]

    assert experiment["kind"] == "topk_detection"
    assert experiment["output_dir"] == "results/experiments/topk_detection"
    assert experiment["dataset"]["split"] == "test"
    assert experiment["dataset"]["images_dir"] == "data/imagenet/test"
    assert experiment["dataset"]["class_indices"] == {
        "cab": 468,
        "panda": 388,
        "zebra": 340,
    }
    assert experiment["model"]["name"] == "googlenet_caffe"
    assert experiment["attack"]["name"] == "fgsm"
    assert experiment["attack"]["epsilon"] == pytest.approx(1.0 / 255.0)
    assert experiment["attack"]["clip_max"] == 255.0
    assert experiment["attack"]["attack_model"] == "FGSM (epsilon=1/255)/GoogLeNet"
    assert experiment["filter"]["type"] == "proposed_detection_filter"
    assert experiment["filter"]["implementation"] == "article_final_detection_filter"
    assert experiment["ambiguity_selection"]["max_samples_per_class"] == 50
    assert experiment["topk_detection"]["candidate_k_values"] == [2, 3, 5]
    assert experiment["topk_detection"]["rules"] == [
        {"name": "topk_overlap"},
        {"name": "top1_in_topk"},
        {"name": "rank_displacement", "thresholds": [1, 2, 3, 5]},
        {"name": "confidence_drop", "ks": [2, 3], "deltas": [0.05, 0.10, 0.20]},
    ]
    assert "top1_detection" not in config["experiments"]
    assert "top3_detection" not in config["experiments"]
    assert "ambiguous_images" not in config["experiments"]


def test_topk_detection_runner_dispatches_official_experiment(monkeypatch, tmp_path) -> None:
    """The central runner should route kind=topk_detection to the top-k runner."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    config["experiments"]["topk_detection"]["output_dir"] = str(tmp_path)
    captured = {}

    def fake_topk_runner(config):
        captured.update(config)
        return {"metrics_csv": str(tmp_path / "metrics.csv")}

    monkeypatch.setattr(experiment_runner, "run_topk_detection_experiment", fake_topk_runner)

    result = experiment_runner.run_experiment("topk_detection", config)

    assert result == {"metrics_csv": str(tmp_path / "metrics.csv")}
    assert captured["kind"] == "topk_detection"
    assert captured["filter"]["type"] == "proposed_detection_filter"
    assert captured["topk_detection"]["baseline_k"] == 1
    assert captured["output"]["dir"] == str(tmp_path)


def test_topk_dataset_validation_rejects_missing_class(tmp_path) -> None:
    """Configured ImageNet test classes must exist locally."""
    from PIL import Image

    images_dir = tmp_path / "imagenet" / "test"
    (images_dir / "cab").mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
        str(images_dir / "cab" / "sample.jpg")
    )

    with pytest.raises(ValueError, match="Missing ImageNet class directory"):
        topk_runner.load_topk_imagenet_samples_by_class(
            {
                "dataset": {
                    "images_dir": str(images_dir),
                    "image_size": 224,
                    "class_indices": {"cab": 468, "panda": 388},
                }
            }
        )
