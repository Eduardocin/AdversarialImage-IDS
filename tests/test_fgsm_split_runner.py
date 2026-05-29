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

from deepdetector.experiments import fgsm_split_runner  # noqa: E402
from deepdetector.experiments.adversarial_examples import AdversarialExampleSet  # noqa: E402


def _config(tmp_path) -> dict:
    return {
        "experiment_id": "split_test",
        "kind": "split_eval",
        "dataset": {
            "name": "mnist",
            "slices": [
                {"name": "Training", "start": 0, "end": 1},
                {"name": "Validation", "start": 1, "end": 2},
            ]
        },
        "model": {"checkpoint_dir": "artifacts/model"},
        "attack": {"epsilon": 0.3, "clip_min": 0.0, "clip_max": 1.0},
        "filter": {"name": "fake", "type": "fake"},
        "evaluation": {"exclude_invalid_pairs": True, "batch_size": 4},
        "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
    }


def test_fgsm_split_runner_uses_one_graph_and_writes_standard_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    """The split runner should restore once, iterate YAML slices, and write CSV/JSON."""
    calls = {"graph": 0, "filter": 0, "materialize": 0, "evaluate": 0, "close": 0}
    materialized_slices = []

    monkeypatch.setattr(
        fgsm_split_runner,
        "build_filter_from_config",
        lambda config: (
            calls.__setitem__("filter", calls["filter"] + 1) or "fake",
            lambda image: image,
            {"filter_name": "fake", "filter_type": "fake"},
        ),
    )
    monkeypatch.setattr(
        fgsm_split_runner,
        "create_restored_mnist_graph",
        lambda checkpoint_dir: calls.__setitem__("graph", calls["graph"] + 1) or {
            "checkpoint_dir": checkpoint_dir
        },
    )

    def fake_materialize(config, graph=None):
        calls["materialize"] += 1
        assert graph == {
            "checkpoint_dir": str(PROJECT_ROOT / "artifacts" / "model")
        }
        dataset_config = config["dataset"]
        materialized_slices.append((dataset_config["start"], dataset_config["end"]))
        n_images = int(dataset_config["end"]) - int(dataset_config["start"])
        return AdversarialExampleSet(
            graph=graph,
            images=np.zeros((n_images, 2, 2, 1), dtype=np.float32),
            labels=np.zeros((n_images,), dtype=np.int64),
            adversarial_images=np.ones((n_images, 2, 2, 1), dtype=np.float32),
            clean_predictions=np.zeros((n_images,), dtype=np.int64),
            adversarial_predictions=np.ones((n_images,), dtype=np.int64),
            metadata={"slice_name": dataset_config["slice_name"]},
        )

    def fake_evaluate(**kwargs):
        calls["evaluate"] += 1
        assert kwargs["exclude_invalid_pairs"] is True
        assert kwargs["batch_size"] == 4
        assert kwargs["adv_images"].shape == kwargs["images"].shape
        return {
            "TP": calls["evaluate"],
            "FN": 2,
            "FP": 3,
            "recall_percent": 25.0,
            "precision_percent": 50.0,
            "f1_percent": 33.33,
        }

    monkeypatch.setattr(
        fgsm_split_runner,
        "prepare_mnist_fgsm_adversarial_set",
        fake_materialize,
    )
    monkeypatch.setattr(
        fgsm_split_runner,
        "evaluate_filter_on_existing_adversarial",
        fake_evaluate,
    )
    monkeypatch.setattr(
        fgsm_split_runner,
        "close_graph",
        lambda graph: calls.__setitem__("close", calls["close"] + 1),
    )

    rows = fgsm_split_runner.run_fgsm_split_experiment(_config(tmp_path))

    assert calls == {"graph": 1, "filter": 1, "materialize": 2, "evaluate": 2, "close": 1}
    assert materialized_slices == [(0, 1), (1, 2)]
    assert [row["split"] for row in rows] == ["Training", "Validation"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "split,TP,FN,FP,recall_percent,precision_percent,f1_percent",
        "Training,1,2,3,25.0,50.0,33.33",
        "Validation,2,2,3,25.0,50.0,33.33",
    ]
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "split_test"
    assert payload["kind"] == "split_eval"
    assert payload["extra"]["filter"] == {
        "filter_name": "fake",
        "filter_type": "fake",
    }
    assert len(payload["metrics"]) == 2


def test_fgsm_split_runner_rejects_missing_slices(tmp_path) -> None:
    """Missing dataset.slices should fail before model restoration."""
    config = _config(tmp_path)
    config["dataset"] = {"name": "mnist"}

    with pytest.raises(ValueError, match="dataset.slices"):
        fgsm_split_runner.run_fgsm_split_experiment(config)


def test_table9_is_combined_mnist_imagenet_like_table6() -> None:
    """Table 9 should declare MNIST and ImageNet internal components."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table6 = config["experiments"]["table_6"]
    table9 = config["experiments"]["table_9"]

    assert table6["kind"] == "table_6"
    assert table9["kind"] == "table_9"
    assert table9["datasets"] == ["mnist", "imagenet"]
    assert table9["mnist"]["filter"]["type"] == "proposed_detection_filter"
    assert table9["imagenet"]["filter"]["type"] == "proposed_detection_filter"
    assert table6["mnist"]["dataset"]["slices"] == table9["mnist"]["dataset"]["slices"]
    assert table6["imagenet"]["dataset"]["splits"] == table9["imagenet"]["dataset"]["splits"]
    assert table6["output_dir"] == "results/experiments/table_6"
    assert table9["output_dir"] == "results/experiments/table_9"
    assert table6["mnist"]["attack"]["epsilon"] == 0.2
    assert table9["mnist"]["attack"]["epsilon"] == 0.2
    assert table9["imagenet"]["attack"]["epsilon_255"] == 1.0
    assert table6["mnist"]["evaluation"]["batch_size"] == 256
    assert table9["mnist"]["evaluation"]["batch_size"] == 256


def test_table6_and_table9_per_table_scripts_were_removed() -> None:
    """Table 6/9 should use the single experiment entry point."""
    for relative_path in (
        "scripts/experiments/table_6.py",
        "scripts/experiments/table_9.py",
        "scripts/article_reproduction/table_6.py",
        "scripts/article_reproduction/table_9.py",
    ):
        assert not (PROJECT_ROOT / relative_path).exists()

    source = (PROJECT_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert "run_experiment" in source
    assert "evaluate_filter_on_images" not in source
    assert "create_restored_mnist_graph" not in source
