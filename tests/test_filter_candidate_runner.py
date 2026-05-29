import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments.adversarial_examples import AdversarialExampleSet  # noqa: E402
from deepdetector.experiments import filter_candidate_runner  # noqa: E402
from deepdetector.experiments.runner import build_experiment_config  # noqa: E402


def _context() -> AdversarialExampleSet:
    graph = {"sess": "session"}
    images = np.zeros((2, 3, 3, 1), dtype=np.float32)
    labels = np.asarray([1, 2], dtype=np.int64)
    adversarial_images = np.ones((2, 3, 3, 1), dtype=np.float32)
    clean_predictions = np.asarray([1, 2], dtype=np.int64)
    adversarial_predictions = np.asarray([2, 1], dtype=np.int64)
    return AdversarialExampleSet(
        graph=graph,
        images=images,
        labels=labels,
        adversarial_images=adversarial_images,
        clean_predictions=clean_predictions,
        adversarial_predictions=adversarial_predictions,
        metadata={
            "num_loaded": 3,
            "num_after_entropy_filter": 2,
            "high_entropy_only": True,
        },
    )


def test_filter_candidate_runner_reuses_context_and_writes_standard_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    """The generic runner should prepare context once and emit CSV/JSON only."""
    calls = {"context": 0, "evaluate": 0, "close": 0}

    def fake_prepare(config):
        del config
        calls["context"] += 1
        return _context()

    def fake_evaluate(**kwargs):
        calls["evaluate"] += 1
        assert kwargs["images"].shape == (2, 3, 3, 1)
        assert kwargs["adv_images"].shape == (2, 3, 3, 1)
        assert kwargs["clean_pred"].tolist() == [1, 2]
        assert kwargs["adv_pred"].tolist() == [2, 1]
        return {
            "TP": calls["evaluate"],
            "FN": 2,
            "FP": 3,
            "recall_percent": 25.0 * calls["evaluate"],
            "precision_percent": 50.0,
            "f1_percent": 40.0 * calls["evaluate"],
        }

    monkeypatch.setattr(
        filter_candidate_runner,
        "prepare_mnist_fgsm_adversarial_set",
        fake_prepare,
    )
    monkeypatch.setattr(
        filter_candidate_runner,
        "evaluate_filter_on_existing_adversarial",
        fake_evaluate,
    )
    monkeypatch.setattr(
        filter_candidate_runner,
        "close_graph",
        lambda graph: calls.__setitem__("close", calls["close"] + 1),
    )

    rows = filter_candidate_runner.run_filter_candidate_experiment(
        {
            "experiment_id": "candidate_test",
            "selection_stage": "validation",
            "filters": [
                {"name": "cross_3x3", "type": "cross_mean", "radius": 1},
                {"name": "box_5x5", "type": "box_mean", "kernel_size": 5},
            ],
            "evaluation": {"exclude_invalid_pairs": False, "batch_size": 8},
            "output": {
                "dir": str(tmp_path),
                "csv": "metrics.csv",
                "json": "metrics.json",
            },
        }
    )

    assert calls == {"context": 1, "evaluate": 2, "close": 1}
    assert [row["filter_name"] for row in rows] == ["cross_3x3", "box_5x5"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "filter_name,filter_type,mask_type,mask_size,radius,kernel_size,"
        "TP,FN,FP,recall_percent,precision_percent,f1_percent"
    )
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "candidate_test"
    assert payload["extra"]["num_filters"] == 2
    assert payload["extra"]["selection_stage"] == "validation"
    assert payload["extra"]["best_filter_by_f1"]["filter_name"] == "box_5x5"


def test_filter_candidate_runner_derives_quantization_csv_schema(
    monkeypatch,
    tmp_path,
) -> None:
    """Table 4 filter grids should emit interval columns without YAML field lists."""
    monkeypatch.setattr(
        filter_candidate_runner,
        "prepare_mnist_fgsm_adversarial_set",
        lambda config: _context(),
    )
    monkeypatch.setattr(
        filter_candidate_runner,
        "evaluate_filter_on_existing_adversarial",
        lambda **kwargs: {
            "TP": 1,
            "FN": 2,
            "FP": 3,
            "recall_percent": 25.0,
            "precision_percent": 50.0,
            "f1_percent": 33.33,
        },
    )
    monkeypatch.setattr(filter_candidate_runner, "close_graph", lambda graph: None)

    filter_candidate_runner.run_filter_candidate_experiment(
        {
            "experiment_id": "table_4",
            "filters": [
                {
                    "name": "scalar_quantization_2",
                    "type": "scalar_quantization",
                    "intervals": 2,
                }
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        (
            "filter_name,filter_type,intervals,interval_size,TP,FN,FP,"
            "recall_percent,precision_percent,f1_percent"
        ),
        "scalar_quantization_2,scalar_quantization,2,128,1,2,3,25.0,50.0,33.33",
    ]


def test_filter_candidate_runner_can_time_filters_and_hide_filter_name(
    monkeypatch,
    tmp_path,
) -> None:
    """Table 3 should include timing without the filter_name CSV column."""
    monkeypatch.setattr(
        filter_candidate_runner,
        "prepare_mnist_fgsm_adversarial_set",
        lambda config: _context(),
    )
    monkeypatch.setattr(
        filter_candidate_runner,
        "evaluate_filter_on_existing_adversarial",
        lambda **kwargs: {
            "TP": 1,
            "FN": 2,
            "FP": 3,
            "recall_percent": 25.0,
            "precision_percent": 50.0,
            "f1_percent": 33.33,
        },
    )
    monkeypatch.setattr(filter_candidate_runner, "time_filter_application", lambda fn, images: 0.125)
    monkeypatch.setattr(filter_candidate_runner, "close_graph", lambda graph: None)

    rows = filter_candidate_runner.run_filter_candidate_experiment(
        {
            "experiment_id": "table_3",
            "filters": [
                {
                    "name": "scalar_quantization_2",
                    "type": "scalar_quantization",
                    "intervals": 2,
                },
                {
                    "name": "nonuniform_quantization",
                    "type": "nonuniform_quantization",
                },
            ],
            "evaluation": {"include_filter_time": True},
            "output": {
                "dir": str(tmp_path),
                "csv": "metrics.csv",
                "json": "metrics.json",
                "include_filter_name": False,
            },
        }
    )

    assert [row["time_s"] for row in rows] == [0.125, 0.125]
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        (
            "filter_type,time_s,TP,FN,FP,recall_percent,"
            "precision_percent,f1_percent"
        ),
        "scalar_quantization,0.125,1,2,3,25.0,50.0,33.33",
        "nonuniform_quantization,0.125,1,2,3,25.0,50.0,33.33",
    ]


def test_filter_candidate_configs_define_expected_candidates() -> None:
    """Table 3, 4, 7, and 8 configs should declare candidates only in YAML."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table3 = config["experiments"]["table_3"]
    table4 = config["experiments"]["table_4"]
    table4_mnist = config["experiments"]["table_4_mnist"]
    table7 = config["experiments"]["table_7"]
    table8 = config["experiments"]["table_8"]

    assert table3["kind"] == "filter_grid"
    assert table4["kind"] == "composite"
    assert table4["components"] == ["table_4_mnist", "table_4_imagenet"]
    assert table4_mnist["kind"] == "filter_grid"
    assert table7["kind"] == "filter_grid"
    assert table8["kind"] == "filter_grid"
    assert len(table3["filters"]) == 2
    assert len(table4_mnist["filters"]) == 9
    assert len(table7["filters"]) == 12
    assert len(table8["filters"]) == 5
    assert table3["output_dir"] == "results/experiments/table_3"
    assert table4["output_dir"] == "results/experiments/table_4"
    assert table4_mnist["output_dir"] == "results/experiments/table_4/mnist"
    assert table7["output_dir"] == "results/experiments/table_7"
    assert table8["output_dir"] == "results/experiments/table_8"

    assert [row["type"] for row in table3["filters"]] == [
        "scalar_quantization",
        "nonuniform_quantization",
    ]
    assert table3["evaluation"]["include_filter_time"] is True
    assert table3["output"]["include_filter_name"] is False
    built_table3 = build_experiment_config("table_3", config)
    assert built_table3["output"]["include_filter_name"] is False
    assert [row["intervals"] for row in table4_mnist["filters"]] == list(range(2, 11))
    assert table7["filters"] == [
        "cross_3x3",
        "cross_5x5",
        "cross_7x7",
        "cross_9x9",
        "diamond_3x3",
        "diamond_5x5",
        "diamond_7x7",
        "diamond_9x9",
        "box_3x3",
        "box_5x5",
        "box_7x7",
        "box_9x9",
    ]
    assert table8["filters"] == [
        "cross_5x5",
        "cross_7x7",
        "diamond_5x5",
        "diamond_7x7",
        "box_5x5",
    ]


def test_table7_and_table8_per_table_scripts_were_removed() -> None:
    """The active Table 7/8 path should be the single experiment entry point."""
    assert not (PROJECT_ROOT / "scripts" / "experiments" / "table_7.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "experiments" / "table_8.py").exists()
    source = (PROJECT_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert "run_experiment" in source
    assert "evaluate_filter_on_existing_adversarial" not in source
    assert "markdown" not in source.lower()


def test_table3_and_table4_legacy_assets_were_removed() -> None:
    """Tables 3 and 4 should use only the consolidated experiment path."""
    for relative_path in (
        "scripts/article_reproduction/table_3.py",
        "scripts/article_reproduction/table_4.py",
        "scripts/article_reproduction/table_4_mnist.py",
        "configs/article_reproduction/mnist_table_3.yaml",
        "configs/article_reproduction/mnist_table_4.yaml",
        "results/mnist/article_reproduction/table_3_uniform_vs_nonuniform.csv",
        "results/mnist/article_reproduction/table_4_scalar_quantization_intervals.csv",
        "results/mnist/article_reproduction/table_6_adaptive_quantization.csv",
    ):
        assert not (PROJECT_ROOT / relative_path).exists()
