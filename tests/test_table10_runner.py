import csv
import json
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.tables.table_10 import (  # noqa: E402
    TABLE_10_SCHEMA,
    build_pending_table_10_row,
    run_table_10_group,
)
from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from scripts import run_experiment as run_experiment_script  # noqa: E402


TABLE10_EXPERIMENTS = {
    "table_10_m1": ("m1", "MNIST", [1, 2, 3, 4]),
    "table_10_googlenet": ("googlenet", "ImageNet", [5, 6, 7]),
    "table_10_caffenet": ("caffenet", "ImageNet", [8]),
    "table_10_m2": ("m2", "MNIST", [9, 10, 11, 12, 13, 19]),
    "table_10_inception_v3": (
        "inception_v3",
        "ImageNet",
        [14, 15, 16, 17, 18, 20],
    ),
}


def _consolidated_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )


def test_table10_config_declares_one_experiment_per_model_group() -> None:
    """Table 10 should be executable independently by model group."""
    config = _consolidated_config()
    experiments = config["experiments"]

    for experiment_name, (model_group, dataset_label, row_numbers) in TABLE10_EXPERIMENTS.items():
        experiment = experiments[experiment_name]
        expected_output_dir = "results/experiments/table_10/{0}".format(model_group)
        if experiment_name == "table_10_googlenet":
            expected_output_dir = "results/experiments/table_10/imagenet/googlenet"

        assert experiment["kind"] == "table_10_group"
        assert experiment["model_group"] == model_group
        assert experiment["dataset_label"] == dataset_label
        assert experiment["output_dir"] == expected_output_dir
        assert [row["no"] for row in experiment["rows"]] == row_numbers


def test_table_10_schema_matches_paper_fields() -> None:
    assert TABLE_10_SCHEMA == [
        "no",
        "attack_model",
        "dataset",
        "num_failures",
        "tp",
        "fn",
        "fp",
        "rtp",
        "rtp_percent",
        "recall",
        "precision",
        "f1",
    ]


def test_build_pending_table_10_row() -> None:
    row = build_pending_table_10_row(
        no=5,
        attack_model="FGSM (\u03b5=1/255)/GoogLeNet",
        dataset="ImageNet",
    )

    assert row == {
        "no": 5,
        "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
        "dataset": "ImageNet",
        "num_failures": None,
        "tp": None,
        "fn": None,
        "fp": None,
        "rtp": None,
        "rtp_percent": None,
        "recall": None,
        "precision": None,
        "f1": None,
    }


def test_table10_runner_writes_official_schema_without_manifest(tmp_path) -> None:
    """A Table 10 group should write only metrics CSV and JSON."""
    rows = run_table_10_group(
        {
            "experiment_id": "table_10_test",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "googlenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 5,
                    "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
                    "status": "planned",
                },
                {
                    "no": 7,
                    "attack_model": "DeepFool/GoogLeNet",
                    "status": "planned",
                },
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert [row["no"] for row in rows] == [5, 7]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.csv", "metrics.json"]
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    assert csv_rows[0] == TABLE_10_SCHEMA
    assert csv_rows[1] == [
        "5",
        "FGSM (\u03b5=1/255)/GoogLeNet",
        "ImageNet",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["table"] == 10
    assert metrics_payload["dataset_group"] == "imagenet"
    assert metrics_payload["model_group"] == "googlenet"
    assert metrics_payload["rows"][0]["num_failures"] is None
    assert metrics_payload["rows"][0]["no"] == 5


def test_table10_blocked_reasons_stay_out_of_outputs(tmp_path) -> None:
    """Blocked reasons should not leak into official metrics outputs."""
    run_table_10_group(
        {
            "experiment_id": "table_10_caffenet",
            "kind": "table_10_group",
            "dataset": {"name": "imagenet"},
            "model_group": "caffenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 8,
                    "attack_model": "DeepFool/CaffeNet",
                    "status": "blocked",
                    "blocked_reason": "CaffeNet is not implemented.",
                }
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert "CaffeNet is not implemented" not in (tmp_path / "metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "CaffeNet is not implemented" not in (tmp_path / "metrics.json").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "manifest.json").exists()


def test_table_10_googlenet_group_generates_three_rows(tmp_path) -> None:
    config = {
        "kind": "table_10_group",
        "output_dir": str(tmp_path),
        "dataset": {"name": "imagenet"},
        "model": {"name": "googlenet"},
        "model_group": "googlenet",
        "dataset_label": "ImageNet",
        "rows": [
            {
                "no": 5,
                "attack_model": "FGSM (\u03b5=1/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 1 / 255},
            },
            {
                "no": 6,
                "attack_model": "FGSM (\u03b5=2/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 2 / 255},
            },
            {
                "no": 7,
                "attack_model": "DeepFool/GoogLeNet",
                "status": "planned",
                "attack": {"name": "deepfool"},
            },
        ],
    }

    rows = run_table_10_group(config)

    assert [row["no"] for row in rows] == [5, 6, 7]
    assert all(row["dataset"] == "ImageNet" for row in rows)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_run_experiment_dispatches_table10_group(monkeypatch) -> None:
    """The official runner should dispatch Table 10 groups through their runner."""
    config = _consolidated_config()
    calls = []

    def fake_table10_runner(component_config):
        calls.append(
            (
                component_config["experiment_id"],
                component_config["kind"],
                component_config["model_group"],
            )
        )
        return {"status": "completed"}

    monkeypatch.setattr(
        experiment_runner,
        "run_table10_group_experiment",
        fake_table10_runner,
    )

    result = experiment_runner.run_experiment("table_10_m2", config)

    assert result == {"status": "completed"}
    assert calls == [("table_10_m2", "table_10_group", "m2")]


@pytest.mark.parametrize("experiment_name", sorted(TABLE10_EXPERIMENTS))
def test_run_experiment_cli_accepts_table10_groups(monkeypatch, experiment_name) -> None:
    """scripts/run_experiment.py should accept every Table 10 model-group command."""
    calls = []

    def fake_run_experiment(name, config):
        calls.append((name, config["experiments"][name]["kind"]))
        return {}

    monkeypatch.setattr(run_experiment_script, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--experiment", experiment_name],
    )

    assert run_experiment_script.main() == 0
    assert calls == [(experiment_name, "table_10_group")]
