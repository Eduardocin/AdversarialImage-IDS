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

from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from deepdetector.experiments.table10_runner import (  # noqa: E402
    TABLE10_SCHEMA,
    run_table10_group_experiment,
)
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
        assert experiment["kind"] == "table_10_group"
        assert experiment["model_group"] == model_group
        assert experiment["dataset_label"] == dataset_label
        assert experiment["output_dir"] == "results/experiments/table_10/{0}".format(
            model_group
        )
        assert [row["no"] for row in experiment["rows"]] == row_numbers


def test_table10_runner_writes_official_schema_and_manifest(tmp_path) -> None:
    """A Table 10 group should write metrics CSV/JSON plus a manifest."""
    rows = run_table10_group_experiment(
        {
            "experiment_id": "table_10_test",
            "kind": "table_10_group",
            "model_group": "googlenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 5,
                    "attack_model": "FGSM (epsilon=1/255)/GoogLeNet",
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

    assert rows["status"] == "completed"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "manifest.json",
        "metrics.csv",
        "metrics.json",
    ]
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    assert csv_rows[0] == list(TABLE10_SCHEMA)
    assert csv_rows[1] == [
        "5",
        "FGSM (epsilon=1/255)/GoogLeNet",
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
    assert metrics_payload["schema"] == list(TABLE10_SCHEMA)
    assert metrics_payload["metrics"][0]["num_failures"] is None
    assert metrics_payload["metrics"][0]["no"] == 5

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["table"] == 10
    assert manifest["model_group"] == "googlenet"
    assert manifest["dataset"] == "ImageNet"
    assert manifest["rows"] == [5, 7]
    assert manifest["outputs"]["metrics_csv"] == str(tmp_path / "metrics.csv")
    assert manifest["outputs"]["metrics_json"] == str(tmp_path / "metrics.json")


def test_table10_blocked_reasons_stay_in_manifest_only(tmp_path) -> None:
    """Blocked groups should not leak blocked reasons into metrics outputs."""
    run_table10_group_experiment(
        {
            "experiment_id": "table_10_caffenet",
            "kind": "table_10_group",
            "model_group": "caffenet",
            "dataset_label": "ImageNet",
            "rows": [
                {
                    "no": 8,
                    "attack_model": "DeepFool/CaffeNet",
                    "status": "blocked",
                    "blocked_reason": "CaffeNet ainda não está implementado.",
                }
            ],
            "output": {"dir": str(tmp_path), "csv": "metrics.csv", "json": "metrics.json"},
        }
    )

    assert "CaffeNet ainda" not in (tmp_path / "metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "CaffeNet ainda" not in (tmp_path / "metrics.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["blocked_reason"] == "CaffeNet ainda não está implementado."


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
