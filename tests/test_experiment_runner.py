from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments import runner as experiment_runner  # noqa: E402
from scripts import run_experiment as run_experiment_script  # noqa: E402


def _consolidated_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "experiments.yaml").read_text())


def test_consolidated_config_contains_defaults_and_tables() -> None:
    """The active experiment config should be consolidated in one YAML."""
    config = _consolidated_config()

    assert "defaults" in config
    assert set(config["experiments"]) == {
        "table_3",
        "table_4",
        "table_4_mnist",
        "table_4_imagenet",
        "table_10_caffenet",
        "table_10_googlenet",
        "table_10_inception_v3",
        "table_10_m1",
        "table_10_m2",
        "table_6",
        "table_7",
        "table_8",
        "table_9",
    }
    assert config["defaults"] == {
        "output": {"csv": "metrics.csv", "json": "metrics.json"}
    }
    for experiment in config["experiments"].values():
        if experiment["kind"] == "composite":
            continue
        assert "dataset" in experiment
        assert "model" in experiment
        assert "attack" in experiment


@pytest.mark.parametrize(
    ("experiment_name", "kind"),
    [
        ("table_6", "split_eval"),
        ("table_7", "filter_grid"),
        ("table_8", "filter_grid"),
        ("table_9", "split_eval"),
        ("table_3", "filter_grid"),
        ("table_4", "composite"),
        ("table_4_mnist", "filter_grid"),
        ("table_4_imagenet", "imagenet_table_4"),
        ("table_10_m1", "table_10_group"),
        ("table_10_googlenet", "table_10_group"),
        ("table_10_caffenet", "table_10_group"),
        ("table_10_m2", "table_10_group"),
        ("table_10_inception_v3", "table_10_group"),
    ],
)
def test_run_experiment_entrypoint_resolves_requested_experiment(
    monkeypatch,
    experiment_name,
    kind,
) -> None:
    """scripts/run_experiment.py should select the requested config entry."""
    calls = []

    def fake_run_experiment(name, config):
        calls.append((name, config["experiments"][name]["kind"]))
        return []

    monkeypatch.setattr(run_experiment_script, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--experiment", experiment_name],
    )

    assert run_experiment_script.main() == 0
    assert calls == [(experiment_name, kind)]


def test_build_experiment_config_rejects_unknown_experiment() -> None:
    """Unknown experiments should fail with a clear error."""
    with pytest.raises(ValueError, match="Unknown experiment"):
        experiment_runner.build_experiment_config("missing", _consolidated_config())


def test_run_experiment_rejects_unknown_kind(monkeypatch) -> None:
    """Unknown experiment kinds should fail before running any workflow."""
    config = _consolidated_config()
    config["experiments"]["table_6"]["kind"] = "mystery"

    with pytest.raises(ValueError, match="Unknown experiment kind"):
        experiment_runner.run_experiment("table_6", config)


def test_table_4_composite_runs_components_in_order(monkeypatch, tmp_path) -> None:
    """The article Table 4 entry should run MNIST and ImageNet under one root."""
    config = _consolidated_config()
    config["experiments"]["table_4"]["output_dir"] = str(tmp_path)
    config["experiments"]["table_4_mnist"]["output_dir"] = str(tmp_path / "mnist")
    config["experiments"]["table_4_imagenet"]["output_dir"] = str(tmp_path / "imagenet")
    calls = []

    def fake_filter_grid(component_config):
        calls.append(component_config["experiment_id"])
        return [{"row": 1}]

    def fake_imagenet(component_config):
        calls.append(component_config["experiment_id"])
        return {
            "status": "completo",
            "csv": str(tmp_path / "imagenet" / "table_4_imagenet.csv"),
        }

    monkeypatch.setattr(
        experiment_runner,
        "run_filter_candidate_experiment",
        fake_filter_grid,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_table4_imagenet_experiment",
        fake_imagenet,
    )

    result = experiment_runner.run_experiment("table_4", config)

    assert calls == ["table_4_mnist", "table_4_imagenet"]
    assert [entry["experiment_id"] for entry in result] == [
        "table_4_mnist",
        "table_4_imagenet",
    ]
    manifest = tmp_path / "manifest.json"
    assert manifest.exists()


def test_runtime_entrypoints_do_not_contain_forbidden_report_strings() -> None:
    """The active runtime should not generate Markdown, diagnostics, or comparisons."""
    forbidden = (
        "write_markdown_table",
        "ARTICLE_TABLE_",
        "Delta F1",
        "comparison_md",
        "results_md",
        "diagnostic",
        "diagnostico",
        "summary.md",
    )
    runtime_paths = [
        PROJECT_ROOT / "scripts" / "run_experiment.py",
        PROJECT_ROOT / "src" / "deepdetector" / "experiments" / "runner.py",
        PROJECT_ROOT / "src" / "deepdetector" / "experiments" / "fgsm_split_runner.py",
        PROJECT_ROOT / "src" / "deepdetector" / "experiments" / "filter_candidate_runner.py",
    ]

    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        for text in forbidden:
            assert text not in source


def test_results_tree_has_no_markdown_outputs() -> None:
    """Unified experiment outputs should not include Markdown reports."""
    experiments_dir = PROJECT_ROOT / "results" / "experiments"
    if experiments_dir.exists():
        assert list(experiments_dir.glob("**/*.md")) == []
