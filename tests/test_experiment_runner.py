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
    assert set(config["experiments"]) == {"table_6", "table_7", "table_8", "table_9"}
    assert config["defaults"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert config["defaults"]["epsilon"] == 0.2


@pytest.mark.parametrize(
    ("experiment_name", "kind"),
    [
        ("table_6", "split_eval"),
        ("table_7", "filter_grid"),
        ("table_8", "filter_grid"),
        ("table_9", "split_eval"),
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
    """Versioned experiment outputs should not include Markdown reports."""
    assert list((PROJECT_ROOT / "results").glob("**/*.md")) == []
