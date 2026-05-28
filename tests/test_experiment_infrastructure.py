import json
from pathlib import Path
import re
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments.metadata import build_experiment_payload  # noqa: E402
from deepdetector.io.config import get_config_section, load_yaml_config  # noqa: E402
from deepdetector.io.paths import (  # noqa: E402
    ensure_dir,
    get_project_root,
    resolve_project_path,
)
from deepdetector.io.result_writers import (  # noqa: E402
    write_experiment_outputs,
    write_metrics_csv,
    write_metrics_json,
)


def test_path_helpers_resolve_project_paths_and_create_dirs(tmp_path) -> None:
    """Path helpers should centralize project-relative resolution."""
    assert get_project_root() == PROJECT_ROOT
    assert resolve_project_path(None) is None

    absolute = tmp_path / "already_absolute"
    assert resolve_project_path(absolute) == absolute
    assert resolve_project_path("configs/experiments.yaml") == (
        PROJECT_ROOT / "configs" / "experiments.yaml"
    )

    created = ensure_dir(tmp_path / "nested" / "outputs")
    assert created.is_dir()


def test_load_yaml_config_requires_mapping_root(tmp_path) -> None:
    """YAML configs should fail clearly when empty or not mappings."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("section:\n  value: 1\n", encoding="utf-8")

    config = load_yaml_config(config_path)
    assert config == {"section": {"value": 1}}
    assert get_config_section(config, "section") == {"value": 1}
    assert get_config_section(config, "missing", {"fallback": True}) == {
        "fallback": True
    }

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(empty_path))):
        load_yaml_config(empty_path)

    list_path = tmp_path / "list.yaml"
    list_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(list_path))):
        load_yaml_config(list_path)


def test_result_writers_emit_stable_csv_and_readable_json(tmp_path) -> None:
    """Result writers should preserve CSV field order and write readable JSON."""
    rows = [{"b": 2, "a": 1, "ignored": 3}]

    csv_path = write_metrics_csv(tmp_path / "metrics.csv", rows, ["a", "b"])
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]

    payload = {"experiment_id": "example", "metrics": rows}
    json_path = write_metrics_json(tmp_path / "metrics.json", payload)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert '\n  "experiment_id": "example"' in json_path.read_text(encoding="utf-8")


def test_write_experiment_outputs_defaults_to_metrics_csv_and_json(tmp_path) -> None:
    """The standard experiment writer should not emit Markdown."""
    rows = [{"split": "Validation", "TP": 1}]
    payload = build_experiment_payload(
        experiment_id="mnist_table_6_adaptive_quantization",
        config={"attack": {"epsilon": 0.2}},
        rows=rows,
    )

    outputs = write_experiment_outputs(
        output_dir=tmp_path,
        rows=rows,
        csv_fields=["split", "TP"],
        metadata=payload,
    )

    assert outputs == {
        "csv": tmp_path / "metrics.csv",
        "json": tmp_path / "metrics.json",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "metrics.csv",
        "metrics.json",
    ]
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == {
        "experiment_id": "mnist_table_6_adaptive_quantization",
        "config": {"attack": {"epsilon": 0.2}},
        "metrics": rows,
        "extra": {},
    }


def test_table6_legacy_script_was_removed() -> None:
    """The representative old script should not remain as a runtime path."""
    assert not (PROJECT_ROOT / "scripts/article_reproduction/table_6.py").exists()
