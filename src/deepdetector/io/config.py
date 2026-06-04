"""Configuration helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from deepdetector.io.paths import resolve_project_path


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config and ensure the root object is a mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        raise ValueError("Config file is empty or invalid: {0}".format(config_path))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping: {0}".format(config_path))
    return config


def get_config_section(
    config: Dict[str, Any],
    section: str,
    default: Optional[Any] = None,
) -> Any:
    """Return a config section with an optional default."""
    return config.get(section, default)


def output_dir_from_config(
    config: Dict[str, Any],
    override: Optional[str],
    project_root: Path,
    default_output_dir: Path,
) -> Path:
    """Resolve an experiment output directory from CLI, YAML, or default."""
    outputs = config.get("outputs", config.get("output", {}))
    configured = override or outputs.get("results_dir") or outputs.get("dir")
    return resolve_project_path(configured, project_root=project_root) or default_output_dir

