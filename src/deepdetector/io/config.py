"""Configuration helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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

