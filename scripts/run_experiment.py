"""Run configured DeepDetector experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments.runner import run_experiment
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path


DEFAULT_CONFIG = resolve_project_path("configs/experiments.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override an experiment field with dotted syntax, e.g. evaluation.n_samples=5.",
    )
    return parser


def _parse_override(value: str) -> tuple[list[str], Any]:
    if "=" not in value:
        raise ValueError("Override must use KEY=VALUE syntax: {0}".format(value))
    key, raw_value = value.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Override key cannot be empty.")
    return parts, yaml.safe_load(raw_value)


def _apply_experiment_overrides(
    config: dict[str, Any],
    experiment_name: str,
    overrides: list[str],
) -> None:
    if not overrides:
        return
    experiments = config.setdefault("experiments", {})
    if experiment_name not in experiments:
        raise ValueError("Unknown experiment: {0}".format(experiment_name))
    experiment = experiments[experiment_name]
    for override in overrides:
        parts, value = _parse_override(override)
        target = experiment
        for part in parts[:-1]:
            next_target = target.setdefault(part, {})
            if not isinstance(next_target, dict):
                raise ValueError("Override path is not a mapping: {0}".format(override))
            target = next_target
        target[parts[-1]] = value


def main() -> int:
    """Load the consolidated config and run one experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config) or DEFAULT_CONFIG
    if config_path is None:
        raise ValueError("An experiment config path is required.")
    try:
        logging.info("Starting experiment: %s", args.experiment)
        config = load_yaml_config(config_path)
        _apply_experiment_overrides(config, args.experiment, args.override)
        run_experiment(args.experiment, config)
        logging.info("Finished experiment: %s", args.experiment)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
