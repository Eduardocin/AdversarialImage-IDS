"""Run configured DeepDetector experiments."""

from __future__ import annotations

import argparse

from deepdetector.experiments.runner import run_experiment
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path


DEFAULT_CONFIG = resolve_project_path("configs/experiments.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main() -> int:
    """Load the consolidated config and run one experiment."""
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config) or DEFAULT_CONFIG
    if config_path is None:
        raise ValueError("An experiment config path is required.")
    run_experiment(args.experiment, load_yaml_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
