"""Run configured DeepDetector experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


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
    return parser


def main() -> int:
    """Load the consolidated config and run one experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config) or DEFAULT_CONFIG
    if config_path is None:
        raise ValueError("An experiment config path is required.")
    try:
        logging.info("Starting experiment: %s", args.experiment)
        run_experiment(args.experiment, load_yaml_config(config_path))
        logging.info("Finished experiment: %s", args.experiment)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
