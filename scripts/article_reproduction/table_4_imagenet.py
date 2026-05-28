"""Deprecated wrapper for the ImageNet half of Table 4.

Prefer:
    python scripts/run_experiment.py --experiment table_4_imagenet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.experiments.table4_imagenet_runner import (  # noqa: E402
    build_model,
    configured_intervals,
    epsilon_255_from_config,
    load_subset_samples,
    run_table4_imagenet_experiment,
    write_status,
)
from deepdetector.io.config import load_yaml_config  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_EXPERIMENT = "table_4_imagenet"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory containing ImageNet class folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to evaluate for a quick run.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="FGSM epsilon in 0-255 Caffe scale. Defaults to config or 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for table_4_imagenet.csv and status JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config, model, and images, then stop before inference.",
    )
    return parser


def load_config(path: Path) -> Dict[str, Any]:
    """Load an ImageNet Table 4 config from either YAML shape."""
    config = load_yaml_config(path)
    if "experiments" in config:
        experiment = str(DEFAULT_EXPERIMENT)
        experiment_config = dict(config["experiments"][experiment])
        experiment_config["experiment_id"] = experiment
        experiment_config["output"] = {
            "dir": experiment_config.get("output_dir"),
            **dict(experiment_config.get("output", {})),
        }
        return experiment_config
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    output_config = dict(config.get("output", {}))
    if "results_dir" in output_config and "dir" not in output_config:
        output_config["dir"] = output_config["results_dir"]
    config["output"] = output_config
    return config


def main() -> int:
    """Run ImageNet Table 4 through the shared experiment module."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config_file = Path(args.config)
    config = load_config(config_file)
    if config.get("experiment_id") != args.experiment and config_file == DEFAULT_CONFIG:
        full_config = load_yaml_config(config_file)
        config = dict(full_config["experiments"][args.experiment])
        config["experiment_id"] = args.experiment
        config["output"] = {
            "dir": config.get("output_dir"),
            **dict(config.get("output", {})),
        }

    try:
        result = run_table4_imagenet_experiment(
            config,
            data_root_override=args.data_root,
            limit_override=args.limit,
            epsilon_override=args.epsilon,
            output_dir_override=args.output_dir,
            dry_run=bool(args.dry_run),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for key in ("csv", "status_json"):
        if key in result:
            print("{0}={1}".format(key, result[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
