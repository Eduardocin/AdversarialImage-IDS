"""Reproduce ImageNet Table 8 validation spatial smoothing filter metrics."""

from __future__ import annotations

import logging
from pathlib import Path

from deepdetector.experiments.article_cli import (
    apply_article_cli_overrides,
    build_imagenet_article_parser,
    find_project_root,
    print_result,
    run_imagenet_article_dry_run,
)
from deepdetector.experiments.table8_imagenet_runner import run_table8_imagenet_experiment
from deepdetector.io.config import load_yaml_config


PROJECT_ROOT = find_project_root(Path(__file__))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_8.yaml"


def main() -> int:
    """Run ImageNet Table 8 and write pivot/status outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_imagenet_article_parser(description=__doc__, default_config=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = apply_article_cli_overrides(
        load_yaml_config(Path(args.config)),
        output_dir=args.output_dir,
        adv_path=args.adv_path,
    )
    result = (
        run_imagenet_article_dry_run(config, table_name="table_8")
        if args.dry_run
        else run_table8_imagenet_experiment(config)
    )
    return print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
