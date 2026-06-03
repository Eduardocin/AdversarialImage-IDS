"""Reproduce ImageNet Table 8 validation spatial smoothing filter metrics."""

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

from deepdetector.experiments.table7_imagenet_runner import (  # noqa: E402
    _output_dir,
    build_imagenet_table7_model,
    load_imagenet_table7_subset,
)
from deepdetector.experiments.table8_imagenet_runner import (  # noqa: E402
    _write_table8_status,
    run_table8_imagenet_experiment,
)
from deepdetector.io.config import load_yaml_config  # noqa: E402
from deepdetector.io.paths import ensure_dir  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_8.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--adv-path",
        default=None,
        help="Path to a pre-generated .npy array with adversarial images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config, model, and images, then stop before inference.",
    )
    return parser


def _apply_cli_overrides(
    config: Dict[str, Any],
    output_dir: str | None,
    adv_path: str | None,
) -> Dict[str, Any]:
    """Apply CLI overrides while preserving the YAML config shape."""
    updated = dict(config)
    output_config = dict(updated.get("output", updated.get("outputs", {})))
    if output_dir:
        output_config["dir"] = output_dir
    if output_config:
        updated["output"] = output_config

    attack_config = dict(updated.get("attack", {}))
    if adv_path:
        attack_config["adversarial_path"] = adv_path
    if attack_config:
        updated["attack"] = attack_config
    return updated


def _dry_run(config: Dict[str, Any]) -> int:
    """Validate model and image loading without running attack inference."""
    output_dir = ensure_dir(_output_dir(config))
    try:
        build_imagenet_table7_model(config)
    except ImportError as exc:
        _write_table8_status(
            config,
            output_dir,
            "parcial",
            limitation="bloqueado_caffe",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return 0
    except OSError as exc:
        _write_table8_status(
            config,
            output_dir,
            "parcial",
            limitation="bloqueado_modelo_googlenet",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return 0

    try:
        images, labels = load_imagenet_table7_subset(config)
    except IOError as exc:
        _write_table8_status(
            config,
            output_dir,
            "parcial",
            limitation="nenhuma_imagem_carregada",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return 0

    print("images_shape={0}".format(images.shape))
    print("labels_shape={0}".format(labels.shape))
    _write_table8_status(
        config,
        output_dir,
        "parcial",
        limitation="dry_run",
        n_loaded=int(len(images)),
        total_images=int(len(images)),
        clean_correct=0,
        skipped_wrong_baseline=0,
        attack_success=0,
        disturbed_failure=0,
    )
    return 0


def main() -> int:
    """Run ImageNet Table 8 and write pivot/status outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = _apply_cli_overrides(
        load_yaml_config(Path(args.config)),
        output_dir=args.output_dir,
        adv_path=args.adv_path,
    )

    result = _dry_run(config) if args.dry_run else run_table8_imagenet_experiment(config)
    if isinstance(result, dict):
        for key, value in result.items():
            print("{0}={1}".format(key, value))
        return 0
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
