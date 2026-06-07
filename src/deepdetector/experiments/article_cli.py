"""Shared CLI helpers for article reproduction scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from deepdetector.experiments.imagenet_common_runner import (
    build_imagenet_caffe_model,
    load_imagenet_subset,
    output_dir_from_config,
    write_imagenet_status,
)
from deepdetector.io.paths import ensure_dir


def find_project_root(start: Path) -> Path:
    """Return the nearest parent directory containing `pyproject.toml`."""
    resolved = start.resolve()
    candidates = resolved.parents if resolved.is_file() else (resolved, *resolved.parents)
    for parent in candidates:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not find project root from {0}.".format(start))


def build_imagenet_article_parser(
    description: Optional[str],
    default_config: Path,
) -> argparse.ArgumentParser:
    """Build common ImageNet article reproduction arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(default_config))
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


def apply_article_cli_overrides(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    adv_path: Optional[str] = None,
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


def _status_defaults(table_name: str) -> Dict[str, Any]:
    """Return status defaults required by a table-specific dry-run."""
    if table_name == "table_8":
        return {
            "default_name": "table_8_status.json",
            "blocked_status": "parcial",
            "base_fields": {
                "total_images": 0,
                "clean_correct": 0,
                "skipped_wrong_baseline": 0,
                "attack_success": 0,
                "disturbed_failure": 0,
            },
        }
    if table_name == "table_7":
        return {
            "default_name": "table_7_status.json",
            "blocked_status": None,
            "base_fields": {},
        }
    raise ValueError("Unsupported ImageNet article table: {0}".format(table_name))


def _write_dry_run_status(
    config: Dict[str, Any],
    table_name: str,
    status: str,
    **fields: Any,
) -> Path:
    """Write a table-specific dry-run status JSON."""
    output_dir = ensure_dir(output_dir_from_config(config))
    defaults = _status_defaults(table_name)
    payload = dict(defaults["base_fields"])
    payload.update(fields)
    return write_imagenet_status(
        config=config,
        output_dir=output_dir,
        status=status,
        default_name=str(defaults["default_name"]),
        **payload,
    )


def run_imagenet_article_dry_run(
    config: Dict[str, Any],
    table_name: str,
) -> Dict[str, Any]:
    """Validate ImageNet model and image loading without attack inference."""
    defaults = _status_defaults(table_name)
    try:
        build_imagenet_caffe_model(config)
    except ImportError as exc:
        if defaults["blocked_status"] == "parcial":
            path = _write_dry_run_status(
                config,
                table_name,
                "parcial",
                limitation="bloqueado_caffe",
                message=str(exc),
            )
            return {"status": "parcial", "status_json": str(path)}
        path = _write_dry_run_status(config, table_name, "bloqueado_caffe", message=str(exc))
        return {"status": "bloqueado_caffe", "status_json": str(path)}
    except OSError as exc:
        if defaults["blocked_status"] == "parcial":
            path = _write_dry_run_status(
                config,
                table_name,
                "parcial",
                limitation="bloqueado_modelo_googlenet",
                message=str(exc),
            )
            return {"status": "parcial", "status_json": str(path)}
        path = _write_dry_run_status(
            config,
            table_name,
            "bloqueado_modelo_googlenet",
            message=str(exc),
        )
        return {"status": "bloqueado_modelo_googlenet", "status_json": str(path)}

    try:
        images, labels = load_imagenet_subset(config)
    except IOError as exc:
        if table_name == "table_8":
            path = _write_dry_run_status(
                config,
                table_name,
                "parcial",
                limitation="nenhuma_imagem_carregada",
                message=str(exc),
            )
        else:
            path = _write_dry_run_status(
                config,
                table_name,
                "parcial",
                limitation="nenhuma_imagem_carregada",
                message=str(exc),
                n_loaded=0,
            )
        return {"status": "parcial", "status_json": str(path)}

    if table_name == "table_8":
        path = _write_dry_run_status(
            config,
            table_name,
            "parcial",
            limitation="dry_run",
            n_loaded=int(len(images)),
            total_images=int(len(images)),
        )
    else:
        path = _write_dry_run_status(
            config,
            table_name,
            "parcial",
            limitation="dry_run",
            n_loaded=int(len(images)),
        )
    return {
        "status": "parcial",
        "limitation": "dry_run",
        "images_shape": tuple(images.shape),
        "labels_shape": tuple(labels.shape),
        "status_json": str(path),
    }


def print_result(result: Any) -> int:
    """Print a script result dictionary and return a process exit code."""
    if isinstance(result, dict):
        for key, value in result.items():
            print("{0}={1}".format(key, value))
        return 0
    return int(result)
