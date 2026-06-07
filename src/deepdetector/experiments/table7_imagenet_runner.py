"""Official ImageNet Table 7 spatial smoothing runner."""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from deepdetector.evaluation.imagenet_utils import epsilon_normalized
from deepdetector.evaluation.table7 import evaluate_table7_filter
from deepdetector.experiments.imagenet_common_runner import (
    build_imagenet_caffe_model,
    load_imagenet_subset,
    output_dir_from_config,
    prepare_clean_imagenet_baseline,
    prepare_imagenet_adversarial_images,
    remove_stale_standard_outputs,
    write_imagenet_pivot_csv,
    write_imagenet_status,
)
from deepdetector.io.paths import ensure_dir


logger = logging.getLogger(__name__)

DEFAULT_PIVOT_CSV = "table_7_imagenet.csv"
TABLE7_STATUS_JSON = "table_7_status.json"

PIVOT_COLUMNS: Tuple[str, ...] = (
    "cross_3x3",
    "cross_5x5",
    "cross_7x7",
    "cross_9x9",
    "diamond_3x3",
    "diamond_5x5",
    "diamond_7x7",
    "diamond_9x9",
    "box_3x3",
    "box_5x5",
    "box_7x7",
    "box_9x9",
)


def configured_masks(config: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    """Yield configured Table 7 mask and size combinations."""
    filter_config = config.get("filter", {})
    mask_types = filter_config.get("mask_types", ["cross", "diamond", "box"])
    sizes = filter_config.get("sizes", [3, 5, 7, 9])
    for mask_type in mask_types:
        for size in sizes:
            yield str(mask_type), int(size)


def _status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Return filter-independent Table 7 sample-selection counters."""
    if not rows:
        return {
            "attack_success": 0,
            "disturbed_failure": 0,
            "skipped_low_entropy_clean": 0,
            "n_high_entropy_clean": 0,
            "n_high_entropy_adversarial": 0,
        }
    first_row = rows[0]
    return {
        "attack_success": int(first_row["attack_success"]),
        "disturbed_failure": int(first_row["disturbed_failure"]),
        "skipped_low_entropy_clean": int(first_row["skipped_low_entropy_clean"]),
        "n_high_entropy_clean": int(first_row["n_high_entropy_clean"]),
        "n_high_entropy_adversarial": int(first_row["n_high_entropy_adversarial"]),
    }


def _pivot_name_from_config(config: Dict[str, Any]) -> str:
    """Return the corrected Table 7 pivot filename."""
    output_config = config.get("output", config.get("outputs", {}))
    return str(output_config.get("pivot_csv", DEFAULT_PIVOT_CSV))


def _write_table7_status(
    config: Dict[str, Any],
    output_dir: Path,
    status: str,
    **fields: Any,
) -> Path:
    """Write the configured Table 7 status JSON."""
    return write_imagenet_status(
        config=config,
        output_dir=output_dir,
        status=status,
        default_name=TABLE7_STATUS_JSON,
        **fields,
    )


def run_table7_imagenet_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the official ImageNet Table 7 experiment and write pivot outputs."""
    output_dir = ensure_dir(output_dir_from_config(config))
    remove_stale_standard_outputs(output_dir)
    try:
        model = build_imagenet_caffe_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="bloqueado_caffe",
            message=str(exc),
        )
        return {"status": "bloqueado_caffe", "status_json": str(path)}
    except OSError as exc:
        logger.warning("%s", exc)
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="bloqueado_modelo_googlenet",
            message=str(exc),
        )
        return {"status": "bloqueado_modelo_googlenet", "status_json": str(path)}

    try:
        images, labels = load_imagenet_subset(config)
    except IOError as exc:
        logger.warning("%s", exc)
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            message=str(exc),
            n_loaded=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    if len(images) == 0:
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            n_loaded=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    images, labels, selected_indices, clean_summary = prepare_clean_imagenet_baseline(
        model=model,
        images=images,
        labels=labels,
    )
    if len(images) == 0:
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_limpa_correta",
            **clean_summary,
        )
        return {"status": "parcial", "status_json": str(path)}

    images, adv_images = prepare_imagenet_adversarial_images(
        config=config,
        model=model,
        images=images,
        selected_indices=selected_indices,
    )
    if adv_images is None:
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            n_loaded=int(len(images)),
            limitation="fgsm_caffe_requer_gradiente_ou_adversariais_salvas",
            message=(
                "GoogLeNetCaffeWrapper must expose Caffe gradient support. "
                "Configure attack.adversarial_path with a compatible .npy file."
            ),
            **clean_summary,
        )
        return {"status": "parcial", "status_json": str(path)}

    dataset = (images, labels, adv_images)
    rows = []
    for mask_type, size in configured_masks(config):
        logger.info("Evaluating Table 7 ImageNet mask=%s size=%s", mask_type, size)
        result = evaluate_table7_filter(
            model=model,
            dataset=dataset,
            mask_type=mask_type,
            size=size,
            epsilon=epsilon_normalized(config),
            entropy_threshold=float(config.get("filter", {}).get("entropy_threshold", 5.0)),
        )
        row = asdict(result)
        row["total_images"] = clean_summary["total_images"]
        row["clean_correct"] = clean_summary["clean_correct"]
        row["skipped_wrong_baseline"] = clean_summary["skipped_wrong_baseline"]
        rows.append(row)

    status_counts = _status_counts(rows)
    if status_counts["n_high_entropy_clean"] == 0:
        path = _write_table7_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhum_adversarial_high_entropy_bem_sucedido",
            n_loaded=int(len(images)),
            **clean_summary,
            **status_counts,
        )
        return {"status": "parcial", "status_json": str(path)}

    pivot_path = write_imagenet_pivot_csv(
        output_dir / _pivot_name_from_config(config),
        rows,
        columns=PIVOT_COLUMNS,
    )
    status_path = _write_table7_status(
        output_dir=output_dir,
        config=config,
        status="completo",
        n_loaded=int(len(images)),
        pivot_csv=str(pivot_path),
        **clean_summary,
        **status_counts,
    )
    return {
        "status": "completo",
        "pivot_csv": str(pivot_path),
        "status_json": str(status_path),
    }
