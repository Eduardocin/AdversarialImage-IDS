"""Official ImageNet Table 8 validation spatial smoothing runner."""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from deepdetector.evaluation.table8 import evaluate_table8_filter
from deepdetector.experiments.table7_imagenet_runner import (
    _article_model_inputs,
    _epsilon_normalized,
    _output_dir,
    _remove_stale_standard_outputs,
    _write_status,
    adversarial_images_for_run,
    build_imagenet_table7_model,
    filter_clean_baseline_images,
    load_imagenet_table7_subset,
    write_pivot_csv,
)
from deepdetector.io.paths import ensure_dir


logger = logging.getLogger(__name__)

TABLE8_FILTERS: Tuple[Tuple[str, int], ...] = (
    ("cross", 5),
    ("cross", 7),
    ("diamond", 5),
    ("diamond", 7),
    ("box", 5),
)
TABLE8_COLUMNS: Tuple[str, ...] = (
    "cross_5x5",
    "cross_7x7",
    "diamond_5x5",
    "diamond_7x7",
    "box_5x5",
)


def configured_filters(config: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    """Yield the fixed Table 8 filters, validating config when it is present."""
    filter_config = config.get("filter", {})
    configured = filter_config.get("filters")
    if configured is not None:
        configured_filters_tuple = tuple(
            (str(item["mask_type"]), int(item["size"])) for item in configured
        )
        if configured_filters_tuple != TABLE8_FILTERS:
            raise ValueError(
                "Table 8 must evaluate exactly: {0}.".format(
                    ", ".join(
                        "{0}_{1}x{1}".format(mask, size)
                        for mask, size in TABLE8_FILTERS
                    )
                )
            )
    return TABLE8_FILTERS


def _status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Return filter-independent attack counters from Table 8 rows."""
    if not rows:
        return {"attack_success": 0, "disturbed_failure": 0}
    return {
        "attack_success": int(rows[0]["attack_success"]),
        "disturbed_failure": int(rows[0]["disturbed_failure"]),
    }


def _write_table8_status(
    config: Dict[str, Any],
    output_dir: Path,
    status: str,
    **fields: Any,
) -> Path:
    """Write the configured Table 8 status JSON."""
    return _write_status(config=config, output_dir=output_dir, status=status, **fields)


def run_table8_imagenet_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the official ImageNet Table 8 experiment and write pivot outputs."""
    output_dir = ensure_dir(_output_dir(config))
    _remove_stale_standard_outputs(output_dir)
    try:
        model = build_imagenet_table7_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="bloqueado_caffe",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return {"status": "parcial", "status_json": str(path)}
    except OSError as exc:
        logger.warning("%s", exc)
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="bloqueado_modelo_googlenet",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    try:
        images, labels = load_imagenet_table7_subset(config)
    except IOError as exc:
        logger.warning("%s", exc)
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            message=str(exc),
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    if len(images) == 0:
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            total_images=0,
            clean_correct=0,
            skipped_wrong_baseline=0,
            attack_success=0,
            disturbed_failure=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    images, labels, selected_indices, clean_summary = filter_clean_baseline_images(
        model=model,
        images=images,
        labels=labels,
    )
    if len(images) == 0:
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_limpa_correta",
            attack_success=0,
            disturbed_failure=0,
            **clean_summary,
        )
        return {"status": "parcial", "status_json": str(path)}

    images = _article_model_inputs(model, images)
    adv_images = adversarial_images_for_run(
        config=config,
        model=model,
        images=images,
        selected_indices=selected_indices,
    )
    if adv_images is None:
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            n_loaded=int(len(images)),
            attack_success=0,
            disturbed_failure=0,
            limitation="fgsm_caffe_requer_gradiente_ou_adversariais_salvas",
            message=(
                "GoogLeNetCaffeWrapper must expose Caffe gradient support. "
                "Configure attack.adversarial_path with a compatible .npy file."
            ),
            **clean_summary,
        )
        return {"status": "parcial", "status_json": str(path)}

    dataset = (images, labels, adv_images)
    rows: List[Dict[str, Any]] = []
    for mask_type, size in configured_filters(config):
        logger.info("Evaluating Table 8 ImageNet mask=%s size=%s", mask_type, size)
        result = evaluate_table8_filter(
            model=model,
            dataset=dataset,
            mask_type=mask_type,
            size=size,
        )
        row = asdict(result)
        row["skipped_wrong_baseline"] = clean_summary["skipped_wrong_baseline"]
        rows.append(row)

    status_counts = _status_counts(rows)
    if status_counts["attack_success"] == 0:
        path = _write_table8_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhum_adversarial_bem_sucedido",
            **clean_summary,
            **status_counts,
        )
        return {"status": "parcial", "status_json": str(path)}

    output_config = config.get("output", {})
    pivot_path = write_pivot_csv(
        output_dir / str(output_config.get("pivot_csv", "table_8_imagenet.csv")),
        rows,
        columns=TABLE8_COLUMNS,
    )
    status_path = _write_table8_status(
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
