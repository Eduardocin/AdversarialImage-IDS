"""Official ImageNet Table 7 spatial smoothing runner."""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.adversarial_loader import adversarial_images_for_run
from deepdetector.data.imagenet import (
    class_image_rows,
    read_rgb_image,
    resize_normalized_image,
)
from deepdetector.evaluation.imagenet_utils import (
    article_model_inputs,
    epsilon_normalized,
    filter_clean_baseline_images,
    validate_imagenet_split_paths,
)
from deepdetector.evaluation.table7 import evaluate_table7_filter
from deepdetector.io.csv_writer import write_pivot_csv as write_shared_pivot_csv
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_json
from deepdetector.models.imagenet_wrappers import (
    GoogLeNetCaffeWrapper,
    build_googlenet_caffe_model,
)


logger = logging.getLogger(__name__)

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


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a project-relative path."""
    return resolve_project_path(path_value)


def _output_dir(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", config.get("outputs", {}))
    output_dir = _resolve_path(output_config.get("dir") or output_config.get("results_dir"))
    if output_dir is None:
        raise ValueError("Table 7 ImageNet must define output.dir.")
    return output_dir


def _status_path(config: Dict[str, Any], output_dir: Path) -> Path:
    """Return the configured status JSON path."""
    output_config = config.get("output", config.get("outputs", {}))
    return output_dir / str(output_config.get("status_json", "table_7_status.json"))


def _write_status(
    config: Dict[str, Any],
    output_dir: Path,
    status: str,
    **fields: Any,
) -> Path:
    """Write a status JSON file for complete or partial ImageNet runs."""
    payload = {"status": status}
    payload.update(fields)
    path = _status_path(config, output_dir)
    write_metrics_json(path, payload)
    return path


def _remove_stale_standard_outputs(output_dir: Path) -> None:
    """Remove stale filter-grid outputs from older Table 7 runs."""
    for filename in ("metrics.csv", "metrics.json"):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def build_imagenet_table7_model(config: Dict[str, Any]) -> GoogLeNetCaffeWrapper:
    """Instantiate the configured GoogLeNet Caffe wrapper."""
    return build_googlenet_caffe_model(config)


def load_imagenet_table7_subset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load configured ImageNet class folders as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    class_configs = dataset_config.get("classes", [])
    if not class_configs:
        raise ValueError("Table 7 ImageNet config must define dataset.classes.")

    validate_imagenet_split_paths(config)
    rows = class_image_rows(class_configs)
    if bool(dataset_config.get("shuffle", False)):
        seed = int(config.get("experiment", {}).get("seed", 20170830))
        rng = np.random.RandomState(seed)
        rows = [rows[int(index)] for index in rng.permutation(len(rows))]

    configured_n_samples = dataset_config.get("n_samples")
    if configured_n_samples not in (None, "", "all"):
        rows = rows[: int(configured_n_samples)]

    image_size = int(dataset_config.get("image_size", 224))
    images = []
    labels = []
    for path, label in rows:
        image = read_rgb_image(path)
        images.append(resize_normalized_image(image, image_size=image_size))
        labels.append(label)

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _article_model_inputs(model: GoogLeNetCaffeWrapper, images: np.ndarray) -> np.ndarray:
    """Return images in the Caffe input space used by the source article."""
    return article_model_inputs(model, images)


def _epsilon_normalized(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in normalized [0, 1] image scale."""
    return epsilon_normalized(config)


def configured_masks(config: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    """Yield configured Table 7 mask and size combinations."""
    filter_config = config.get("filter", {})
    mask_types = filter_config.get("mask_types", ["cross", "diamond", "box"])
    sizes = filter_config.get("sizes", [3, 5, 7, 9])
    for mask_type in mask_types:
        for size in sizes:
            yield str(mask_type), int(size)


def write_pivot_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str] = PIVOT_COLUMNS,
) -> Path:
    """Write a Table 7 pivot with metric rows and mask-size columns."""
    return write_shared_pivot_csv(path=path, rows=rows, columns=columns)


def run_table7_imagenet_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the official ImageNet Table 7 experiment and write pivot outputs."""
    output_dir = ensure_dir(_output_dir(config))
    _remove_stale_standard_outputs(output_dir)
    try:
        model = build_imagenet_table7_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        path = _write_status(
            output_dir=output_dir,
            config=config,
            status="bloqueado_caffe",
            message=str(exc),
        )
        return {"status": "bloqueado_caffe", "status_json": str(path)}
    except OSError as exc:
        logger.warning("%s", exc)
        path = _write_status(
            output_dir=output_dir,
            config=config,
            status="bloqueado_modelo_googlenet",
            message=str(exc),
        )
        return {"status": "bloqueado_modelo_googlenet", "status_json": str(path)}

    try:
        images, labels = load_imagenet_table7_subset(config)
    except IOError as exc:
        logger.warning("%s", exc)
        path = _write_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            message=str(exc),
            n_loaded=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    if len(images) == 0:
        path = _write_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_carregada",
            n_loaded=0,
        )
        return {"status": "parcial", "status_json": str(path)}

    images, labels, selected_indices, clean_summary = filter_clean_baseline_images(
        model=model,
        images=images,
        labels=labels,
    )
    if len(images) == 0:
        path = _write_status(
            output_dir=output_dir,
            config=config,
            status="parcial",
            limitation="nenhuma_imagem_limpa_correta",
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
        path = _write_status(
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
            epsilon=_epsilon_normalized(config),
            entropy_threshold=float(config.get("filter", {}).get("entropy_threshold", 5.0)),
        )
        row = asdict(result)
        row["skipped_wrong_baseline"] = clean_summary["skipped_wrong_baseline"]
        rows.append(row)

    output_config = config.get("output", config.get("outputs", {}))
    pivot_path = write_pivot_csv(
        output_dir / str(output_config.get("pivot_csv", "table_7_imagnet.csv")),
        rows,
    )
    status_path = _write_status(
        output_dir=output_dir,
        config=config,
        status="completo",
        n_loaded=int(len(images)),
        pivot_csv=str(pivot_path),
        **clean_summary,
    )
    return {
        "status": "completo",
        "pivot_csv": str(pivot_path),
        "status_json": str(status_path),
    }
