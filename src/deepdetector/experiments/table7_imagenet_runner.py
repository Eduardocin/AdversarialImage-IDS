"""Official ImageNet Table 7 spatial smoothing runner."""

from __future__ import annotations

import csv
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.fgsm_imagenet import (
    generate_fgsm_imagenet,
    predict_caffe_label,
    preprocess_caffe_inputs,
)
from deepdetector.data.imagenet import resize_normalized_image
from deepdetector.evaluation.table7 import evaluate_table7_filter
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_json
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")
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
    output_dir = _resolve_path(config.get("output", {}).get("dir"))
    if output_dir is None:
        raise ValueError("Table 7 ImageNet must define output.dir.")
    return output_dir


def _status_path(config: Dict[str, Any], output_dir: Path) -> Path:
    """Return the configured status JSON path."""
    output_config = config.get("output", {})
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
    model_config = config.get("model", {})
    return GoogLeNetCaffeWrapper(
        model_dir=str(_resolve_path(model_config.get("model_dir"))),
        deploy_prototxt=str(_resolve_path(model_config.get("deploy_proto"))),
        caffemodel=str(_resolve_path(model_config.get("caffemodel"))),
        mean_file=(
            str(_resolve_path(model_config.get("mean_file")))
            if model_config.get("mean_file")
            else None
        ),
        use_gpu=bool(model_config.get("use_gpu", False)),
        batch_size=int(model_config.get("batch_size", 32)),
    )


def _read_rgb_image(path: Path) -> np.ndarray:
    """Load one image as normalized RGB float32 data."""
    from PIL import Image

    with Image.open(str(path)) as image:
        rgb_image = image.convert("RGB")
        return (np.asarray(rgb_image, dtype=np.float32) / 255.0).astype(np.float32)


def _class_image_rows(class_configs: Sequence[Dict[str, Any]]) -> List[Tuple[Path, int]]:
    """Return sorted image paths and labels for configured class directories."""
    rows: List[Tuple[Path, int]] = []
    for class_config in class_configs:
        class_dir = _resolve_path(class_config.get("path"))
        if class_dir is None or not class_dir.is_dir():
            raise IOError("Missing ImageNet class directory: {0}".format(class_dir))

        label = int(class_config["label"])
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, label))
    return rows


def validate_imagenet_split_paths(config: Dict[str, Any]) -> None:
    """Reject class paths that point at a different ImageNet split."""
    dataset_config = config.get("dataset", {})
    split = str(dataset_config.get("split", "")).strip().lower()
    if not split:
        return
    if split == "training":
        split = "train"

    known_splits = {"train", "validation", "test"}
    for class_config in dataset_config.get("classes", []):
        class_path = _resolve_path(class_config.get("path"))
        if class_path is None:
            continue

        path_parts = {part.lower() for part in class_path.parts}
        mismatched = sorted((known_splits - {split}) & path_parts)
        if mismatched:
            raise ValueError(
                "Configured ImageNet {0} split cannot use {1} path: {2}".format(
                    split,
                    mismatched[0],
                    class_path,
                )
            )


def load_imagenet_table7_subset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load configured ImageNet class folders as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    class_configs = dataset_config.get("classes", [])
    if not class_configs:
        raise ValueError("Table 7 ImageNet config must define dataset.classes.")

    validate_imagenet_split_paths(config)
    rows = _class_image_rows(class_configs)
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
        image = _read_rgb_image(path)
        images.append(resize_normalized_image(image, image_size=image_size))
        labels.append(label)

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _predict_label(model: GoogLeNetCaffeWrapper, image: np.ndarray) -> int:
    """Predict the top-1 label for one HWC image or Caffe tensor."""
    return predict_caffe_label(model, image)


def _label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to a Python int."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def filter_clean_baseline_images(
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Keep only images correctly classified before attack evaluation."""
    keep_indices = []
    for index, image in enumerate(images):
        clean_pred = _predict_label(model, image)
        if clean_pred == _label_to_int(labels[index]):
            keep_indices.append(index)

    selected = np.asarray(keep_indices, dtype=np.int64)
    summary = {
        "total_images": int(len(images)),
        "clean_correct": int(len(selected)),
        "skipped_wrong_baseline": int(len(images) - len(selected)),
    }
    return images[selected], labels[selected], selected, summary


def _article_model_inputs(model: GoogLeNetCaffeWrapper, images: np.ndarray) -> np.ndarray:
    """Return images in the Caffe input space used by the source article."""
    return preprocess_caffe_inputs(model, images)


def _epsilon_normalized(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in normalized [0, 1] image scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"]) / 255.0
    return float(attack_config.get("epsilon", attack_config.get("eps", 1.0 / 255.0)))


def _epsilon_255(config: Dict[str, Any]) -> float:
    """Return FGSM epsilon in raw 0-255 image scale."""
    attack_config = config.get("attack", {})
    if "epsilon_255" in attack_config:
        return float(attack_config["epsilon_255"])
    return float(_epsilon_normalized(config) * 255.0)


def load_adversarial_images(
    path: Path,
    expected_shape: Tuple[int, ...],
    selected_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load adversarial images and validate their shape."""
    adv_images = np.load(str(path)).astype(np.float32)
    if (
        selected_indices is not None
        and adv_images.shape != expected_shape
        and adv_images.ndim == len(expected_shape)
        and adv_images.shape[1:] == expected_shape[1:]
        and len(selected_indices) > 0
        and int(np.max(selected_indices)) < len(adv_images)
    ):
        adv_images = adv_images[selected_indices]

    if adv_images.shape != expected_shape:
        raise ValueError(
            "Expected adversarial array shape {0}, got {1}.".format(
                expected_shape,
                adv_images.shape,
            )
        )
    return adv_images


def generate_adversarial_images(
    config: Dict[str, Any],
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
) -> Optional[np.ndarray]:
    """Generate FGSM examples with the shared Caffe article implementation."""
    result = generate_fgsm_imagenet(
        model=model,
        images=images,
        labels=None,
        epsilon_255=_epsilon_255(config),
        skip_wrong_baseline=False,
        clip_min=0.0,
        clip_max=255.0,
    )
    return result.adversarial_images


def adversarial_images_for_run(
    config: Dict[str, Any],
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Load or generate adversarial images for the table run."""
    attack_config = config.get("attack", {})
    adv_path = _resolve_path(attack_config.get("adversarial_path"))
    if adv_path is not None and adv_path.is_file():
        logger.info("Loaded Table 7 ImageNet FGSM cache: %s", adv_path)
        return load_adversarial_images(
            adv_path,
            images.shape,
            selected_indices=selected_indices,
        )

    adv_images = generate_adversarial_images(config, model, images)
    if adv_images is None:
        return None

    save_path = _resolve_path(attack_config.get("save_adversarial_path"))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), adv_images)
        logger.info("Wrote Table 7 ImageNet FGSM cache: %s", save_path)
    return adv_images


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
    path.parent.mkdir(parents=True, exist_ok=True)
    by_column = {
        "{0}_{1}x{1}".format(row["mask_type"], int(row["size"])): row
        for row in rows
    }
    metric_rows = [
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("F1 Score", "f1"),
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + list(columns))
        for display_name, field in metric_rows:
            writer.writerow(
                [display_name]
                + [
                    "{0:.6f}".format(float(by_column[column][field]))
                    if column in by_column
                    else ""
                    for column in columns
                ]
            )
    return path


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

    images, labels = load_imagenet_table7_subset(config)
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

    output_config = config.get("output", {})
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
