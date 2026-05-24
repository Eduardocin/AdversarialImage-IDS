"""Reproduce ImageNet Table 7 spatial smoothing filter metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import (  # noqa: E402
    generate_fgsm_imagenet,
    predict_caffe_label,
    preprocess_caffe_inputs,
)
from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.evaluation.table7 import evaluate_table7_filter  # noqa: E402
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_7.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "imagenet" / "article_reproduction"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


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


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root."""
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    """Load the experiment YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def output_dir_from_config(config: Dict[str, Any], override: Optional[str]) -> Path:
    """Return the configured output directory."""
    outputs = config.get("outputs", config.get("output", {}))
    return _resolve_path(override or outputs.get("results_dir")) or DEFAULT_OUTPUT_DIR


def write_status(output_dir: Path, status: str, **fields: Any) -> Path:
    """Write a status JSON file for partial ImageNet runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(fields)
    path = output_dir / "table_7_status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_model(config: Dict[str, Any]) -> GoogLeNetCaffeWrapper:
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


def _article_model_inputs(model: GoogLeNetCaffeWrapper, images: np.ndarray) -> np.ndarray:
    """Return images in the Caffe input space used by the source article."""
    return preprocess_caffe_inputs(model, images)


def _predict_label(model: GoogLeNetCaffeWrapper, image: np.ndarray) -> int:
    """Predict the top-1 label for one HWC image or preprocessed Caffe tensor."""
    return predict_caffe_label(model, image)


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


def load_subset_images(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load configured ImageNet class folders as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    class_configs = dataset_config.get("classes", [])
    if not class_configs:
        raise ValueError("Config must define dataset.classes.")

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
    override_path: Optional[str],
    selected_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Load or generate adversarial images for the table run."""
    attack_config = config.get("attack", {})
    adv_path = _resolve_path(override_path or attack_config.get("adversarial_path"))
    if adv_path is not None and adv_path.is_file():
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
    return adv_images


def configured_masks(config: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    """Yield configured Table 7 mask and size combinations."""
    filter_config = config.get("filter", {})
    mask_types = filter_config.get("mask_types", ["cross", "diamond", "box"])
    sizes = filter_config.get("sizes", [3, 5, 7, 9])
    for mask_type in mask_types:
        for size in sizes:
            yield str(mask_type), int(size)


def write_pivot_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Write a Table 7 pivot with metric rows and mask-size columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
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
        writer.writerow(["metric"] + columns)
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


def main() -> int:
    """Run the ImageNet Table 7 experiment and write CSV outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)
    output_dir = output_dir_from_config(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = build_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        write_status(output_dir, "bloqueado_caffe", message=str(exc))
        return 0
    except OSError as exc:
        logger.warning("%s", exc)
        write_status(output_dir, "bloqueado_modelo_googlenet", message=str(exc))
        return 0

    images, labels = load_subset_images(config)
    print("images_shape={0}".format(images.shape))
    print("labels_shape={0}".format(labels.shape))

    if args.dry_run:
        write_status(output_dir, "parcial", limitation="dry_run", n_loaded=int(len(images)))
        return 0
    if len(images) == 0:
        write_status(output_dir, "parcial", limitation="nenhuma_imagem_carregada", n_loaded=0)
        return 0

    images, labels, selected_indices, clean_summary = filter_clean_baseline_images(
        model=model,
        images=images,
        labels=labels,
    )
    print("total_images={0}".format(clean_summary["total_images"]))
    print("clean_correct={0}".format(clean_summary["clean_correct"]))
    print("skipped_wrong_baseline={0}".format(clean_summary["skipped_wrong_baseline"]))
    if len(images) == 0:
        write_status(
            output_dir,
            "parcial",
            limitation="nenhuma_imagem_limpa_correta",
            **clean_summary,
        )
        return 0
    images = _article_model_inputs(model, images)

    adv_images = adversarial_images_for_run(
        config,
        model,
        images,
        args.adv_path,
        selected_indices=selected_indices,
    )
    if adv_images is None:
        write_status(
            output_dir,
            "parcial",
            n_loaded=int(len(images)),
            **clean_summary,
            limitation="fgsm_caffe_requer_gradiente_ou_adversariais_salvas",
            message=(
                "GoogLeNetCaffeWrapper must expose Caffe gradient support. "
                "Provide --adv-path or configure attack.adversarial_path with a compatible .npy file."
            ),
        )
        return 0

    filter_config = config.get("filter", {})
    dataset = (images, labels, adv_images)
    rows = []
    for mask_type, size in configured_masks(config):
        logger.info("Evaluating mask=%s size=%s", mask_type, size)
        result = evaluate_table7_filter(
            model=model,
            dataset=dataset,
            mask_type=mask_type,
            size=size,
            epsilon=_epsilon_normalized(config),
            entropy_threshold=float(filter_config.get("entropy_threshold", 5.0)),
            quantization_step=int(filter_config.get("quantization_step", 43)),
        )
        row = asdict(result)
        row["skipped_wrong_baseline"] = clean_summary["skipped_wrong_baseline"]
        rows.append(row)

    outputs = config.get("outputs", config.get("output", {}))
    pivot_path = write_pivot_csv(
        output_dir / str(outputs.get("pivot_csv", "table_7_imagnet.csv")),
        rows,
    )

    write_status(
        output_dir,
        "completo",
        n_loaded=int(len(images)),
        **clean_summary,
        pivot_csv=str(pivot_path),
    )
    print("pivot_csv={0}".format(pivot_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
