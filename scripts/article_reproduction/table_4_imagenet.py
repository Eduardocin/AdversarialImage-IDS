"""Compare scalar quantization intervals on ImageNet adversarial samples."""

from __future__ import print_function

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "imagenet_table_4.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "imagenet" / "article_reproduction"
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_imagenet  # noqa: E402
from deepdetector.data.imagenet import resize_normalized_image  # noqa: E402
from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    apply_filter_batch,
    evaluate_filter_predictions,
    format_percent,
    interval_size,
    scalar_filter_for_intervals,
    write_csv,
    write_markdown_table,
)
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper  # noqa: E402


logger = logging.getLogger(__name__)


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


def configured_intervals(config: Dict[str, Any]) -> Iterable[int]:
    """Return scalar quantization interval counts from config."""
    intervals = config.get("quantization", {}).get("intervals", [])
    if not intervals:
        raise ValueError("Config must define quantization.intervals.")
    for value in intervals:
        yield int(value)


def output_dir_from_config(config: Dict[str, Any], override: Optional[str]) -> Path:
    """Return the configured output directory."""
    output_config = config.get("output", {})
    output_dir = _resolve_path(override or output_config.get("results_dir"))
    return output_dir or DEFAULT_OUTPUT_DIR


def write_status(output_dir: Path, status: str, **fields: Any) -> Path:
    """Write a status JSON file for partial ImageNet runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(fields)
    path = output_dir / "status.json"
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


def _class_image_rows(images_dir: Path, class_indices: Dict[str, Any]) -> List[Tuple[Path, int]]:
    """Return sorted image paths and labels for a class-folder ImageNet subset."""
    rows = []
    for class_name, label_index in sorted(class_indices.items()):
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            raise IOError("Missing ImageNet class directory: {0}".format(class_dir))
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, int(label_index)))
    return rows


def load_subset_images(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load the configured local ImageNet subset from class folders."""
    dataset_config = config.get("dataset", {})
    images_dir = _resolve_path(dataset_config.get("images_dir"))
    if images_dir is None:
        raise ValueError("Config must define dataset.images_dir.")

    class_indices = dataset_config.get("class_indices", {})
    if not isinstance(class_indices, dict) or not class_indices:
        raise ValueError("Config must define dataset.class_indices.")

    rows = _class_image_rows(images_dir, class_indices)
    if bool(dataset_config.get("shuffle", False)):
        rng = np.random.RandomState(config.get("experiment", {}).get("seed", 20170830))
        order = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]

    configured_n_samples = dataset_config.get("n_samples")
    if configured_n_samples in (None, "", "all"):
        n_samples = len(rows)
    else:
        n_samples = int(configured_n_samples)
    rows = rows[:n_samples]
    image_size = int(dataset_config.get("image_size", 224))

    images = []
    labels = []
    for path, label_index in rows:
        image = _read_rgb_image(path)
        images.append(resize_normalized_image(image, image_size=image_size))
        labels.append(int(label_index))

    if not images:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def predict_labels(model: GoogLeNetCaffeWrapper, images: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict labels for a batch of ImageNet images."""
    labels = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        labels.extend(np.asarray(model.predict_label(batch), dtype=np.int32).tolist())
    return np.asarray(labels, dtype=np.int32)


def load_adversarial_images(path: Path, expected_shape: Tuple[int, ...]) -> np.ndarray:
    """Load adversarial images and validate their shape."""
    adv_images = np.load(str(path)).astype(np.float32)
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
    """Generate FGSM examples when the model exposes TF1 attack handles."""
    sess = getattr(model, "sess", None)
    x_placeholder = getattr(model, "x_placeholder", None)
    logits = getattr(model, "logits", None)
    if sess is None or x_placeholder is None or logits is None:
        return None

    attack_config = config.get("attack", {})
    return generate_fgsm_imagenet(
        sess=sess,
        x_placeholder=x_placeholder,
        logits=logits,
        images=images,
        eps=float(attack_config.get("epsilon", attack_config.get("eps", 4.0 / 255.0))),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 1.0)),
        batch_size=int(attack_config.get("batch_size", 32)),
    )


def adversarial_images_for_run(
    config: Dict[str, Any],
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    override_path: Optional[str],
) -> Optional[np.ndarray]:
    """Load or generate adversarial images for the table run."""
    attack_config = config.get("attack", {})
    adv_path = _resolve_path(override_path or attack_config.get("adversarial_path"))
    if adv_path is not None and adv_path.is_file():
        return load_adversarial_images(adv_path, images.shape)

    adv_images = generate_adversarial_images(config, model, images)
    if adv_images is None:
        return None

    save_path = _resolve_path(attack_config.get("save_adversarial_path"))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), adv_images)
    return adv_images


def evaluate_interval(
    model: GoogLeNetCaffeWrapper,
    images: np.ndarray,
    labels: np.ndarray,
    adv_images: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    intervals: int,
    batch_size: int,
    exclude_invalid_pairs: bool,
) -> Dict[str, Any]:
    """Evaluate one scalar quantization interval count."""
    filter_fn = scalar_filter_for_intervals(intervals)
    filtered_clean = apply_filter_batch(filter_fn, images)
    filtered_adv = apply_filter_batch(filter_fn, adv_images)
    filtered_clean_pred = predict_labels(model, filtered_clean, batch_size)
    filtered_adv_pred = predict_labels(model, filtered_adv, batch_size)
    metrics = evaluate_filter_predictions(
        y_true=labels,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
        exclude_invalid_pairs=exclude_invalid_pairs,
    )
    metrics.update(
        {
            "intervals": int(intervals),
            "interval_size": interval_size(intervals),
        }
    )
    return metrics


def main() -> int:
    """Run the interval comparison and write CSV and Markdown outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)
    output_config = config.get("output", {})
    metrics_config = config.get("metrics", {})
    evaluation_config = config.get("evaluation", {})
    model_config = config.get("model", {})

    output_dir = output_dir_from_config(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = build_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        write_status(
            output_dir,
            "bloqueado_caffe",
            limitation="caffe_indisponivel",
            message=str(exc),
        )
        return 0
    except OSError as exc:
        logger.warning("%s", exc)
        write_status(
            output_dir,
            "bloqueado_modelo_googlenet",
            limitation="assets_googlenet_indisponiveis",
            message=str(exc),
            setup_command="python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet",
        )
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

    adv_images = adversarial_images_for_run(config, model, images, args.adv_path)
    if adv_images is None:
        write_status(
            output_dir,
            "parcial",
            n_loaded=int(len(images)),
            limitation="fgsm_requer_adversariais_salvas_ou_grafo_tensorflow",
            message=(
                "GoogLeNetCaffeWrapper does not expose TensorFlow tensors. "
                "Provide --adv-path or configure attack.adversarial_path with a compatible .npy file."
            ),
        )
        return 0

    batch_size = int(model_config.get("batch_size", 32))
    clean_pred = predict_labels(model, images, batch_size)
    adv_pred = predict_labels(model, adv_images, batch_size)
    exclude_invalid_pairs = bool(evaluation_config.get("exclude_invalid_pairs", False))

    rows = []
    for intervals in configured_intervals(config):
        rows.append(
            evaluate_interval(
                model=model,
                images=images,
                labels=labels,
                adv_images=adv_images,
                clean_pred=clean_pred,
                adv_pred=adv_pred,
                intervals=intervals,
                batch_size=batch_size,
                exclude_invalid_pairs=exclude_invalid_pairs,
            )
        )

    csv_path = write_csv(
        str(output_dir / str(output_config.get("csv", "table_4_imagenet_scalar_quantization_intervals.csv"))),
        rows,
        metrics_config.get(
            "raw_fields",
            [
                "intervals",
                "interval_size",
                "recall_percent",
                "precision_percent",
                "f1_percent",
            ],
        ),
    )

    markdown_rows = []
    for row in rows:
        markdown_rows.append(
            [
                row["intervals"],
                row["interval_size"],
                format_percent(row["recall_percent"]),
                format_percent(row["precision_percent"]),
                format_percent(row["f1_percent"]),
            ]
        )

    md_path = write_markdown_table(
        str(output_dir / str(output_config.get("markdown", "table_4_imagenet_scalar_quantization_intervals.md"))),
        "ImageNet Scalar Quantization Intervals",
        metrics_config.get(
            "columns",
            ["Intervals", "Interval size", "Recall", "Precision", "F1"],
        ),
        markdown_rows,
    )

    write_status(
        output_dir,
        "completo",
        n_loaded=int(len(images)),
        results_csv=str(csv_path),
        results_md=str(md_path),
    )
    print("results_csv={0}".format(csv_path))
    print("results_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
