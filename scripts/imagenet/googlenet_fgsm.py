"""Run GoogLeNet ImageNet detection with FGSM adversarial inputs."""

from __future__ import print_function

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_imagenet
from deepdetector.data.imagenet import load_imagenet_images, resize_normalized_image
from deepdetector.detection.prediction_change import PredictionChangeDetector
from deepdetector.evaluation.detector_metrics import (
    compute_detector_counts,
    compute_precision_recall,
    save_results_csv,
)
from deepdetector.filters.registry import FILTER_REGISTRY
from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "imagenet_googlenet_fgsm.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "imagenet" / "googlenet_fgsm"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the ImageNet track."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--adv-path",
        default=None,
        help="Path to a pre-generated .npy array with adversarial images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load Caffe and images, print shapes, and stop before inference.",
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
    """Load an experiment YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def results_dir_from_config(config: Dict[str, Any]) -> Path:
    """Return the configured output directory."""
    output = config.get("output", {})
    return _resolve_path(output.get("results_dir")) or DEFAULT_RESULTS_DIR


def write_status(results_dir: Path, status: str, **fields: Any) -> Path:
    """Write the track status as JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    payload.update(fields)
    path = results_dir / "status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def set_seeds(config: Dict[str, Any]) -> np.random.RandomState:
    """Set TensorFlow and NumPy seeds from config."""
    import tensorflow as tf

    seeds = config.get("seeds", {})
    tf_seed = int(seeds.get("tf_seed", 1234))
    np_seed = seeds.get("np_seed", [2017, 8, 30])

    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(tf_seed)
    else:
        tf.compat.v1.set_random_seed(tf_seed)
    return np.random.RandomState(np_seed)


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
    )


def load_images(
    config: Dict[str, Any],
    rng: np.random.RandomState,
) -> tuple:
    """Load the configured ImageNet sample as normalized NHWC images."""
    dataset_config = config.get("dataset", {})
    image_size = int(dataset_config.get("image_size", 224))
    preprocess_fn = lambda image: resize_normalized_image(image, image_size=image_size)

    return load_imagenet_images(
        csv_path=str(_resolve_path(dataset_config.get("labels_csv"))),
        images_dir=str(_resolve_path(dataset_config.get("images_dir"))),
        n_samples=int(dataset_config.get("n_samples", 500)),
        preprocess_fn=preprocess_fn,
        rng=rng,
    )


def top1_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute top-1 accuracy for integer labels."""
    if len(labels) == 0:
        return 0.0
    return float(np.mean(np.asarray(predictions, dtype=np.int32) == labels))


def label_to_int(label: Any) -> int:
    """Convert an integer or one-hot label to an integer class."""
    label_array = np.asarray(label)
    if label_array.ndim == 0:
        return int(label_array)
    return int(np.argmax(label_array))


def discard_record(
    filter_name: str,
    sample_index: int,
    true_label: int,
    clean_pred: int,
    adv_pred: int,
) -> Dict[str, Any]:
    """Create a detector row for a discarded pair."""
    clean_error = int(clean_pred) != int(true_label)
    attack_failed = (not clean_error) and int(adv_pred) == int(true_label)

    if clean_error:
        reason = "clean_error"
    elif attack_failed:
        reason = "attack_failed"
    else:
        reason = ""

    return {
        "filter_name": filter_name,
        "sample_index": int(sample_index),
        "true_label": int(true_label),
        "clean_pred": int(clean_pred),
        "adv_pred": int(adv_pred),
        "filtered_clean_pred": "",
        "filtered_adv_pred": "",
        "detected": False,
        "corrected": False,
        "false_positive": False,
        "discarded_clean_error": bool(clean_error),
        "discarded_attack_failed": bool(attack_failed),
        "discard_reason": reason,
    }


def evaluate_filter(
    filter_name: str,
    filter_fn: Any,
    model: GoogLeNetCaffeWrapper,
    clean_images: np.ndarray,
    adv_images: np.ndarray,
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    """Evaluate one filter over clean/adversarial ImageNet pairs."""
    detector = PredictionChangeDetector(
        sess=None,
        x_placeholder=None,
        predictions=None,
        filter_fn=filter_fn,
        predict_label_fn=model.predict_label,
    )
    records = []

    for sample_index, (clean_image, adv_image, label_value) in enumerate(
        zip(clean_images, adv_images, labels)
    ):
        true_label = label_to_int(label_value)
        clean_pred = detector.predict_label(clean_image)
        adv_pred = detector.predict_label(adv_image)

        if clean_pred != true_label or adv_pred == true_label:
            records.append(
                discard_record(
                    filter_name=filter_name,
                    sample_index=sample_index,
                    true_label=true_label,
                    clean_pred=clean_pred,
                    adv_pred=adv_pred,
                )
            )
            continue

        record = detector.detect_pair(clean_image, adv_image, true_label)
        record.update(
            {
                "filter_name": filter_name,
                "sample_index": int(sample_index),
                "discarded_clean_error": False,
                "discarded_attack_failed": False,
                "discard_reason": "",
            }
        )
        records.append(record)

    return records


def summarize_filter(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate detector metrics for one filter."""
    counts = compute_detector_counts(records)
    rates = compute_precision_recall(counts)
    n_discarded = int(counts["n_discarded_clean_error"]) + int(
        counts["n_discarded_attack_failed"]
    )

    summary = dict(counts)
    summary.update(rates)
    summary["n_total"] = len(records)
    summary["n_discarded"] = n_discarded
    return summary


def save_summary_md(
    summaries: Dict[str, Dict[str, Any]],
    path: Path,
    top1_clean: float,
    clean_count: int,
) -> Path:
    """Save a Markdown summary for the ImageNet detector run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ImageNet GoogLeNet FGSM",
        "",
        "## Clean Inference",
        "",
        "- top1_clean: {0:.6f}".format(top1_clean),
        "- n_images: {0}".format(clean_count),
        "",
        "## Detector Metrics",
        "",
        "| filter_name | n_total | n_discarded | TP | FP | FN | TN | TTP | precision | recall | f1 | ttp_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for filter_name, row in summaries.items():
        lines.append(
            "| {filter_name} | {n_total} | {n_discarded} | {TP} | {FP} | {FN} | {TN} | {TTP} | "
            "{precision:.6f} | {recall:.6f} | {f1:.6f} | {ttp_rate:.6f} |".format(
                filter_name=filter_name,
                n_total=int(row["n_total"]),
                n_discarded=int(row["n_discarded"]),
                TP=int(row["TP"]),
                FP=int(row["FP"]),
                FN=int(row["FN"]),
                TN=int(row["TN"]),
                TTP=int(row["TTP"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                ttp_rate=float(row["ttp_rate"]),
            )
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def configured_filters(config: Dict[str, Any]) -> Iterable[tuple]:
    """Yield configured filter names and callables."""
    filter_names = config.get("detection", {}).get("filters", [])
    for filter_name in filter_names:
        if filter_name not in FILTER_REGISTRY:
            raise KeyError("Unknown filter in config: {0}".format(filter_name))
        yield filter_name, FILTER_REGISTRY[filter_name]


def load_adversarial_images(path: Path, expected_shape: tuple) -> np.ndarray:
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
    """Generate adversarial images when the model exposes TF1 attack handles."""
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
        eps=float(attack_config.get("eps", 4.0 / 255.0)),
        clip_min=float(attack_config.get("clip_min", 0.0)),
        clip_max=float(attack_config.get("clip_max", 1.0)),
        batch_size=int(attack_config.get("batch_size", 32)),
    )


def run(args: argparse.Namespace) -> int:
    """Run the configured ImageNet track."""
    config = load_config(Path(args.config))
    results_dir = results_dir_from_config(config)
    rng = set_seeds(config)

    try:
        model = build_model(config)
    except ImportError as exc:
        logger.warning("%s", exc)
        write_status(
            results_dir,
            "bloqueado_caffe",
            limitation="caffe_indisponivel",
            message=str(exc),
        )
        return 0

    images, labels = load_images(config, rng)
    print("images_shape={0}".format(images.shape))
    print("labels_shape={0}".format(labels.shape))

    if args.dry_run:
        write_status(
            results_dir,
            "parcial",
            limitation="dry_run",
            n_loaded=int(len(images)),
        )
        return 0

    if len(images) == 0:
        write_status(
            results_dir,
            "parcial",
            limitation="nenhuma_imagem_carregada",
            n_loaded=0,
        )
        return 0

    clean_predictions = model.predict_label(images)
    clean_accuracy = top1_accuracy(clean_predictions, labels)
    logger.info("top1_clean=%.6f", clean_accuracy)

    results_dir.mkdir(parents=True, exist_ok=True)
    if args.adv_path:
        adv_path = Path(args.adv_path)
        adv_images = load_adversarial_images(adv_path, images.shape)
    else:
        adv_images = generate_adversarial_images(config, model, images)
        if adv_images is None:
            write_status(
                results_dir,
                "parcial",
                top1_clean=clean_accuracy,
                n_loaded=int(len(images)),
                limitation="fgsm_requer_grafo_tensorflow",
                message=(
                    "GoogLeNetCaffeWrapper does not expose TensorFlow tensors. "
                    "Provide --adv-path with adversarial images generated by a compatible TF1 graph."
                ),
            )
            return 0

    saved_adv_path = results_dir / "adv_googlenet_fgsm.npy"
    np.save(str(saved_adv_path), adv_images)

    all_records = []
    summaries = {}
    for filter_name, filter_fn in configured_filters(config):
        records = evaluate_filter(
            filter_name=filter_name,
            filter_fn=filter_fn,
            model=model,
            clean_images=images,
            adv_images=adv_images,
            labels=labels,
        )
        all_records.extend(records)
        summaries[filter_name] = summarize_filter(records)
        logger.info(
            "%s: f1=%.6f n_total=%s",
            filter_name,
            summaries[filter_name]["f1"],
            summaries[filter_name]["n_total"],
        )

    csv_path = save_results_csv(all_records, str(results_dir / "results.csv"))
    summary_path = save_summary_md(
        summaries=summaries,
        path=results_dir / "summary.md",
        top1_clean=clean_accuracy,
        clean_count=len(images),
    )
    best_f1 = max((float(row["f1"]) for row in summaries.values()), default=0.0)

    write_status(
        results_dir,
        "completo",
        top1_clean=clean_accuracy,
        f1_detector=best_f1,
        n_loaded=int(len(images)),
        adversarial_path=str(saved_adv_path),
        results_csv=str(csv_path),
        summary_md=str(summary_path),
    )
    return 0


def main() -> int:
    """Parse arguments, run the track, and always persist status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    results_dir = DEFAULT_RESULTS_DIR

    try:
        config = load_config(Path(args.config))
        results_dir = results_dir_from_config(config)
    except Exception as exc:
        write_status(
            results_dir,
            "parcial",
            limitation="config_invalida",
            message=str(exc),
        )
        logger.exception("Could not load config.")
        return 1

    try:
        return run(args)
    except Exception as exc:
        write_status(
            results_dir,
            "parcial",
            limitation="erro_execucao",
            message=str(exc),
        )
        logger.exception("ImageNet track failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
