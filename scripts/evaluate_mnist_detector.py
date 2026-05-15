"""Evaluate MNIST DeepDetector-style prediction-change detection."""

from __future__ import print_function

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.data.mnist import load_mnist_data
from deepdetector.detection.prediction_change import PredictionChangeDetector
from deepdetector.evaluation.detector_metrics import (
    compute_detector_counts,
    compute_precision_recall,
    save_results_csv,
    save_summary_md,
)
from deepdetector.filters.quantization import nonuniform_quantization, scalar_quantization
from deepdetector.models.mnist_cnn import build_mnist_model, create_tf_session
from deepdetector.training.train_mnist import train_or_load_mnist_model


CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "clean_baseline"
FGSM_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "fgsm"
DETECTOR_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "detector"


def format_epsilon(eps: float) -> str:
    """Format epsilon using the same convention as the FGSM generator."""
    return ("{0:g}".format(eps)).replace(".", "p")


def default_adversarial_path(eps: float) -> Path:
    """Return the default FGSM adversarial examples path for an epsilon."""
    return FGSM_RESULTS_DIR / "eps_{0}".format(format_epsilon(eps)) / "adversarial_examples.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=4500)
    parser.add_argument("--adversarial-path", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory used for TensorFlow checkpoint files.",
    )
    parser.add_argument("--filename", default="mnist.ckpt")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Restore an existing checkpoint from --train-dir when available.",
    )
    parser.add_argument("--output-dir", default=str(DETECTOR_RESULTS_DIR))
    return parser


def build_filter_fns() -> "OrderedDict[str, Callable[[np.ndarray], np.ndarray]]":
    """Return the Sprint 4 quantization filters in a stable order."""
    filters = OrderedDict()
    filters["scalar_interval_128"] = lambda image: scalar_quantization(image, interval=128)
    filters["scalar_interval_64"] = lambda image: scalar_quantization(image, interval=64)
    filters["scalar_interval_43"] = lambda image: scalar_quantization(image, interval=43)
    filters["nonuniform_quantization"] = nonuniform_quantization
    return filters


def _record_is_valid(record: Dict[str, Any]) -> bool:
    """Keep only examples matching the original detector evaluation protocol."""
    true_label = int(record["true_label"])
    clean_pred = int(record["clean_pred"])
    adv_pred = int(record["adv_pred"])
    return clean_pred == true_label and adv_pred != clean_pred


def evaluate_filter(
    filter_name: str,
    filter_fn: Callable[[np.ndarray], np.ndarray],
    sess: Any,
    x: Any,
    predictions: Any,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    Y_true: np.ndarray,
    epsilon: float,
) -> List[Dict[str, Any]]:
    """Evaluate one filter and return per-example valid records."""
    detector = PredictionChangeDetector(sess, x, predictions, filter_fn)
    records = []

    for sample_index, (clean_image, adv_image, true_label) in enumerate(
        zip(X_clean, X_adv, Y_true)
    ):
        record = detector.detect_pair(clean_image, adv_image, true_label)
        if not _record_is_valid(record):
            continue
        record.update(
            {
                "filter_name": filter_name,
                "epsilon": epsilon,
                "sample_index": sample_index,
            }
        )
        records.append(record)
    return records


def aggregate_by_filter(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate counts and rates by filter name."""
    grouped = OrderedDict()
    for record in records:
        grouped.setdefault(record["filter_name"], []).append(record)

    metrics = OrderedDict()
    for filter_name, filter_records in grouped.items():
        counts = compute_detector_counts(filter_records)
        rates = compute_precision_recall(counts)
        row = dict(counts)
        row.update(rates)
        row["evaluated_pairs"] = len(filter_records)
        metrics[filter_name] = row
    return metrics


def main() -> int:
    """Run detector evaluation for saved MNIST FGSM adversarial examples."""
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    import tensorflow as tf

    if args.adversarial_path:
        adversarial_path = Path(args.adversarial_path)
    else:
        adversarial_path = default_adversarial_path(args.epsilon)
    if not adversarial_path.exists():
        raise IOError(
            "Adversarial examples not found at {0}. Run scripts/generate_mnist_fgsm.py first.".format(
                adversarial_path
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = create_tf_session()
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    X_adv = np.load(str(adversarial_path)).astype(np.float32)
    sample_count = min(args.samples, len(X_test), len(X_adv))
    X_clean = X_test[:sample_count]
    Y_eval = Y_test[:sample_count]
    X_adv = X_adv[:sample_count]

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.placeholder(tf.float32, shape=(None, 10), name="y")
    _, predictions = build_mnist_model(x)

    train_or_load_mnist_model(
        sess=sess,
        x=x,
        y=y,
        predictions=predictions,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        config={
            "nb_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_dir": args.train_dir,
            "filename": args.filename,
            "load_model": args.load_model,
        },
    )

    all_records = []
    metrics = OrderedDict()
    for filter_name, filter_fn in build_filter_fns().items():
        records = evaluate_filter(
            filter_name=filter_name,
            filter_fn=filter_fn,
            sess=sess,
            x=x,
            predictions=predictions,
            X_clean=X_clean,
            X_adv=X_adv,
            Y_true=Y_eval,
            epsilon=args.epsilon,
        )
        all_records.extend(records)
        counts = compute_detector_counts(records)
        rates = compute_precision_recall(counts)
        row = dict(counts)
        row.update(rates)
        row["evaluated_pairs"] = len(records)
        metrics[filter_name] = row
        print("{0}: evaluated_pairs={1}".format(filter_name, len(records)))

    csv_path = save_results_csv(all_records, str(output_dir / "detector_results.csv"))
    summary_path = save_summary_md(metrics, str(output_dir / "summary.md"))

    print("results_csv={0}".format(csv_path))
    print("summary_md={0}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
