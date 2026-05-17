"""Evaluate MNIST adversarial detection by prediction change after filtering."""

from __future__ import print_function

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


SCRIPTS_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "_project_root.py").is_file()
)
sys.path.insert(0, str(SCRIPTS_ROOT))
from _project_root import configure_project_paths

PROJECT_ROOT = configure_project_paths(__file__)

from deepdetector.data.mnist import load_mnist_data
from deepdetector.detection.prediction_change import PredictionChangeDetector
from deepdetector.evaluation.detector_metrics import (
    compute_detector_counts,
    compute_precision_recall,
    save_results_csv,
    save_summary_md,
)
from deepdetector.filters.quantization import nonuniform_quantization, scalar_quantization
from deepdetector.models.mnist_cnn import build_mnist_model


CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "clean_baseline"
FGSM_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "fgsm"
DETECTOR_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "detector"


FilterFn = Callable[[np.ndarray], np.ndarray]


def format_epsilon(eps: float) -> str:
    """Format epsilon using the same convention as the FGSM generator."""
    return ("{0:g}".format(eps)).replace(".", "p")


def default_adversarial_path(eps: float) -> Path:
    """Return the default FGSM adversarial examples path for an epsilon."""
    return FGSM_RESULTS_DIR / "eps_{0}".format(format_epsilon(eps)) / "adversarial_examples.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for MNIST detector evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=4500)
    parser.add_argument("--adversarial-path", default=None)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing TensorFlow checkpoint files.",
    )
    parser.add_argument("--output-dir", default=str(DETECTOR_RESULTS_DIR))
    return parser


def create_tf1_session() -> tf.compat.v1.Session:
    """Create a TF1 session and attach it to the standalone Keras backend."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    if hasattr(K, "set_learning_phase"):
        K.set_learning_phase(0)
    if hasattr(keras.backend, "set_image_dim_ordering"):
        keras.backend.set_image_dim_ordering("tf")
    return sess


def restore_latest_checkpoint(sess: tf.compat.v1.Session, train_dir: str) -> str:
    """Restore the latest TensorFlow Saver checkpoint from ``train_dir``."""
    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is None or checkpoint.model_checkpoint_path is None:
        raise IOError("No TensorFlow checkpoint found in {0}".format(train_dir))

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint.model_checkpoint_path)
    return checkpoint.model_checkpoint_path


def build_filter_fns() -> "OrderedDict[str, FilterFn]":
    """Return the four Sprint 4 quantization filters in stable order."""
    filters = OrderedDict()
    filters["scalar_interval_128"] = lambda image: scalar_quantization(
        image, interval=128, left=True
    )
    filters["scalar_interval_64"] = lambda image: scalar_quantization(
        image, interval=64, left=True
    )
    filters["scalar_interval_43"] = lambda image: scalar_quantization(
        image, interval=43, left=True
    )
    filters["nonuniform_quantization"] = nonuniform_quantization
    return filters


def _discard_flags(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return discard flags for one clean/adversarial pair."""
    clean_error = int(record["clean_pred"]) != int(record["true_label"])
    attack_failed = (not clean_error) and int(record["adv_pred"]) == int(
        record["true_label"]
    )

    if clean_error:
        reason = "clean_error"
    elif attack_failed:
        reason = "attack_failed"
    else:
        reason = ""

    return {
        "discarded_clean_error": bool(clean_error),
        "discarded_attack_failed": bool(attack_failed),
        "discard_reason": reason,
    }


def evaluate_filter(
    filter_name: str,
    filter_fn: FilterFn,
    sess: tf.compat.v1.Session,
    x_placeholder: tf.Tensor,
    predictions: tf.Tensor,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    Y_true: np.ndarray,
) -> List[Dict[str, Any]]:
    """Evaluate one filter over all pairs, retaining discarded rows."""
    detector = PredictionChangeDetector(sess, x_placeholder, predictions, filter_fn)
    records = []

    for sample_index, (clean_image, adv_image, true_label) in enumerate(
        zip(X_clean, X_adv, Y_true)
    ):
        record = detector.detect_pair(clean_image, adv_image, true_label)
        record.update(_discard_flags(record))
        record.update(
            {
                "filter_name": filter_name,
                "sample_index": sample_index,
            }
        )
        records.append(record)

    return records


def _summarize_filter(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all summary metrics for one filter."""
    counts = compute_detector_counts(records)
    rates = compute_precision_recall(counts)
    summary = dict(counts)
    summary.update(rates)
    summary["n_total"] = len(records)
    return summary


def main() -> int:
    """Run detector evaluation for saved FGSM examples."""
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    if args.adversarial_path:
        adversarial_path = Path(args.adversarial_path)
    else:
        adversarial_path = default_adversarial_path(args.epsilon)
    if not adversarial_path.exists():
        raise IOError(
            "Adversarial examples not found at {0}. Run scripts/mnist_m1_fgsm/generate_mnist_fgsm.py first.".format(
                adversarial_path
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = create_tf1_session()
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    del X_train, Y_train

    X_adv = np.load(str(adversarial_path)).astype(np.float32)
    if X_adv.ndim != 4 or X_adv.shape[1:] != (28, 28, 1):
        raise ValueError("Expected adversarial array shape (N, 28, 28, 1).")

    sample_count = min(args.samples, len(X_test), len(X_adv))
    X_clean = np.asarray(X_test[:sample_count], dtype=np.float32)
    X_adv = np.asarray(X_adv[:sample_count], dtype=np.float32)
    Y_eval = Y_test[:sample_count]

    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, 28, 28, 1),
        name="x",
    )
    model, predictions = build_mnist_model(x_placeholder)
    del model

    checkpoint_path = restore_latest_checkpoint(sess, args.train_dir)
    print("checkpoint_path={0}".format(checkpoint_path))

    all_records = []
    metrics = OrderedDict()
    for filter_name, filter_fn in build_filter_fns().items():
        records = evaluate_filter(
            filter_name=filter_name,
            filter_fn=filter_fn,
            sess=sess,
            x_placeholder=x_placeholder,
            predictions=predictions,
            X_clean=X_clean,
            X_adv=X_adv,
            Y_true=Y_eval,
        )
        all_records.extend(records)
        metrics[filter_name] = _summarize_filter(records)
        print(
            "{0}: n_total={1} TP={2} FP={3} FN={4} TTP={5}".format(
                filter_name,
                metrics[filter_name]["n_total"],
                metrics[filter_name]["TP"],
                metrics[filter_name]["FP"],
                metrics[filter_name]["FN"],
                metrics[filter_name]["TTP"],
            )
        )

    csv_path = save_results_csv(all_records, str(output_dir / "detector_results.csv"))
    summary_path = save_summary_md(metrics, "all_filters", str(output_dir / "summary.md"))

    print("results_csv={0}".format(csv_path))
    print("summary_md={0}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
