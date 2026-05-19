"""Evaluate MNIST detection with entropy-based quantization."""

from __future__ import print_function

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.data.mnist import load_mnist_data
from deepdetector.detection.prediction_change import PredictionChangeDetector
from deepdetector.evaluation.detector_metrics import (
    compute_detector_counts,
    compute_precision_recall,
)
from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.models.mnist_cnn import build_mnist_model


CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "clean_baseline"
FGSM_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "fgsm"
ENTROPY_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "entropy"


FilterFn = Callable[[np.ndarray], np.ndarray]
RANGE_NAMES = ("low", "mid", "high")


def format_epsilon(eps: float) -> str:
    """Format epsilon for result directory names."""
    return ("{0:g}".format(eps)).replace(".", "p")


def default_adversarial_path(eps: float) -> Path:
    """Return the default adversarial examples path for an epsilon."""
    return FGSM_RESULTS_DIR / "eps_{0}".format(format_epsilon(eps)) / "adversarial_examples.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for entropy detector evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=4500)
    parser.add_argument("--adversarial-path", default=None)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing TensorFlow checkpoint files.",
    )
    parser.add_argument("--output-dir", default=str(ENTROPY_RESULTS_DIR))
    return parser


def create_tf1_session() -> tf.compat.v1.Session:
    """Create a TensorFlow 1.x session and attach it to Keras."""
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
    """Restore the latest TensorFlow Saver checkpoint from a directory."""
    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is None or checkpoint.model_checkpoint_path is None:
        raise IOError("No TensorFlow checkpoint found in {0}".format(train_dir))

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint.model_checkpoint_path)
    return checkpoint.model_checkpoint_path


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


def _empty_range_counts() -> "OrderedDict[str, Dict[str, int]]":
    """Create counters for entropy ranges."""
    counts = OrderedDict()
    for name in RANGE_NAMES:
        prefix = name
        counts[name] = {
            "{0}TP".format(prefix): 0,
            "{0}FP".format(prefix): 0,
            "{0}FN".format(prefix): 0,
            "{0}TTP".format(prefix): 0,
            "n_valid": 0,
        }
    return counts


def _increment_range_counts(
    range_counts: "OrderedDict[str, Dict[str, int]]",
    record: Dict[str, Any],
) -> None:
    """Update per-range counters for one non-discarded record."""
    adv_range = str(record["entropy_range"])
    clean_range = str(record["clean_entropy_range"])

    range_counts[adv_range]["n_valid"] += 1

    if bool(record["detected"]):
        range_counts[adv_range]["{0}TP".format(adv_range)] += 1
        if bool(record["corrected"]):
            range_counts[adv_range]["{0}TTP".format(adv_range)] += 1
    else:
        if adv_range == "mid":
            # BUG ORIGINAL: highFN incrementado no caso mid nÃ£o detectado â€” reproduzido fielmente
            range_counts["high"]["highFN"] += 1
        else:
            range_counts[adv_range]["{0}FN".format(adv_range)] += 1

    if bool(record["false_positive"]):
        range_counts[clean_range]["{0}FP".format(clean_range)] += 1


def _range_rates(counts: Dict[str, int], name: str) -> Dict[str, float]:
    """Compute precision, recall, F1 and TTP rate for one range."""
    tp = int(counts.get("{0}TP".format(name), 0))
    fp = int(counts.get("{0}FP".format(name), 0))
    fn = int(counts.get("{0}FN".format(name), 0))
    ttp = int(counts.get("{0}TTP".format(name), 0))

    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    ttp_rate = ttp / float(tp) if tp else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ttp_rate": float(ttp_rate),
    }


def evaluate_entropy_detector(
    sess: tf.compat.v1.Session,
    x_placeholder: tf.Tensor,
    predictions: tf.Tensor,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    Y_true: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate entropy-based quantization over clean/adversarial pairs."""
    filter_fn = lambda img: entropy_based_quantization(img)[0]
    detector = PredictionChangeDetector(sess, x_placeholder, predictions, filter_fn)
    records = []
    range_counts = _empty_range_counts()

    for sample_index, (clean_image, adv_image, true_label) in enumerate(
        zip(X_clean, X_adv, Y_true)
    ):
        record = detector.detect_pair(clean_image, adv_image, true_label)
        record.update(_discard_flags(record))

        _, clean_meta = entropy_based_quantization(clean_image)
        _, adv_meta = entropy_based_quantization(adv_image)
        record.update(
            {
                "filter_name": "entropy_based_quantization",
                "sample_index": sample_index,
                "entropy": float(adv_meta["entropy"]),
                "entropy_range": adv_meta["range"],
                "interval_used": int(adv_meta["interval_used"]),
                "clean_entropy": float(clean_meta["entropy"]),
                "clean_entropy_range": clean_meta["range"],
                "clean_interval_used": int(clean_meta["interval_used"]),
            }
        )

        records.append(record)
        if not record["discarded_clean_error"] and not record["discarded_attack_failed"]:
            _increment_range_counts(range_counts, record)

    return {"records": records, "range_counts": range_counts}


def save_entropy_results_csv(records: Iterable[Dict[str, Any]], path: str) -> str:
    """Save entropy detector records to CSV."""
    rows = list(records)
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)

    preferred = [
        "filter_name",
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "filtered_clean_pred",
        "filtered_adv_pred",
        "detected",
        "corrected",
        "false_positive",
        "discarded_clean_error",
        "discarded_attack_failed",
        "discard_reason",
        "entropy",
        "entropy_range",
        "interval_used",
        "clean_entropy",
        "clean_entropy_range",
        "clean_interval_used",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = preferred + extra

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _global_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute global counts and rates."""
    counts = compute_detector_counts(records)
    rates = compute_precision_recall(counts)
    summary = dict(counts)
    summary.update(rates)
    summary["n_total"] = len(records)
    return summary


def save_entropy_summary_md(
    global_metrics: Dict[str, Any],
    range_counts: "OrderedDict[str, Dict[str, int]]",
    path: str,
) -> str:
    """Save global and per-range metrics to Markdown."""
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)

    n_discarded = int(global_metrics["n_discarded_clean_error"]) + int(
        global_metrics["n_discarded_attack_failed"]
    )

    lines = [
        "# MNIST Entropy Detector",
        "",
        "## MÃ©tricas globais",
        "",
        "| n_total | n_descartados | TP | FP | FN | TN | TTP | precision | recall | f1 | ttp_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {n_total} | {n_discarded} | {TP} | {FP} | {FN} | {TN} | {TTP} | "
        "{precision:.6f} | {recall:.6f} | {f1:.6f} | {ttp_rate:.6f} |".format(
            n_total=int(global_metrics["n_total"]),
            n_discarded=n_discarded,
            TP=int(global_metrics["TP"]),
            FP=int(global_metrics["FP"]),
            FN=int(global_metrics["FN"]),
            TN=int(global_metrics["TN"]),
            TTP=int(global_metrics["TTP"]),
            precision=float(global_metrics["precision"]),
            recall=float(global_metrics["recall"]),
            f1=float(global_metrics["f1"]),
            ttp_rate=float(global_metrics["ttp_rate"]),
        ),
        "",
        "## MÃ©tricas por faixa",
        "",
        "| faixa | n_valid | TP | FP | FN | TTP | precision | recall | f1 | ttp_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, counts in range_counts.items():
        rates = _range_rates(counts, name)
        lines.append(
            "| {name} | {n_valid} | {TP} | {FP} | {FN} | {TTP} | "
            "{precision:.6f} | {recall:.6f} | {f1:.6f} | {ttp_rate:.6f} |".format(
                name=name,
                n_valid=int(counts["n_valid"]),
                TP=int(counts["{0}TP".format(name)]),
                FP=int(counts["{0}FP".format(name)]),
                FN=int(counts["{0}FN".format(name)]),
                TTP=int(counts["{0}TTP".format(name)]),
                precision=float(rates["precision"]),
                recall=float(rates["recall"]),
                f1=float(rates["f1"]),
                ttp_rate=float(rates["ttp_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Nota de reproduÃ§Ã£o",
            "",
            "Quando a entropia adversarial estÃ¡ na faixa `mid` e o filtro nÃ£o muda a "
            "prediÃ§Ã£o, o contador incrementado Ã© `highFN`, nÃ£o `midFN`. Esse "
            "comportamento desloca falsos negativos da faixa `mid` para a faixa "
            "`high`, reduzindo o denominador de recall em `mid` e aumentando o "
            "denominador de recall em `high`.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> int:
    """Run entropy detector evaluation for saved adversarial examples."""
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

    result = evaluate_entropy_detector(
        sess=sess,
        x_placeholder=x_placeholder,
        predictions=predictions,
        X_clean=X_clean,
        X_adv=X_adv,
        Y_true=Y_eval,
    )
    records = result["records"]
    range_counts = result["range_counts"]
    global_metrics = _global_summary(records)

    csv_path = save_entropy_results_csv(
        records,
        str(output_dir / "entropy_results.csv"),
    )
    summary_path = save_entropy_summary_md(
        global_metrics,
        range_counts,
        str(output_dir / "summary.md"),
    )

    print("results_csv={0}".format(csv_path))
    print("summary_md={0}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

