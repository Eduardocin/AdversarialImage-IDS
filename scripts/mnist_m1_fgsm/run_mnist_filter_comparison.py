"""Run the MNIST prediction-change detector for every registered filter."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List

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
)
from deepdetector.filters.registry import FILTER_REGISTRY
from deepdetector.models.mnist_cnn import build_mnist_model


CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "clean_baseline"
FGSM_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist" / "fgsm"
FINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"


FilterFn = Callable[[np.ndarray], np.ndarray]
SUMMARY_COLUMNS = [
    "filter_name",
    "n_total",
    "n_discarded",
    "TP",
    "FP",
    "FN",
    "TN",
    "TTP",
    "precision",
    "recall",
    "f1",
    "ttp_rate",
]


def format_epsilon(eps: float) -> str:
    """Format epsilon for stable result directory names."""
    return ("{0:g}".format(eps)).replace(".", "p")


def default_adversarial_path(eps: float) -> Path:
    """Return the default adversarial examples path for an epsilon."""
    return FGSM_RESULTS_DIR / "eps_{0}".format(format_epsilon(eps)) / "adversarial_examples.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the filter comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=4500)
    parser.add_argument("--adversarial-path", default=None)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing TensorFlow checkpoint files.",
    )
    parser.add_argument("--output-dir", default=str(FINAL_RESULTS_DIR))
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
    """Restore the latest TensorFlow Saver checkpoint from a directory."""
    checkpoint = tf.train.get_checkpoint_state(train_dir)
    if checkpoint is None or checkpoint.model_checkpoint_path is None:
        raise IOError("No TensorFlow checkpoint found in {0}".format(train_dir))

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, checkpoint.model_checkpoint_path)
    return checkpoint.model_checkpoint_path


def label_to_int(label: Any) -> int:
    """Convert an integer or one-hot encoded label to an integer class."""
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
    """Create a record for a pair removed before filtered prediction checks."""
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
    filter_fn: FilterFn,
    sess: tf.compat.v1.Session,
    x_placeholder: tf.Tensor,
    predictions: tf.Tensor,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    Y_true: np.ndarray,
) -> List[Dict[str, Any]]:
    """Evaluate one filter over all clean/adversarial pairs."""
    detector = PredictionChangeDetector(sess, x_placeholder, predictions, filter_fn)
    records = []

    for sample_index, (clean_image, adv_image, true_label_value) in enumerate(
        zip(X_clean, X_adv, Y_true)
    ):
        true_label = label_to_int(true_label_value)
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
    """Compute one aggregate CSV row from detector records."""
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


def save_final_results_csv(rows: Iterable[Dict[str, Any]], path: Path) -> Path:
    """Save one aggregate result row per filter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})
    return path


def _format_metric(value: Any) -> str:
    """Format floats consistently for Markdown tables."""
    if isinstance(value, float):
        return "{0:.6f}".format(value)
    return str(value)


def _best_rows(rows: List[Dict[str, Any]], metric: str, reverse: bool = True) -> List[Dict[str, Any]]:
    """Return all rows tied for the best value of a metric."""
    if not rows:
        return []
    values = [float(row[metric]) for row in rows]
    target = max(values) if reverse else min(values)
    return [row for row in rows if float(row[metric]) == target]


def _describe_best(rows: List[Dict[str, Any]], metric: str, reverse: bool = True) -> str:
    """Describe the best filter names and value for one metric."""
    best = _best_rows(rows, metric, reverse=reverse)
    if not best:
        return "n/a"
    names = ", ".join(row["filter_name"] for row in best)
    return "{0} ({1}={2})".format(names, metric, _format_metric(best[0][metric]))


def save_final_report_md(rows: List[Dict[str, Any]], path: Path) -> Path:
    """Save a Markdown report with full results and automatic highlights."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# MNIST Filter Comparison",
        "",
        "## 1. Resultados consolidados",
        "",
        "| filter_name | n_total | n_discarded | TP | FP | FN | TN | TTP | precision | recall | f1 | ttp_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {filter_name} | {n_total} | {n_discarded} | {TP} | {FP} | {FN} | {TN} | {TTP} | "
            "{precision} | {recall} | {f1} | {ttp_rate} |".format(
                filter_name=row["filter_name"],
                n_total=int(row["n_total"]),
                n_discarded=int(row["n_discarded"]),
                TP=int(row["TP"]),
                FP=int(row["FP"]),
                FN=int(row["FN"]),
                TN=int(row["TN"]),
                TTP=int(row["TTP"]),
                precision=_format_metric(row["precision"]),
                recall=_format_metric(row["recall"]),
                f1=_format_metric(row["f1"]),
                ttp_rate=_format_metric(row["ttp_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## 2. Analise automatica",
            "",
            "- Maior recall: {0}".format(_describe_best(rows, "recall", reverse=True)),
            "- Maior precision: {0}".format(_describe_best(rows, "precision", reverse=True)),
            "- Maior ttp_rate: {0}".format(_describe_best(rows, "ttp_rate", reverse=True)),
            "- Menor FP: {0}".format(_describe_best(rows, "FP", reverse=False)),
            "",
            "## 3. Observacoes sobre descartes",
            "",
            "| filter_name | n_total | n_discarded | discarded_clean_error | discarded_attack_failed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        lines.append(
            "| {filter_name} | {n_total} | {n_discarded} | {clean_error} | {attack_failed} |".format(
                filter_name=row["filter_name"],
                n_total=int(row["n_total"]),
                n_discarded=int(row["n_discarded"]),
                clean_error=int(row["n_discarded_clean_error"]),
                attack_failed=int(row["n_discarded_attack_failed"]),
            )
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_summary_row(filter_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a named summary row for one filter."""
    row = summarize_filter(records)
    row["filter_name"] = filter_name
    return row


def main() -> int:
    """Run the full comparison and save aggregate outputs."""
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
    try:
        X_train, Y_train, X_test, Y_test = load_mnist_data()
        del X_train, Y_train

        X_adv = np.load(str(adversarial_path)).astype(np.float32)
        if X_adv.ndim != 4 or X_adv.shape[1:] != (28, 28, 1):
            raise ValueError("Expected adversarial array shape (N, 28, 28, 1).")

        sample_count = min(args.samples, len(X_test), len(X_adv))
        X_clean = np.asarray(X_test[:sample_count], dtype=np.float32)
        X_adv = np.asarray(X_adv[:sample_count], dtype=np.float32)
        Y_eval = Y_test[:sample_count]
        print("sample_count={0}".format(sample_count))

        x_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 28, 28, 1),
            name="x",
        )
        model, predictions = build_mnist_model(x_placeholder)
        del model

        checkpoint_path = restore_latest_checkpoint(sess, args.train_dir)
        print("checkpoint_path={0}".format(checkpoint_path))

        rows = []
        for filter_name, filter_fn in FILTER_REGISTRY.items():
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
            row = build_summary_row(filter_name, records)
            rows.append(row)
            print(
                "{0}: n_total={1} n_discarded={2} TP={3} FP={4} FN={5} TN={6} TTP={7}".format(
                    filter_name,
                    row["n_total"],
                    row["n_discarded"],
                    row["TP"],
                    row["FP"],
                    row["FN"],
                    row["TN"],
                    row["TTP"],
                )
            )

        csv_path = save_final_results_csv(rows, output_dir / "final_mnist_results.csv")
        report_path = save_final_report_md(rows, output_dir / "final_mnist_report.md")

        print("results_csv={0}".format(csv_path))
        print("report_md={0}".format(report_path))
        return 0
    finally:
        sess.close()


if __name__ == "__main__":
    raise SystemExit(main())
