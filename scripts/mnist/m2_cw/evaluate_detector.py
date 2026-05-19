"""Evaluate prediction-change detection on MNIST M2 CW examples."""

from __future__ import print_function

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from deepdetector.data.mnist import load_mnist_data
from deepdetector.evaluation.article_reproduction import (
    adaptive_quantization_filter,
    apply_filter_batch,
    label_to_int,
    predict_labels,
    proposed_detection_filter,
    scalar_filter_for_intervals,
)
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model, load_mnist_m2_model


FilterFn = Callable[[np.ndarray], np.ndarray]
CLEAN_BASELINE_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw" / "clean_baseline"
M2_CW_DIR = PROJECT_ROOT / "results" / "mnist" / "m2_cw"
DETECTOR_RESULTS_DIR = M2_CW_DIR / "detector"


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated float list."""
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def format_kappa(kappa: float) -> str:
    """Format kappa for stable result directory names."""
    value = float(kappa)
    if value.is_integer():
        return "{0:.1f}".format(value).replace(".", "p")
    return "{0:g}".format(value).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=["cw_l2", "cw_linf", "all"], default="all")
    parser.add_argument("--kappas", default="0.0,0.5,1.0,2.0,4.0")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=9000)
    parser.add_argument(
        "--train-dir",
        default=str(CLEAN_BASELINE_DIR / "checkpoints"),
        help="Directory containing the M2 TensorFlow checkpoint.",
    )
    parser.add_argument("--output-dir", default=str(DETECTOR_RESULTS_DIR))
    parser.add_argument(
        "--filter",
        choices=["final", "scalar", "adaptive"],
        default="final",
        help="Filter family used by the prediction-change detector.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def select_filter(name: str) -> FilterFn:
    """Return the requested image transformation function."""
    if name == "final":
        return proposed_detection_filter
    if name == "adaptive":
        return adaptive_quantization_filter
    if name == "scalar":
        return scalar_filter_for_intervals(6)
    raise ValueError("Unsupported filter: {0}".format(name))


def create_restored_m2_graph(train_dir: str) -> Dict[str, Any]:
    """Create and restore the M2 TF1 graph."""
    import tensorflow as tf

    sess = create_tf_session()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    model, predictions = build_mnist_m2_model(x)
    checkpoint = load_mnist_m2_model(sess, train_dir)
    if checkpoint is None:
        raise IOError("No M2 TensorFlow checkpoint found in {0}".format(train_dir))
    return {"sess": sess, "x": x, "model": model, "predictions": predictions, "checkpoint": checkpoint}


def compute_detector_metrics(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    filtered_clean_pred: np.ndarray,
    filtered_adv_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute detector counts and derived percent metrics."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    clean_pred = np.asarray(clean_pred, dtype=np.int64)
    adv_pred = np.asarray(adv_pred, dtype=np.int64)
    filtered_clean_pred = np.asarray(filtered_clean_pred, dtype=np.int64)
    filtered_adv_pred = np.asarray(filtered_adv_pred, dtype=np.int64)

    clean_correct = clean_pred == true_labels
    clean_wrong = ~clean_correct
    attack_failed = clean_correct & (adv_pred == true_labels)
    effectual = clean_correct & (adv_pred != true_labels)
    detected = filtered_adv_pred != adv_pred
    true_positive = effectual & detected
    false_negative = effectual & ~detected
    false_positive = filtered_clean_pred != clean_pred
    recovered_true_positive = true_positive & (filtered_adv_pred == true_labels)

    tp = int(np.sum(true_positive))
    fn = int(np.sum(false_negative))
    fp = int(np.sum(false_positive))
    rtp = int(np.sum(recovered_true_positive))
    failed = int(np.sum(attack_failed))
    clean_wrong_count = int(np.sum(clean_wrong))

    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    rtp_percent = 100.0 * rtp / float(tp) if tp else 0.0

    return {
        "n_total": int(len(true_labels)),
        "#F": failed,
        "F": failed,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "RTP": rtp,
        "RTP_percent": float(rtp_percent),
        "recall": float(recall * 100.0),
        "precision": float(precision * 100.0),
        "f1": float(f1 * 100.0),
        "n_clean_wrong": clean_wrong_count,
        "n_attack_failed": failed,
    }


def evaluate_adversarial_file(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    adversarial_path: Path,
    filter_fn: FilterFn,
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate one saved adversarial array."""
    if not adversarial_path.exists():
        raise IOError("Adversarial examples not found: {0}".format(adversarial_path))

    adv_images = np.load(str(adversarial_path)).astype(np.float32)
    if adv_images.ndim != 4 or adv_images.shape[1:] != (28, 28, 1):
        raise ValueError("Expected adversarial array shape (N, 28, 28, 1).")
    sample_count = min(len(clean_images), len(adv_images), len(labels))
    clean = np.asarray(clean_images[:sample_count], dtype=np.float32)
    adv = np.asarray(adv_images[:sample_count], dtype=np.float32)
    y_true = label_to_int(labels[:sample_count])

    sess = graph["sess"]
    x = graph["x"]
    predictions = graph["predictions"]
    clean_pred = predict_labels(sess, x, predictions, clean, batch_size)
    adv_pred = predict_labels(sess, x, predictions, adv, batch_size)
    filtered_clean = apply_filter_batch(filter_fn, clean)
    filtered_adv = apply_filter_batch(filter_fn, adv)
    filtered_clean_pred = predict_labels(sess, x, predictions, filtered_clean, batch_size)
    filtered_adv_pred = predict_labels(sess, x, predictions, filtered_adv, batch_size)

    return compute_detector_metrics(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write detector results CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "attack",
        "norm",
        "kappa",
        "dataset",
        "model",
        "n_total",
        "F",
        "#F",
        "TP",
        "FN",
        "FP",
        "RTP",
        "RTP_percent",
        "recall",
        "precision",
        "f1",
        "n_clean_wrong",
        "n_attack_failed",
        "filter_name",
        "filter",
        "adversarial_examples_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def write_summary_md(path: Path, title: str, rows: List[Dict[str, Any]]) -> Path:
    """Write detector summary Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(value: Any, decimals: Optional[int] = None) -> str:
        """Format a Markdown table cell."""
        if value == "" or value is None:
            return ""
        if decimals is None:
            return str(value)
        return "{0:.{1}f}".format(float(value), decimals)

    lines = [
        "# {0}".format(title),
        "",
        "| attack | norm | kappa | n_total | #F | TP | FN | FP | RTP | RTP% | recall | precision | f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} |".format(
                cell(row.get("attack", "")),
                cell(row.get("norm", "")),
                cell(row.get("kappa", "")),
                cell(row.get("n_total", "")),
                cell(row.get("#F", "")),
                cell(row.get("TP", "")),
                cell(row.get("FN", "")),
                cell(row.get("FP", "")),
                cell(row.get("RTP", "")),
                cell(row.get("RTP_percent", ""), 2),
                cell(row.get("recall", ""), 2),
                cell(row.get("precision", ""), 2),
                cell(row.get("f1", ""), 2),
            )
        )
    lines.extend(
        [
            "",
            "Metrics are percentages on the 0-100 scale. Clean errors and attack failures are recorded separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate_cw_l2(args: argparse.Namespace, graph: Dict[str, Any], clean_images: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluate all requested CW L2 kappas."""
    filter_fn = select_filter(args.filter)
    rows = []
    for kappa in args.kappas:
        adv_path = M2_CW_DIR / "cw_l2" / "kappa_{0}".format(format_kappa(kappa)) / "adversarial_examples.npy"
        metrics = evaluate_adversarial_file(
            graph=graph,
            clean_images=clean_images,
            labels=labels,
            adversarial_path=adv_path,
            filter_fn=filter_fn,
            batch_size=args.batch_size,
        )
        row = OrderedDict()
        row.update(
            {
                "row_id": "cw_l2_kappa_{0}".format(format_kappa(kappa)),
                "attack": "CW",
                "norm": "L2",
                "kappa": float(kappa),
                "dataset": "mnist",
                "model": "M2",
            }
        )
        row.update(metrics)
        row.update(
            {
                "filter_name": args.filter,
                "filter": args.filter,
                "adversarial_examples_path": str(adv_path),
                "notes": "Last 1000 MNIST digits by default; clean wrong images excluded from TP/FN/#F.",
            }
        )
        rows.append(row)
        print("cw_l2 kappa={0:g} TP={1} FN={2} FP={3}".format(kappa, row["TP"], row["FN"], row["FP"]))
    return rows


def evaluate_cw_linf(args: argparse.Namespace, graph: Dict[str, Any], clean_images: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluate CW Linf if adversarial examples exist."""
    adv_path = M2_CW_DIR / "cw_linf" / "adversarial_examples.npy"
    if not adv_path.exists():
        return [
            {
                "row_id": "cw_linf",
                "attack": "CW",
                "norm": "Linf",
                "kappa": "",
                "dataset": "mnist",
                "model": "M2",
                "n_total": 0,
                "F": "",
                "#F": "",
                "TP": "",
                "FN": "",
                "FP": "",
                "RTP": "",
                "RTP_percent": "",
                "recall": "",
                "precision": "",
                "f1": "",
                "filter_name": args.filter,
                "filter": args.filter,
                "adversarial_examples_path": str(adv_path),
                "notes": "not_executed: CW Linf adversarial examples were not found.",
            }
        ]
    filter_fn = select_filter(args.filter)
    metrics = evaluate_adversarial_file(
        graph=graph,
        clean_images=clean_images,
        labels=labels,
        adversarial_path=adv_path,
        filter_fn=filter_fn,
        batch_size=args.batch_size,
    )
    row = {
        "row_id": "cw_linf",
        "attack": "CW",
        "norm": "Linf",
        "kappa": "",
        "dataset": "mnist",
        "model": "M2",
        "filter_name": args.filter,
        "filter": args.filter,
        "adversarial_examples_path": str(adv_path),
        "notes": "CW Linf evaluated from saved adversarial examples.",
    }
    row.update(metrics)
    return [row]


def main() -> int:
    """Run detector evaluation for saved CW examples."""
    args = build_parser().parse_args()
    args.kappas = parse_float_list(args.kappas)

    end_index = args.start_index + args.samples
    _, _, X_test, Y_test = load_mnist_data(test_start=args.start_index, test_end=end_index)
    graph = create_restored_m2_graph(args.train_dir)
    output_dir = Path(args.output_dir)

    try:
        if args.attack in ("cw_l2", "all"):
            rows = evaluate_cw_l2(args, graph, X_test, Y_test)
            csv_path = write_results_csv(output_dir / "cw_l2_detector_results.csv", rows)
            md_path = write_summary_md(output_dir / "cw_l2_detector_summary.md", "MNIST M2 CW L2 Detector", rows)
            print("cw_l2_results_csv={0}".format(csv_path))
            print("cw_l2_summary_md={0}".format(md_path))

        if args.attack in ("cw_linf", "all"):
            rows = evaluate_cw_linf(args, graph, X_test, Y_test)
            csv_path = write_results_csv(output_dir / "cw_linf_detector_results.csv", rows)
            md_path = write_summary_md(output_dir / "cw_linf_detector_summary.md", "MNIST M2 CW Linf Detector", rows)
            print("cw_linf_results_csv={0}".format(csv_path))
            print("cw_linf_summary_md={0}".format(md_path))
    finally:
        graph["sess"].close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

