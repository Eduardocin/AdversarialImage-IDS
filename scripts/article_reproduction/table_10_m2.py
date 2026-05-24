"""Evaluate MNIST M2 CW detector metrics for saved adversarial examples."""

from __future__ import print_function

import argparse
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "mnist_table_10_m2.yaml"

from deepdetector.data.mnist import load_mnist_data  # noqa: E402
from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    ARTICLE_OUTPUT_DIR,
    adaptive_quantization_filter,
    apply_filter_batch,
    ensure_output_dir,
    format_percent,
    label_to_int,
    predict_labels,
    proposed_detection_filter,
    scalar_filter_for_intervals,
    write_csv,
    write_markdown_table,
)
from deepdetector.models.mnist_cnn import create_tf_session  # noqa: E402
from deepdetector.models.mnist_m2 import build_mnist_m2_model, load_mnist_m2_model  # noqa: E402
from deepdetector.paths import MNIST_M2_CHECKPOINT_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--output-dir", default=None)
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


def format_kappa(kappa: float) -> str:
    """Format kappa for existing CW L2 result directory names."""
    value = float(kappa)
    if value.is_integer():
        return "{0:.1f}".format(value).replace(".", "p")
    return "{0:g}".format(value).replace(".", "p")


def format_attack_model_label(name: str, kappa: Optional[float]) -> str:
    """Format the displayed Attack/Model label without a separate kappa column."""
    if kappa in (None, ""):
        return name
    return "{0} (kappa={1:.1f})".format(name, float(kappa))


def create_restored_m2_graph(train_dir: str) -> Dict[str, Any]:
    """Create and restore the M2 TF1 graph."""
    import tensorflow as tf

    sess = create_tf_session()
    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, 28, 28, 1),
        name="x",
    )
    model, predictions = build_mnist_m2_model(x_placeholder)
    checkpoint = load_mnist_m2_model(sess, train_dir)
    if checkpoint is None:
        raise IOError("No M2 TensorFlow checkpoint found in {0}".format(train_dir))
    return {
        "sess": sess,
        "x": x_placeholder,
        "model": model,
        "predictions": predictions,
        "checkpoint": checkpoint,
    }


def select_filter(name: str):
    """Return the configured detection filter."""
    if name == "final":
        return proposed_detection_filter
    if name == "adaptive":
        return adaptive_quantization_filter
    if name == "scalar":
        return scalar_filter_for_intervals(6)
    raise ValueError("Unsupported filter: {0}".format(name))


def load_mnist_test_slice(start: int, samples: int) -> tuple:
    """Load one MNIST test slice for M2 evaluation."""
    _, _, x_test, y_test = load_mnist_data(test_start=start, test_end=start + samples)
    return np.asarray(x_test, dtype=np.float32), np.asarray(y_test)


def compute_detector_metrics(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    filtered_clean_pred: np.ndarray,
    filtered_adv_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute Table 10 detector counts and percentage metrics."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    clean_pred = np.asarray(clean_pred, dtype=np.int64)
    adv_pred = np.asarray(adv_pred, dtype=np.int64)
    filtered_clean_pred = np.asarray(filtered_clean_pred, dtype=np.int64)
    filtered_adv_pred = np.asarray(filtered_adv_pred, dtype=np.int64)

    clean_correct = clean_pred == true_labels
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
    clean_wrong = int(np.sum(~clean_correct))

    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    rtp_rate = rtp / float(tp) if tp else 0.0

    return {
        "n_total": int(len(true_labels)),
        "F": failed,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "RTP": rtp,
        "RTP_percent": float(rtp_rate * 100.0),
        "recall_percent": float(recall * 100.0),
        "precision_percent": float(precision * 100.0),
        "f1_percent": float(f1 * 100.0),
        "n_clean_wrong": clean_wrong,
    }


def evaluate_adversarial_path(
    graph: Dict[str, Any],
    clean_images: np.ndarray,
    labels: np.ndarray,
    adversarial_path: Path,
    filter_fn: Any,
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
    x_placeholder = graph["x"]
    predictions = graph["predictions"]

    clean_pred = predict_labels(sess, x_placeholder, predictions, clean, batch_size)
    adv_pred = predict_labels(sess, x_placeholder, predictions, adv, batch_size)
    filtered_clean = apply_filter_batch(filter_fn, clean)
    filtered_adv = apply_filter_batch(filter_fn, adv)
    filtered_clean_pred = predict_labels(
        sess,
        x_placeholder,
        predictions,
        filtered_clean,
        batch_size,
    )
    filtered_adv_pred = predict_labels(
        sess,
        x_placeholder,
        predictions,
        filtered_adv,
        batch_size,
    )

    return compute_detector_metrics(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )


def configured_attack_rows(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield concrete attack rows from the YAML config."""
    for attack_config in config.get("attacks", []):
        kappas = attack_config.get("kappas", [None])
        for kappa in kappas:
            if attack_config.get("adversarial_template"):
                kappa_key = format_kappa(float(kappa))
                path = str(attack_config["adversarial_template"]).format(kappa=kappa_key)
            else:
                path = attack_config["adversarial_path"]
            yield {
                "name": attack_config["name"],
                "attack": attack_config["attack"],
                "norm": attack_config["norm"],
                "kappa": kappa,
                "adversarial_path": _resolve_path(path),
            }


def main() -> int:
    """Run Table 10 M2 evaluation using existing adversarial examples."""
    args = build_parser().parse_args()
    config = load_config(_resolve_path(args.config) or DEFAULT_CONFIG)

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    detection_config = config.get("detection", {})
    evaluation_config = config.get("evaluation", {})
    metrics_config = config.get("metrics", {})
    output_config = config.get("output", {})

    output_dir = ensure_output_dir(
        str(
            _resolve_path(args.output_dir or output_config.get("results_dir"))
            or (PROJECT_ROOT / ARTICLE_OUTPUT_DIR)
        )
    )
    train_dir = str(
        _resolve_path(args.train_dir or model_config.get("checkpoint_dir"))
        or MNIST_M2_CHECKPOINT_DIR
    )
    start = int(dataset_config.get("start", 9000))
    samples = int(dataset_config.get("samples", 1000))
    batch_size = int(evaluation_config.get("batch_size", 256))
    filter_name = str(detection_config.get("filter", "final"))
    filter_fn = select_filter(filter_name)

    clean_images, labels = load_mnist_test_slice(start, samples)
    graph = create_restored_m2_graph(train_dir)
    rows: List[Dict[str, Any]] = []

    try:
        for attack_row in configured_attack_rows(config):
            adversarial_path = attack_row["adversarial_path"]
            metrics = evaluate_adversarial_path(
                graph=graph,
                clean_images=clean_images,
                labels=labels,
                adversarial_path=adversarial_path,
                filter_fn=filter_fn,
                batch_size=batch_size,
            )
            kappa = attack_row["kappa"]
            row = {
                "attack_model": attack_row["name"],
                "attack": attack_row["attack"],
                "norm": attack_row["norm"],
                "dataset": dataset_config.get("name", "mnist").upper(),
                "model": model_config.get("name", "M2"),
                "kappa": "" if kappa is None else float(kappa),
                "filter": filter_name,
                "adversarial_examples_path": str(adversarial_path),
                "notes": "Evaluated from saved adversarial examples.",
            }
            row.update(metrics)
            rows.append(row)
    finally:
        graph["sess"].close()

    columns = metrics_config.get(
        "columns",
        [
            "Attack/Model",
            "Dataset",
            "#F",
            "TP",
            "FN",
            "FP",
            "RTP",
            "RTP%",
            "Recall",
            "Precision",
            "F1",
        ],
    )
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "Attack/Model": format_attack_model_label(row["attack_model"], row["kappa"]),
                "Dataset": row["dataset"],
                "#F": row["F"],
                "TP": row["TP"],
                "FN": row["FN"],
                "FP": row["FP"],
                "RTP": row["RTP"],
                "RTP%": format_percent(row["RTP_percent"]),
                "Recall": format_percent(row["recall_percent"]),
                "Precision": format_percent(row["precision_percent"]),
                "F1": format_percent(row["f1_percent"]),
            }
        )

    csv_path = write_csv(
        str(Path(output_dir) / str(output_config.get("csv", "table_10_m2_mnist_cw.csv"))),
        table_rows,
        metrics_config.get("raw_fields", columns),
    )

    md_path = write_markdown_table(
        str(Path(output_dir) / str(output_config.get("markdown", "table_10_m2_mnist_cw.md"))),
        "MNIST M2 CW Detector Metrics",
        columns,
        [[row[column] for column in columns] for row in table_rows],
    )

    print("results_csv={0}".format(csv_path))
    print("results_md={0}".format(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
