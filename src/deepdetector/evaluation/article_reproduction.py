"""Orchestration helpers for reproducing MNIST tables from DeepDetector.

This module intentionally reuses the existing data loader, MNIST model,
FGSM attack, quantization filters, and metric logic. It only adds the
article-specific experiment splits, interval mappings, table formatting,
and cross-filter orchestration needed to reproduce the reported tables.
"""

from __future__ import print_function

import csv
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deepdetector.attacks.fgsm import generate_fgsm_examples
from deepdetector.data.mnist import load_mnist_data
from deepdetector.filters.adaptive_quantization import entropy_based_quantization
from deepdetector.filters.entropy import one_d_entropy
from deepdetector.filters.mean_filters import cross_mean_filter
from deepdetector.filters.quantization import (
    nonuniform_quantization,
    scalar_quantization,
)
from deepdetector.models.mnist_cnn import (
    build_mnist_model,
    create_tf_session,
    load_mnist_model,
)


FilterFn = Callable[[np.ndarray], np.ndarray]
ARTICLE_OUTPUT_DIR = os.path.join("results", "mnist", "article_reproduction")

INTERVAL_TO_SIZE = {
    2: 128,
    3: 85,
    4: 64,
    5: 51,
    6: 43,
    7: 37,
    8: 32,
    9: 28,
    10: 26,
}


def interval_size(number_of_intervals: int) -> int:
    """Return the scalar quantization interval size used by the article."""
    try:
        return INTERVAL_TO_SIZE[int(number_of_intervals)]
    except KeyError:
        raise ValueError("Unsupported interval count: {0}".format(number_of_intervals))


def scalar_filter_for_intervals(number_of_intervals: int) -> FilterFn:
    """Build a uniform scalar quantization filter for an interval count."""
    interval = interval_size(number_of_intervals)
    return lambda image: scalar_quantization(image, interval=interval, left=True)


def adaptive_quantization_filter(image: np.ndarray) -> np.ndarray:
    """Apply the Table 6 adaptive scalar quantization rule."""
    return entropy_based_quantization(image)[0]


def proposed_detection_filter(image: np.ndarray) -> np.ndarray:
    """Apply the article's final MNIST detection filter.

    Low-entropy images use 2 intervals, mid-entropy images use 4 intervals,
    and high-entropy images use 6 intervals plus the 7x7 cross smoothing
    combination described in Equation 9.
    """
    image_array = np.asarray(image, dtype=np.float32)
    entropy = one_d_entropy(image_array)
    if entropy < 4.0:
        return scalar_quantization(image_array, interval=interval_size(2), left=True)
    if entropy < 5.0:
        return scalar_quantization(image_array, interval=interval_size(4), left=True)

    quantized = scalar_quantization(image_array, interval=interval_size(6), left=True)
    smoothed = cross_mean_filter(quantized, radius=3)
    use_quantized = np.abs(quantized - image_array) <= np.abs(smoothed - image_array)
    return np.where(use_quantized, quantized, smoothed).astype(np.float32)


def ensure_output_dir(output_dir: str = ARTICLE_OUTPUT_DIR) -> str:
    """Create the article reproduction output directory."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def label_to_int(labels: np.ndarray) -> np.ndarray:
    """Convert one-hot or integer labels to integer classes."""
    label_array = np.asarray(labels)
    if label_array.ndim == 1:
        return label_array.astype(np.int64)
    return np.argmax(label_array, axis=1).astype(np.int64)


def create_restored_mnist_graph(train_dir: str) -> Dict[str, Any]:
    """Create the TF1 graph, restore the clean MNIST checkpoint, and return handles."""
    import tensorflow as tf

    sess = create_tf_session()
    x_placeholder = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, 28, 28, 1),
        name="x",
    )
    model, predictions = build_mnist_model(x_placeholder)
    checkpoint = load_mnist_model(sess, train_dir)
    if checkpoint is None:
        raise IOError("No TensorFlow checkpoint found in {0}".format(train_dir))

    return {
        "sess": sess,
        "x": x_placeholder,
        "model": model,
        "predictions": predictions,
        "checkpoint": checkpoint,
    }


def load_mnist_test_slice(start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one slice of the MNIST test set."""
    _, _, x_test, y_test = load_mnist_data(test_start=start, test_end=end)
    return np.asarray(x_test, dtype=np.float32), np.asarray(y_test)


def predict_labels(
    sess: Any,
    x_placeholder: Any,
    predictions: Any,
    images: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict labels for a batch of MNIST images."""
    def learning_phase_feed() -> Dict[Any, Any]:
        """Return a feed dict for Keras learning phase when needed."""
        try:
            from keras import backend as K
        except Exception:
            return {}
        if not hasattr(K, "learning_phase"):
            return {}
        phase = K.learning_phase()
        if hasattr(phase, "op"):
            return {phase: 0}
        return {}

    feed = learning_phase_feed()
    labels = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        feed_dict = {x_placeholder: batch}
        if feed:
            feed_dict.update(feed)
        probs = sess.run(predictions, feed_dict=feed_dict)
        labels.extend(np.argmax(probs, axis=1).astype(np.int64).tolist())
    return np.asarray(labels, dtype=np.int64)


def apply_filter_batch(filter_fn: FilterFn, images: np.ndarray) -> np.ndarray:
    """Apply a filter to a batch of images."""
    filtered = [filter_fn(image) for image in images]
    return np.asarray(filtered, dtype=np.float32).reshape(images.shape)


def evaluate_filter_predictions(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    filtered_clean_pred: np.ndarray,
    filtered_adv_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute article-style detection counts and derived metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    clean_pred = np.asarray(clean_pred, dtype=np.int64)
    adv_pred = np.asarray(adv_pred, dtype=np.int64)
    filtered_clean_pred = np.asarray(filtered_clean_pred, dtype=np.int64)
    filtered_adv_pred = np.asarray(filtered_adv_pred, dtype=np.int64)

    attack_failed = adv_pred == y_true
    effectual = ~attack_failed
    detected = filtered_adv_pred != adv_pred
    true_positive = effectual & detected
    false_negative = effectual & ~detected
    false_positive = filtered_clean_pred != clean_pred
    recovered_true_positive = true_positive & (filtered_adv_pred == y_true)

    tp = int(np.sum(true_positive))
    fn = int(np.sum(false_negative))
    fp = int(np.sum(false_positive))
    rtp = int(np.sum(recovered_true_positive))
    failed = int(np.sum(attack_failed))

    recall = tp / float(tp + fn) if tp + fn else 0.0
    precision = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    rtp_rate = rtp / float(tp) if tp else 0.0

    return {
        "n_total": int(len(y_true)),
        "clean_errors": int(np.sum(clean_pred != y_true)),
        "F": failed,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "RTP": rtp,
        "RTP_percent": float(rtp_rate * 100.0),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "recall_percent": float(recall * 100.0),
        "precision_percent": float(precision * 100.0),
        "f1_percent": float(f1 * 100.0),
    }


def evaluate_filter_on_images(
    graph: Dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    filter_fn: FilterFn,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Generate FGSM examples and evaluate one detection filter."""
    sess = graph["sess"]
    x_placeholder = graph["x"]
    predictions = graph["predictions"]
    adv_images = generate_fgsm_examples(
        sess=sess,
        model=graph["model"],
        x_placeholder=x_placeholder,
        images=images,
        eps=epsilon,
        clip_min=0.0,
        clip_max=1.0,
    )

    clean_pred = predict_labels(sess, x_placeholder, predictions, images, batch_size)
    adv_pred = predict_labels(sess, x_placeholder, predictions, adv_images, batch_size)
    filtered_clean = apply_filter_batch(filter_fn, images)
    filtered_adv = apply_filter_batch(filter_fn, adv_images)
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
    return evaluate_filter_predictions(
        y_true=label_to_int(labels),
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )


def evaluate_filter_on_existing_adversarial(
    graph: Dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    adv_images: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    filter_fn: FilterFn,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Evaluate one filter when adversarial examples and base predictions exist."""
    sess = graph["sess"]
    x_placeholder = graph["x"]
    predictions = graph["predictions"]
    filtered_clean = apply_filter_batch(filter_fn, images)
    filtered_adv = apply_filter_batch(filter_fn, adv_images)
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
    return evaluate_filter_predictions(
        y_true=label_to_int(labels),
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )


def time_filter_application(filter_fn: FilterFn, images: np.ndarray) -> float:
    """Return elapsed seconds for applying a filter to a batch."""
    start = time.perf_counter()
    apply_filter_batch(filter_fn, images)
    return float(time.perf_counter() - start)


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> str:
    """Write dictionaries to CSV with a stable column order."""
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def format_percent(value: Any) -> str:
    """Format a percent value already on the 0-100 scale."""
    return "{0:.2f}%".format(float(value))


def percent_delta(ours: Any, article: Any) -> float:
    """Return ours minus article, both on the 0-100 scale."""
    return float(ours) - float(article)


def write_markdown_table(
    path: str,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    notes: Optional[Sequence[str]] = None,
) -> str:
    """Write a simple Markdown report with one table."""
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    lines = ["# {0}".format(title), ""]
    lines.append("| {0} |".format(" | ".join(headers)))
    lines.append("| {0} |".format(" | ".join(["---"] * len(headers))))
    for row in rows:
        lines.append("| {0} |".format(" | ".join(str(value) for value in row)))
    if notes:
        lines.append("")
        lines.extend(notes)
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def close_graph(graph: Dict[str, Any]) -> None:
    """Close the TensorFlow session in a graph dictionary."""
    graph["sess"].close()
