"""Validate M1/FGSM examples after adaptive noise reduction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np

from .config import MnistExperimentConfig
from .detection import DetectionDecision
from .evaluation import DetectorCounts
from .noise_reduction import adaptive_reduce


@dataclass(frozen=True)
class FgsmFilterValidationResult:
    """Summary of the M1/FGSM adaptive filtering validation."""

    input_npz: Path
    output_npz: Path
    metrics_json: Path
    samples: int
    restored_count: int
    detected_count: int


def _default_input_path(config: MnistExperimentConfig) -> Path:
    return config.paths.outputs_root / "adversarial" / "fgsm" / "examples.npz"


def _default_output_dir(config: MnistExperimentConfig) -> Path:
    return config.paths.outputs_root / "validation" / "fgsm_m1_filter"


def _predict_labels(sess, x, predictions, images: np.ndarray, batch_size: int) -> np.ndarray:
    labels = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        labels.append(sess.run(predictions, feed_dict={x: batch}))
    return np.concatenate(labels).astype(np.int64)


def _filter_images(images: np.ndarray, config: MnistExperimentConfig) -> np.ndarray:
    filtered = [
        adaptive_reduce(image.reshape(28, 28), config.detection, offset=0.0).reshape(28, 28, 1)
        for image in images
    ]
    return np.asarray(filtered, dtype=np.float32)


def _summarize_decisions(
    true_labels: np.ndarray,
    original_labels: np.ndarray,
    adversarial_labels: np.ndarray,
    filtered_original_labels: np.ndarray,
    filtered_adversarial_labels: np.ndarray,
) -> Dict[str, object]:
    counts = DetectorCounts()
    evaluated_indices = []
    skipped_original_wrong = []
    skipped_attack_failed = []
    restored_indices = []

    for index in range(len(true_labels)):
        true_label = int(true_labels[index])
        original_label = int(original_labels[index])
        adversarial_label = int(adversarial_labels[index])

        if original_label != true_label:
            counts.original_classified_wrong_number += 1
            skipped_original_wrong.append(index)
            continue

        if adversarial_label == original_label:
            counts.disturbed_failure_number += 1
            skipped_attack_failed.append(index)
            continue

        filtered_original_label = int(filtered_original_labels[index])
        filtered_adversarial_label = int(filtered_adversarial_labels[index])
        decision = DetectionDecision(
            original_label=original_label,
            adversarial_label=adversarial_label,
            filtered_original_label=filtered_original_label,
            filtered_adversarial_label=filtered_adversarial_label,
            true_label=true_label,
        )
        counts.update_detection(decision)
        evaluated_indices.append(index)
        if decision.adversarial_restored:
            restored_indices.append(index)

    clean_filter_preserved_count = int(np.sum(original_labels == filtered_original_labels))
    adversarial_filter_matches_original_count = int(np.sum(filtered_adversarial_labels == original_labels))
    adversarial_filter_changes_prediction_count = int(np.sum(filtered_adversarial_labels != adversarial_labels))
    sample_count = int(len(true_labels))
    effective_attack_count = int(counts.test_number)

    metrics = counts.as_dict()
    metrics.update(
        {
            "samples": sample_count,
            "effective_attack_count": effective_attack_count,
            "effective_attack_restored_count": int(counts.ttp),
            "effective_attack_restored_rate": (
                counts.ttp / effective_attack_count if effective_attack_count else 0.0
            ),
            "clean_filter_preserved_count": clean_filter_preserved_count,
            "clean_filter_preserved_rate": clean_filter_preserved_count / sample_count if sample_count else 0.0,
            "adversarial_filter_matches_original_count": adversarial_filter_matches_original_count,
            "adversarial_filter_matches_original_rate": (
                adversarial_filter_matches_original_count / sample_count if sample_count else 0.0
            ),
            "adversarial_filter_changes_prediction_count": adversarial_filter_changes_prediction_count,
            "adversarial_filter_changes_prediction_rate": (
                adversarial_filter_changes_prediction_count / sample_count if sample_count else 0.0
            ),
            "evaluated_indices": evaluated_indices,
            "restored_indices": restored_indices,
            "skipped_original_wrong_indices": skipped_original_wrong,
            "skipped_attack_failed_indices": skipped_attack_failed,
        }
    )
    return metrics


def validate_fgsm_filter(
    config: Optional[MnistExperimentConfig] = None,
    input_npz: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
) -> FgsmFilterValidationResult:
    """Compare M1 predictions before and after filtering FGSM images."""

    started_at = time.time()
    cfg = config or MnistExperimentConfig()
    source_npz = input_npz or _default_input_path(cfg)
    target_dir = output_dir or _default_output_dir(cfg)
    prediction_batch_size = batch_size or cfg.m1.batch_size

    if not source_npz.exists():
        raise FileNotFoundError(
            "FGSM artifact not found at {}. Run scripts/mnist/generate_adversarial.py fgsm first.".format(
                source_npz
            )
        )

    import keras
    from keras import backend as keras_backend
    import tensorflow as tf
    from cleverhans.utils_keras import cnn_model

    print("[FGSM_FILTER] input_npz={}".format(source_npz), flush=True)
    print("[FGSM_FILTER] checkpoint_dir={}".format(cfg.paths.m1_checkpoint_dir), flush=True)

    artifact = np.load(str(source_npz))
    clean_images = artifact["clean_images"].astype(np.float32)
    adversarial_images = artifact["adversarial_images"].astype(np.float32)
    labels = artifact["labels"].astype(np.float32)
    true_labels = labels.argmax(axis=1).astype(np.int64)

    keras_backend.set_image_dim_ordering("tf")
    sess = tf.Session()
    keras_backend.set_session(sess)

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    model = cnn_model()
    logits = model(x)
    predictions = tf.argmax(logits, axis=1)

    checkpoint = tf.train.get_checkpoint_state(str(cfg.paths.m1_checkpoint_dir))
    checkpoint_path = None if checkpoint is None else checkpoint.model_checkpoint_path
    if checkpoint_path is None:
        raise FileNotFoundError(
            "M1 checkpoint not found in {}. Run scripts/mnist/run_m1.py first.".format(
                cfg.paths.m1_checkpoint_dir
            )
        )

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print("[FGSM_FILTER] loaded_checkpoint={}".format(checkpoint_path), flush=True)

    filtered_clean_images = _filter_images(clean_images, cfg)
    filtered_adversarial_images = _filter_images(adversarial_images, cfg)

    original_labels = _predict_labels(sess, x, predictions, clean_images, prediction_batch_size)
    adversarial_labels = _predict_labels(sess, x, predictions, adversarial_images, prediction_batch_size)
    filtered_original_labels = _predict_labels(sess, x, predictions, filtered_clean_images, prediction_batch_size)
    filtered_adversarial_labels = _predict_labels(
        sess,
        x,
        predictions,
        filtered_adversarial_images,
        prediction_batch_size,
    )

    metrics = _summarize_decisions(
        true_labels,
        original_labels,
        adversarial_labels,
        filtered_original_labels,
        filtered_adversarial_labels,
    )
    metrics.update(
        {
            "category": "faithful_reproduction",
            "model": "M1 CleverHans cnn_model",
            "attack": "CleverHans FastGradientMethod",
            "filter": "DeepDetector adaptive noise reduction",
            "checkpoint_path": checkpoint_path,
            "input_npz": str(source_npz),
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    output_npz = target_dir / "filtered_examples.npz"
    metrics_json = target_dir / "metrics.json"
    np.savez_compressed(
        str(output_npz),
        clean_images=clean_images,
        adversarial_images=adversarial_images,
        filtered_clean_images=filtered_clean_images,
        filtered_adversarial_images=filtered_adversarial_images,
        labels=labels,
        original_predictions=original_labels,
        adversarial_predictions=adversarial_labels,
        filtered_original_predictions=filtered_original_labels,
        filtered_adversarial_predictions=filtered_adversarial_labels,
    )
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("[FGSM_FILTER] saved_filtered={}".format(output_npz), flush=True)
    print("[FGSM_FILTER] saved_metrics={}".format(metrics_json), flush=True)
    print(
        "[FGSM_FILTER] evaluated={} tp={} fn={} fp={} ttp={} recall={:.4f} precision={:.4f}".format(
            metrics["test_number"],
            metrics["tp"],
            metrics["fn"],
            metrics["fp"],
            metrics["ttp"],
            metrics["recall"],
            metrics["precision"],
        ),
        flush=True,
    )
    print(
        "[FGSM_FILTER] filtered_adversarial_matches_original={}/{}".format(
            metrics["adversarial_filter_matches_original_count"],
            metrics["samples"],
        ),
        flush=True,
    )

    return FgsmFilterValidationResult(
        input_npz=source_npz,
        output_npz=output_npz,
        metrics_json=metrics_json,
        samples=int(metrics["samples"]),
        restored_count=int(metrics["ttp"]),
        detected_count=int(metrics["tp"]),
    )
