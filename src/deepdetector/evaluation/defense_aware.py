"""Defense-aware CW-L2 evaluation for MNIST M2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from deepdetector.attacks.adaptive_cw_l2 import generate_adaptive_cw_l2_attack
from deepdetector.attacks.nn_robust import generate_nn_robust_cw_l2_attack
from deepdetector.evaluation.article_reproduction import (
    label_to_int,
    load_mnist_test_slice,
    predict_labels,
)
from deepdetector.filters.adaptive_noise_reduction import (
    build_final_adaptive_detection_filter,
)
from deepdetector.io.paths import ensure_dir, resolve_project_path
from deepdetector.io.result_writers import write_metrics_csv, write_metrics_json
from deepdetector.models.mnist_cnn import create_tf_session
from deepdetector.models.mnist_m2 import build_mnist_m2_model, load_mnist_m2_model
from deepdetector.paths import MNIST_M2_CHECKPOINT_DIR


DEFENSE_AWARE_SCHEMA = [
    "attack",
    "total_valid",
    "success",
    "detected",
    "undetected",
    "failures",
    "attack_success_rate_percent",
    "detection_rate_percent",
    "evasion_rate_percent",
    "failure_rate_percent",
    "mean_l2",
]


PredictFn = Callable[[np.ndarray], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]
AttackFn = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass
class ScenarioCounts:
    """Accumulate one attack scenario's defense-aware metrics."""

    success: int = 0
    detected: int = 0
    undetected: int = 0
    failures: int = 0
    l2_distances: list[float] = field(default_factory=list)


def _attack_kwargs(attack_config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(attack_config)
    kwargs.pop("name", None)
    kwargs.pop("type", None)
    return kwargs


def _as_batch(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    return image_array.reshape((1,) + tuple(image_array.shape))


def _predict_one(predict_fn: PredictFn, image: np.ndarray) -> int:
    predictions = np.asarray(predict_fn(_as_batch(image)))
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return int(predictions.astype(np.int64).reshape(-1)[0])


def _l2_distance(adversarial: np.ndarray, clean: np.ndarray) -> float:
    delta = np.asarray(adversarial, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
    return float(np.linalg.norm(delta.reshape(-1), ord=2))


def _percent(numerator: int, denominator: int) -> float:
    if int(denominator) == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _row_for_counts(
    *,
    attack: str,
    total_valid: int,
    counts: ScenarioCounts,
) -> Dict[str, Any]:
    mean_l2 = float(np.mean(counts.l2_distances)) if counts.l2_distances else 0.0
    return {
        "attack": attack,
        "total_valid": int(total_valid),
        "success": int(counts.success),
        "detected": int(counts.detected),
        "undetected": int(counts.undetected),
        "failures": int(counts.failures),
        "attack_success_rate_percent": _percent(counts.success, total_valid),
        "detection_rate_percent": _percent(counts.detected, counts.success),
        "evasion_rate_percent": _percent(counts.undetected, counts.success),
        "failure_rate_percent": _percent(counts.failures, total_valid),
        "mean_l2": mean_l2,
    }


def rows_to_metrics_json(rows: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert CSV rows to the metrics.json payload without extra metadata."""
    payload: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        attack_name = str(row["attack"])
        payload[attack_name] = {
            field: row[field]
            for field in DEFENSE_AWARE_SCHEMA
            if field != "attack"
        }
    return payload


def evaluate_defense_aware_arrays(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    predict_fn: PredictFn,
    defense_unaware_attack_fn: AttackFn,
    defense_aware_attack_fn: AttackFn,
    transform_fn: TransformFn,
) -> list[Dict[str, Any]]:
    """Evaluate defense-unaware and defense-aware scenarios over arrays."""
    image_array = np.asarray(images, dtype=np.float32)
    label_ints = label_to_int(np.asarray(labels))
    total_valid = 0
    blind_counts = ScenarioCounts()
    adaptive_counts = ScenarioCounts()

    for index, clean_image in enumerate(image_array):
        true_label = int(label_ints[index])
        clean_pred = _predict_one(predict_fn, clean_image)
        if clean_pred != true_label:
            continue
        total_valid += 1

        blind_adv = np.asarray(
            defense_unaware_attack_fn(clean_image, true_label, clean_pred),
            dtype=np.float32,
        ).reshape(clean_image.shape)
        blind_pred = _predict_one(predict_fn, blind_adv)
        if blind_pred == true_label:
            blind_counts.failures += 1
        else:
            blind_counts.success += 1
            blind_counts.l2_distances.append(_l2_distance(blind_adv, clean_image))
            blind_transformed = np.asarray(transform_fn(blind_adv), dtype=np.float32).reshape(
                clean_image.shape
            )
            blind_transformed_pred = _predict_one(predict_fn, blind_transformed)
            if blind_pred != blind_transformed_pred:
                blind_counts.detected += 1
            else:
                blind_counts.undetected += 1

        adaptive_adv = np.asarray(
            defense_aware_attack_fn(clean_image, true_label, clean_pred),
            dtype=np.float32,
        ).reshape(clean_image.shape)
        adaptive_pred = _predict_one(predict_fn, adaptive_adv)
        adaptive_transformed = np.asarray(transform_fn(adaptive_adv), dtype=np.float32).reshape(
            clean_image.shape
        )
        adaptive_transformed_pred = _predict_one(predict_fn, adaptive_transformed)
        if adaptive_pred != true_label and adaptive_pred == adaptive_transformed_pred:
            adaptive_counts.success += 1
            adaptive_counts.undetected += 1
            adaptive_counts.l2_distances.append(_l2_distance(adaptive_adv, clean_image))
        else:
            adaptive_counts.failures += 1

    return [
        _row_for_counts(
            attack="defense_unaware",
            total_valid=total_valid,
            counts=blind_counts,
        ),
        _row_for_counts(
            attack="defense_aware",
            total_valid=total_valid,
            counts=adaptive_counts,
        ),
    ]


def create_restored_mnist_m2_graph(train_dir: str) -> Dict[str, Any]:
    """Create the TF1 graph, restore MNIST M2, and return graph handles."""
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
        raise IOError("No TensorFlow checkpoint found in {0}".format(train_dir))

    setattr(model, "sess", sess)
    setattr(model, "input_tensor", x_placeholder)
    setattr(model, "num_labels", 10)
    return {
        "sess": sess,
        "x": x_placeholder,
        "model": model,
        "predictions": predictions,
        "checkpoint": checkpoint,
    }


def _checkpoint_dir(config: Dict[str, Any]) -> str:
    checkpoint_dir = resolve_project_path(config.get("model", {}).get("checkpoint_dir"))
    return str(checkpoint_dir or MNIST_M2_CHECKPOINT_DIR)


def _load_images(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dataset_config = config.get("dataset", {})
    if str(dataset_config.get("name", "")).lower() != "mnist":
        raise ValueError("defense_aware requires dataset.name=mnist.")
    start = int(dataset_config.get("start", 9000))
    end = int(dataset_config.get("end", 10000))
    return load_mnist_test_slice(start, end)


def _predict_fn(graph: Dict[str, Any], batch_size: int) -> PredictFn:
    def predict(images: np.ndarray) -> np.ndarray:
        return predict_labels(
            graph["sess"],
            graph["x"],
            graph["predictions"],
            np.asarray(images, dtype=np.float32),
            batch_size=batch_size,
        )

    return predict


def _cw_attack_fn(
    *,
    graph: Dict[str, Any],
    attack_config: Dict[str, Any],
) -> AttackFn:
    kwargs = _attack_kwargs(attack_config)
    kwargs.setdefault("clip_min", 0.0)
    kwargs.setdefault("clip_max", 1.0)

    def attack(image: np.ndarray, true_label: int, clean_pred: int) -> np.ndarray:
        adversarial = generate_nn_robust_cw_l2_attack(
            model=graph["model"],
            images=_as_batch(image),
            labels=np.asarray([true_label], dtype=np.int32),
            **kwargs,
        )
        return np.asarray(adversarial[0], dtype=np.float32)

    return attack


def _adaptive_attack_fn(
    *,
    graph: Dict[str, Any],
    attack_config: Dict[str, Any],
    transform_fn: TransformFn,
    predict_fn: PredictFn,
) -> AttackFn:
    kwargs = _attack_kwargs(attack_config)
    kwargs.setdefault("clip_min", 0.0)
    kwargs.setdefault("clip_max", 1.0)

    def attack(image: np.ndarray, true_label: int, clean_pred: int) -> np.ndarray:
        adversarial = generate_adaptive_cw_l2_attack(
            model=graph["model"],
            images=_as_batch(image),
            labels=np.asarray([true_label], dtype=np.int32),
            transform_fn=transform_fn,
            predict_fn=predict_fn,
            **kwargs,
        )
        return np.asarray(adversarial[0], dtype=np.float32)

    return attack


def save_defense_aware_outputs(
    *,
    rows: list[Dict[str, Any]],
    output_dir: Path,
    csv_name: str = "metrics.csv",
    json_name: str = "metrics.json",
) -> Dict[str, Path]:
    """Write only the official defense-aware output artifacts."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / csv_name, rows, DEFENSE_AWARE_SCHEMA)
    json_path = write_metrics_json(output_path / json_name, rows_to_metrics_json(rows))
    return {"csv": csv_path, "json": json_path}


def run_defense_aware_evaluation(
    config: Dict[str, Any],
    graph: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Run the configured defense-aware MNIST M2 evaluation."""
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    images, labels = _load_images(config)
    active_graph = graph or create_restored_mnist_m2_graph(_checkpoint_dir(config))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 256))
    predict_fn = _predict_fn(active_graph, batch_size=batch_size)
    transform_fn = build_final_adaptive_detection_filter(config.get("detector", {}))

    attacks_config = config.get("attacks", {})
    rows = evaluate_defense_aware_arrays(
        images=images,
        labels=labels,
        predict_fn=predict_fn,
        defense_unaware_attack_fn=_cw_attack_fn(
            graph=active_graph,
            attack_config=dict(attacks_config.get("defense_unaware", {})),
        ),
        defense_aware_attack_fn=_adaptive_attack_fn(
            graph=active_graph,
            attack_config=dict(attacks_config.get("defense_aware", {})),
            transform_fn=transform_fn,
            predict_fn=predict_fn,
        ),
        transform_fn=transform_fn,
    )
    return rows


def run_defense_aware_experiment(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Run defense-aware evaluation and write official outputs."""
    rows = run_defense_aware_evaluation(config)
    output_config = config.get("output", {})
    configured_dir = output_config.get("dir") or config.get("output_dir")
    output_dir = resolve_project_path(configured_dir)
    if output_dir is None:
        raise ValueError("defense_aware must define output.dir or output_dir.")
    save_defense_aware_outputs(
        rows=rows,
        output_dir=output_dir,
        csv_name=str(output_config.get("csv", "metrics.csv")),
        json_name=str(output_config.get("json", "metrics.json")),
    )
    return rows
