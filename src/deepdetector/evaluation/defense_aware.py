"""Defense-aware CW-L2 evaluation for MNIST M2.

This module is intentionally TensorFlow 1.x / legacy Keras friendly because the
MNIST M2 model used by the CW experiments is built with old standalone Keras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from deepdetector.attacks.adaptive_cw_l2 import (
    generate_native_adaptive_cw_l2_attack,
    generate_original_adaptive_cw_l2_attack,
)
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
    l2_distances: List[float] = field(default_factory=list)


def _prepare_tensorflow_legacy_mode() -> Any:
    """Return TensorFlow after forcing TF1 graph mode when possible."""
    import tensorflow as tf

    compat_v1 = getattr(tf, "compat", None)
    compat_v1 = getattr(compat_v1, "v1", None)

    if compat_v1 is not None:
        if hasattr(compat_v1, "disable_eager_execution"):
            compat_v1.disable_eager_execution()
        if hasattr(compat_v1, "disable_v2_behavior"):
            compat_v1.disable_v2_behavior()

    return tf


def _set_keras_inference_phase(sess: Optional[Any] = None) -> None:
    """Force legacy Keras dropout/batch-norm layers to run in inference mode.

    This must run before ``build_mnist_m2_model`` because the old Keras Dropout
    layer creates/uses ``keras_learning_phase`` while the graph is being built.
    """
    try:
        from keras import backend as K

        if sess is not None and hasattr(K, "set_session"):
            K.set_session(sess)

        if hasattr(K, "set_learning_phase"):
            K.set_learning_phase(0)
    except Exception:
        # Keep this best-effort because tests often monkeypatch TensorFlow/Keras.
        pass


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


def rows_to_metrics_json(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


def defense_aware_diagnostics_payload(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an aggregate diagnostics payload for native adaptive CW-L2."""
    summary = {
        "samples": int(len(records)),
        "total_candidates": 0,
        "adversarial_candidates": 0,
        "detected_adversarial_candidates": 0,
        "evading_adversarial_candidates": 0,
        "samples_with_adversarial_candidates": 0,
        "samples_with_detected_adversarial_candidates": 0,
        "samples_with_evading_adversarial_candidates": 0,
        "final_successes": 0,
        "final_failures": 0,
    }
    for record in records:
        total = int(record.get("total_candidates", 0))
        adversarial = int(record.get("adversarial_candidates", 0))
        detected = int(record.get("detected_adversarial_candidates", 0))
        evading = int(record.get("evading_adversarial_candidates", 0))
        summary["total_candidates"] += total
        summary["adversarial_candidates"] += adversarial
        summary["detected_adversarial_candidates"] += detected
        summary["evading_adversarial_candidates"] += evading
        if adversarial > 0:
            summary["samples_with_adversarial_candidates"] += 1
        if detected > 0:
            summary["samples_with_detected_adversarial_candidates"] += 1
        if evading > 0:
            summary["samples_with_evading_adversarial_candidates"] += 1
        if bool(record.get("final_success", False)):
            summary["final_successes"] += 1
        elif "final_success" in record:
            summary["final_failures"] += 1

    return {
        "summary": summary,
        "samples": [_json_safe_value(record) for record in records],
    }


def evaluate_defense_aware_arrays(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    predict_fn: PredictFn,
    defense_unaware_attack_fn: AttackFn,
    defense_aware_attack_fn: AttackFn,
    transform_fn: TransformFn,
    diagnostic_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
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

            blind_transformed = np.asarray(
                transform_fn(blind_adv),
                dtype=np.float32,
            ).reshape(clean_image.shape)
            blind_transformed_pred = _predict_one(predict_fn, blind_transformed)

            if blind_pred != blind_transformed_pred:
                blind_counts.detected += 1
            else:
                blind_counts.undetected += 1

        diagnostics_start = len(diagnostic_records) if diagnostic_records is not None else 0
        adaptive_adv = np.asarray(
            defense_aware_attack_fn(clean_image, true_label, clean_pred),
            dtype=np.float32,
        ).reshape(clean_image.shape)
        adaptive_pred = _predict_one(predict_fn, adaptive_adv)
        adaptive_transformed = np.asarray(
            transform_fn(adaptive_adv),
            dtype=np.float32,
        ).reshape(clean_image.shape)
        adaptive_transformed_pred = _predict_one(predict_fn, adaptive_transformed)

        if adaptive_pred != true_label and adaptive_pred == adaptive_transformed_pred:
            adaptive_counts.success += 1
            adaptive_counts.undetected += 1
            final_l2 = _l2_distance(adaptive_adv, clean_image)
            adaptive_counts.l2_distances.append(final_l2)
            final_success = True
        else:
            adaptive_counts.failures += 1
            final_l2 = 0.0
            final_success = False

        if diagnostic_records is not None and len(diagnostic_records) > diagnostics_start:
            record = diagnostic_records[diagnostics_start]
            record["final_pred"] = int(adaptive_pred)
            record["final_transformed_pred"] = int(adaptive_transformed_pred)
            record["final_success"] = bool(final_success)
            record["final_l2"] = float(final_l2)

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
    tf = _prepare_tensorflow_legacy_mode()
    sess = create_tf_session()
    graph = getattr(sess, "graph", None)

    def build_on_active_graph() -> Dict[str, Any]:
        _set_keras_inference_phase(sess)

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
        setattr(model, "predictions", predictions)
        setattr(model, "logits_tensor", predictions)
        setattr(model, "num_labels", 10)

        return {
            "sess": sess,
            "x": x_placeholder,
            "model": model,
            "predictions": predictions,
            "checkpoint": checkpoint,
        }

    if graph is not None and hasattr(graph, "as_default"):
        with graph.as_default():
            return build_on_active_graph()

    return build_on_active_graph()


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
    diagnostic_records: Optional[List[Dict[str, Any]]] = None,
) -> AttackFn:
    kwargs = _attack_kwargs(attack_config)
    kwargs.setdefault("clip_min", 0.0)
    kwargs.setdefault("clip_max", 1.0)

    attack_type = str(attack_config.get("type", "native_adaptive_cw_l2")).strip().lower()
    if attack_type == "native_adaptive_cw_l2":
        attack_generator = generate_native_adaptive_cw_l2_attack
    elif attack_type == "original_adaptive_cw_l2":
        attack_generator = generate_original_adaptive_cw_l2_attack
    else:
        raise ValueError("Unsupported defense-aware attack type: {0}".format(attack_type))

    def attack(image: np.ndarray, true_label: int, clean_pred: int) -> np.ndarray:
        call_kwargs = dict(kwargs)

        if attack_type == "native_adaptive_cw_l2":
            call_kwargs["transform_fn"] = transform_fn
            call_kwargs["predict_fn"] = predict_fn
        if attack_type == "native_adaptive_cw_l2" and diagnostic_records is not None:
            call_kwargs["diagnostics"] = diagnostic_records
            call_kwargs["diagnostic_context"] = {
                "valid_index": len(diagnostic_records),
                "clean_pred": int(clean_pred),
            }

        adversarial = attack_generator(
            model=graph["model"],
            images=_as_batch(image),
            labels=np.asarray([true_label], dtype=np.int32),
            **call_kwargs,
        )
        return np.asarray(adversarial[0], dtype=np.float32)

    return attack


def save_defense_aware_outputs(
    *,
    rows: List[Dict[str, Any]],
    output_dir: Path,
    csv_name: str = "metrics.csv",
    json_name: str = "metrics.json",
) -> Dict[str, Path]:
    """Write only the official defense-aware output artifacts."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / csv_name, rows, DEFENSE_AWARE_SCHEMA)
    json_path = write_metrics_json(output_path / json_name, rows_to_metrics_json(rows))
    return {"csv": csv_path, "json": json_path}


def save_defense_aware_diagnostics(
    *,
    records: List[Dict[str, Any]],
    output_dir: Path,
    json_name: str = "diagnostics.json",
) -> Path:
    """Write optional native adaptive CW-L2 diagnostics."""
    output_path = ensure_dir(output_dir)
    return write_metrics_json(
        output_path / json_name,
        defense_aware_diagnostics_payload(records),
    )


def run_defense_aware_evaluation(
    config: Dict[str, Any],
    graph: Optional[Dict[str, Any]] = None,
    diagnostic_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
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
            diagnostic_records=diagnostic_records,
        ),
        transform_fn=transform_fn,
        diagnostic_records=diagnostic_records,
    )
    return rows


def run_defense_aware_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run defense-aware evaluation and write official outputs."""
    diagnostics_config = dict(config.get("diagnostics", {}))
    diagnostics_enabled = bool(diagnostics_config.get("enabled", False))
    diagnostic_records: Optional[List[Dict[str, Any]]] = [] if diagnostics_enabled else None
    rows = run_defense_aware_evaluation(
        config,
        diagnostic_records=diagnostic_records,
    )
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
    if diagnostics_enabled and diagnostic_records is not None:
        save_defense_aware_diagnostics(
            records=diagnostic_records,
            output_dir=output_dir,
            json_name=str(diagnostics_config.get("json", "diagnostics.json")),
        )
    return rows
