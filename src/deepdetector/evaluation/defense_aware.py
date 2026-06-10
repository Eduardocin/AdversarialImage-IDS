"""Defense-aware CW-L2 evaluation for MNIST M2."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from deepdetector.attacks.registry import generate_attack
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


logger = logging.getLogger(__name__)

ARTICLE_AWARE_SUCCESS_RATE_PERCENT = 67.37

DEFENSE_AWARE_SCHEMA = [
    "attack",
    "total_valid",
    "success",
    "failures",
    "success_rate_percent",
    "failure_rate_percent",
    "mean_l2",
    "detected",
    "undetected",
    "detection_rate_percent",
    "evasion_rate_percent",
    "article_success_rate_percent",
    "delta_pp",
]

SAMPLE_DIAGNOSTIC_SCHEMA = [
    "sample_index",
    "true_label",
    "clean_pred",
    "unaware_adv_pred",
    "unaware_success",
    "unaware_l2",
    "aware_adv_pred",
    "aware_filtered_adv_pred",
    "aware_success",
    "failure_reason",
    "aware_l2",
    "aware_linf",
    "aware_adv_min",
    "aware_adv_max",
    "aware_filtered_min",
    "aware_filtered_max",
    "internal_adv_pred",
    "internal_filtered_adv_pred",
    "internal_l2",
    "internal_success_defense_aware",
    "internal_external_match",
    "elapsed_seconds",
    "reproduction_mode",
    "use_internal_filter",
    "use_project_transform_fn",
]


PredictFn = Callable[[np.ndarray], np.ndarray]
TransformFn = Callable[[np.ndarray], np.ndarray]
AttackFn = Callable[..., np.ndarray]


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


def _attack_type(attack_config: Dict[str, Any]) -> str:
    attack_name = str(attack_config.get("type") or attack_config.get("name") or "").strip()
    if not attack_name:
        raise ValueError("Defense-aware attack config must define type or name.")
    return attack_name


def _as_batch(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    return image_array.reshape((1,) + tuple(image_array.shape))


def _predict_one(predict_fn: PredictFn, image: np.ndarray) -> int:
    predictions = np.asarray(predict_fn(_as_batch(image)))
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return int(predictions.astype(np.int64).reshape(-1)[0])


def _transform_one(transform_fn: TransformFn, image: np.ndarray) -> np.ndarray:
    transformed = np.asarray(transform_fn(np.asarray(image, dtype=np.float32)), dtype=np.float32)
    return transformed.reshape(image.shape)


def _l2_distance(adversarial: np.ndarray, clean: np.ndarray) -> float:
    delta = np.asarray(adversarial, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
    return float(np.linalg.norm(delta.reshape(-1), ord=2))


def _linf_distance(adversarial: np.ndarray, clean: np.ndarray) -> float:
    delta = np.asarray(adversarial, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
    return float(np.max(np.abs(delta.reshape(-1))))


def _percent(numerator: int, denominator: int) -> float:
    if int(denominator) == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "unknown"
    seconds_float = max(0.0, float(seconds))
    if seconds_float < 60.0:
        return "{0:.1f}s".format(seconds_float)
    minutes = int(seconds_float // 60)
    remainder = seconds_float - minutes * 60
    return "{0}m{1:.0f}s".format(minutes, remainder)


def _row_for_counts(
    *,
    attack: str,
    total_valid: int,
    counts: ScenarioCounts,
    article_reference: float | None = None,
) -> Dict[str, Any]:
    mean_l2 = float(np.mean(counts.l2_distances)) if counts.l2_distances else 0.0
    success_rate = _percent(counts.success, total_valid)
    reference = "" if article_reference is None else float(article_reference)
    delta = "" if article_reference is None else success_rate - float(article_reference)
    return {
        "attack": attack,
        "total_valid": int(total_valid),
        "success": int(counts.success),
        "failures": int(counts.failures),
        "success_rate_percent": success_rate,
        "failure_rate_percent": _percent(counts.failures, total_valid),
        "mean_l2": mean_l2,
        "detected": int(counts.detected),
        "undetected": int(counts.undetected),
        "detection_rate_percent": _percent(counts.detected, counts.success),
        "evasion_rate_percent": _percent(counts.undetected, counts.success),
        "article_success_rate_percent": reference,
        "delta_pp": delta,
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


def _valid_adversarial_output(
    adversarial: np.ndarray,
    expected_shape: tuple[int, ...],
) -> bool:
    adversarial_array = np.asarray(adversarial)
    if adversarial_array.shape != expected_shape:
        return False
    if not np.all(np.isfinite(adversarial_array)):
        return False
    min_value = float(np.min(adversarial_array))
    max_value = float(np.max(adversarial_array))
    return min_value >= -1e-6 and max_value <= 1.0 + 1e-6


def _range_bounds(image: np.ndarray | None) -> tuple[float, float]:
    if image is None:
        return float("nan"), float("nan")
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0 or not np.all(np.isfinite(image_array)):
        return float("nan"), float("nan")
    return float(np.min(image_array)), float(np.max(image_array))


def _failure_reason(
    *,
    valid_output: bool,
    true_label: int,
    adv_pred: int,
    filtered_adv_pred: int,
) -> str:
    if not valid_output:
        return "invalid_output"
    if int(adv_pred) == int(true_label):
        return "attack_failed"
    if int(filtered_adv_pred) != int(adv_pred):
        return "detected"
    return ""


def _latest_internal_diagnostic(
    diagnostics: list[Dict[str, Any]],
) -> Dict[str, Any]:
    if not diagnostics:
        return {}
    return dict(diagnostics[-1])


def _internal_external_match(
    internal: Dict[str, Any],
    adv_pred: int,
    filtered_adv_pred: int,
) -> bool | str:
    if not internal:
        return ""
    internal_adv = internal.get("internal_adv_pred")
    internal_filtered = internal.get("internal_filtered_adv_pred")
    if internal_adv in (None, "") or internal_filtered in (None, ""):
        return ""
    return int(internal_adv) == int(adv_pred) and int(internal_filtered) == int(
        filtered_adv_pred
    )


def _build_attack_fn(
    *,
    graph: Dict[str, Any],
    attack_config: Dict[str, Any],
    predict_fn: PredictFn | None = None,
    transform_fn: TransformFn | None = None,
    diagnostics: Any = None,
) -> AttackFn:
    attack_type = _attack_type(attack_config)
    kwargs = _attack_kwargs(attack_config)

    def attack(
        image: np.ndarray,
        true_label: int,
        clean_pred: int,
        sample_index: int | None = None,
        internal_diagnostics: list[Dict[str, Any]] | None = None,
    ) -> np.ndarray:
        diagnostic_sink = internal_diagnostics if internal_diagnostics is not None else diagnostics
        result = generate_attack(
            attack_type,
            graph=graph,
            model=graph.get("model"),
            images=_as_batch(image),
            labels=np.asarray([true_label], dtype=np.int32),
            predict_fn=predict_fn,
            transform_fn=transform_fn,
            diagnostics=diagnostic_sink,
            diagnostic_context={
                "sample_index": sample_index,
                "true_label": int(true_label),
                "clean_pred": int(clean_pred),
            },
            **kwargs,
        )
        return np.asarray(result[0], dtype=np.float32)

    return attack


def _append_sample_diagnostic(
    *,
    sample_diagnostics: list[Dict[str, Any]] | None,
    sample_index: int,
    true_label: int,
    clean_pred: int,
    unaware_adv_pred: int,
    unaware_success: bool,
    unaware_l2: float,
    aware_adv_pred: int,
    aware_filtered_adv_pred: int,
    aware_success: bool,
    failure_reason: str,
    aware_l2: float,
    aware_linf: float,
    aware_adv: np.ndarray | None,
    aware_filtered: np.ndarray | None,
    internal: Dict[str, Any],
    elapsed_seconds: float,
    reproduction_config: Dict[str, Any],
) -> None:
    if sample_diagnostics is None:
        return
    adv_min, adv_max = _range_bounds(aware_adv)
    filtered_min, filtered_max = _range_bounds(aware_filtered)
    sample_diagnostics.append(
        {
            "sample_index": int(sample_index),
            "true_label": int(true_label),
            "clean_pred": int(clean_pred),
            "unaware_adv_pred": int(unaware_adv_pred),
            "unaware_success": bool(unaware_success),
            "unaware_l2": float(unaware_l2),
            "aware_adv_pred": int(aware_adv_pred),
            "aware_filtered_adv_pred": int(aware_filtered_adv_pred),
            "aware_success": bool(aware_success),
            "failure_reason": str(failure_reason),
            "aware_l2": float(aware_l2),
            "aware_linf": float(aware_linf),
            "aware_adv_min": adv_min,
            "aware_adv_max": adv_max,
            "aware_filtered_min": filtered_min,
            "aware_filtered_max": filtered_max,
            "internal_adv_pred": internal.get("internal_adv_pred", ""),
            "internal_filtered_adv_pred": internal.get("internal_filtered_adv_pred", ""),
            "internal_l2": internal.get("internal_l2", ""),
            "internal_success_defense_aware": internal.get(
                "internal_success_defense_aware",
                "",
            ),
            "internal_external_match": _internal_external_match(
                internal,
                aware_adv_pred,
                aware_filtered_adv_pred,
            ),
            "elapsed_seconds": float(elapsed_seconds),
            "reproduction_mode": str(reproduction_config.get("mode", "")),
            "use_internal_filter": bool(reproduction_config.get("use_internal_filter", "")),
            "use_project_transform_fn": bool(
                reproduction_config.get("use_project_transform_fn", "")
            ),
        }
    )


def _log_aware_sample(
    *,
    success: bool,
    count: int,
    limit: int,
    sample_index: int,
    true_label: int,
    clean_pred: int,
    adv_pred: int,
    filtered_pred: int,
    l2_value: float,
    linf_value: float,
    failure_reason: str,
    aware_adv: np.ndarray | None,
    aware_filtered: np.ndarray | None,
    internal: Dict[str, Any],
) -> None:
    adv_min, adv_max = _range_bounds(aware_adv)
    filtered_min, filtered_max = _range_bounds(aware_filtered)
    if success:
        logger.info(
            "Defense-aware success sample %d/%d | sample_index=%d true_label=%d "
            "clean_pred=%d adv_pred_external=%d filtered_adv_pred_external=%d "
            "l2=%.6f linf=%.6f adv=[%.6f,%.6f] filtered=[%.6f,%.6f] "
            "internal_adv_pred=%s internal_filtered_adv_pred=%s internal_l2=%s "
            "internal_external_match=%s",
            count,
            limit,
            sample_index,
            true_label,
            clean_pred,
            adv_pred,
            filtered_pred,
            l2_value,
            linf_value,
            adv_min,
            adv_max,
            filtered_min,
            filtered_max,
            internal.get("internal_adv_pred", ""),
            internal.get("internal_filtered_adv_pred", ""),
            internal.get("internal_l2", ""),
            _internal_external_match(internal, adv_pred, filtered_pred),
        )
        return

    logger.info(
        "Defense-aware failure sample %d/%d | sample_index=%d true_label=%d "
        "clean_pred=%d adv_pred_external=%d filtered_adv_pred_external=%d "
        "l2=%.6f linf=%.6f failure_reason=%s adv=[%.6f,%.6f] "
        "filtered=[%.6f,%.6f] internal_adv_pred=%s "
        "internal_filtered_adv_pred=%s internal_l2=%s internal_external_match=%s",
        count,
        limit,
        sample_index,
        true_label,
        clean_pred,
        adv_pred,
        filtered_pred,
        l2_value,
        linf_value,
        failure_reason,
        adv_min,
        adv_max,
        filtered_min,
        filtered_max,
        internal.get("internal_adv_pred", ""),
        internal.get("internal_filtered_adv_pred", ""),
        internal.get("internal_l2", ""),
        _internal_external_match(internal, adv_pred, filtered_pred),
    )


def evaluate_defense_aware_arrays(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    predict_fn: PredictFn,
    defense_unaware_attack_fn: AttackFn,
    defense_aware_attack_fn: AttackFn,
    transform_fn: TransformFn,
    sample_indices: np.ndarray | None = None,
    evaluation_config: Dict[str, Any] | None = None,
    reproduction_config: Dict[str, Any] | None = None,
    sample_diagnostics: list[Dict[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    """Evaluate defense-unaware and defense-aware scenarios over arrays."""
    image_array = np.asarray(images, dtype=np.float32)
    label_ints = label_to_int(np.asarray(labels))
    if sample_indices is None:
        sample_indices = np.arange(len(image_array), dtype=np.int64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    eval_config = dict(evaluation_config or {})
    reproduction = dict(reproduction_config or {})
    log_progress_every = int(eval_config.get("log_progress_every", 25))
    log_success_limit = int(eval_config.get("log_success_samples", 0))
    log_failure_limit = int(eval_config.get("log_failure_samples", 0))

    total_valid = 0
    skipped_clean = 0
    unaware_counts = ScenarioCounts()
    aware_counts = ScenarioCounts()
    success_logs = 0
    failure_logs = 0
    started = time.perf_counter()
    last_started = started
    total_scanned = len(image_array)

    for index, clean_image in enumerate(image_array):
        sample_started = time.perf_counter()
        scanned = index + 1
        sample_index = int(sample_indices[index])
        true_label = int(label_ints[index])
        clean_pred = _predict_one(predict_fn, clean_image)
        if clean_pred != true_label:
            skipped_clean += 1
            last_started = sample_started
            _log_progress_if_needed(
                scanned=scanned,
                total_scanned=total_scanned,
                total_valid=total_valid,
                skipped_clean=skipped_clean,
                unaware_counts=unaware_counts,
                aware_counts=aware_counts,
                started=started,
                last_image_seconds=time.perf_counter() - sample_started,
                log_progress_every=log_progress_every,
            )
            continue

        total_valid += 1
        unaware_adv_pred = -1
        unaware_success = False
        unaware_l2 = 0.0

        unaware_adv = np.asarray(
            defense_unaware_attack_fn(
                clean_image,
                true_label,
                clean_pred,
                sample_index,
                None,
            ),
            dtype=np.float32,
        )
        if _valid_adversarial_output(unaware_adv, clean_image.shape):
            unaware_adv = unaware_adv.reshape(clean_image.shape)
            unaware_adv_pred = _predict_one(predict_fn, unaware_adv)
            unaware_success = unaware_adv_pred != true_label
            if unaware_success:
                unaware_counts.success += 1
                unaware_l2 = _l2_distance(unaware_adv, clean_image)
                unaware_counts.l2_distances.append(unaware_l2)
                unaware_transformed = _transform_one(transform_fn, unaware_adv)
                unaware_transformed_pred = _predict_one(predict_fn, unaware_transformed)
                if unaware_transformed_pred != unaware_adv_pred:
                    unaware_counts.detected += 1
                else:
                    unaware_counts.undetected += 1
            else:
                unaware_counts.failures += 1
        else:
            unaware_counts.failures += 1

        internal_diagnostics: list[Dict[str, Any]] = []
        aware_adv_raw = np.asarray(
            defense_aware_attack_fn(
                clean_image,
                true_label,
                clean_pred,
                sample_index,
                internal_diagnostics,
            ),
            dtype=np.float32,
        )
        valid_aware = _valid_adversarial_output(aware_adv_raw, clean_image.shape)
        aware_adv: np.ndarray | None = None
        aware_filtered: np.ndarray | None = None
        aware_adv_pred = -1
        aware_filtered_pred = -1
        aware_l2 = 0.0
        aware_linf = 0.0
        if valid_aware:
            aware_adv = aware_adv_raw.reshape(clean_image.shape)
            aware_adv_pred = _predict_one(predict_fn, aware_adv)
            aware_filtered = _transform_one(transform_fn, aware_adv)
            aware_filtered_pred = _predict_one(predict_fn, aware_filtered)
            if aware_adv_pred != true_label:
                aware_l2 = _l2_distance(aware_adv, clean_image)
                aware_linf = _linf_distance(aware_adv, clean_image)

        failure_reason = _failure_reason(
            valid_output=valid_aware,
            true_label=true_label,
            adv_pred=aware_adv_pred,
            filtered_adv_pred=aware_filtered_pred,
        )
        aware_success = failure_reason == ""
        if aware_success:
            aware_counts.success += 1
            aware_counts.undetected += 1
            aware_counts.l2_distances.append(aware_l2)
        else:
            aware_counts.failures += 1
            if failure_reason == "detected":
                aware_counts.detected += 1

        elapsed = time.perf_counter() - sample_started
        internal = _latest_internal_diagnostic(internal_diagnostics)
        _append_sample_diagnostic(
            sample_diagnostics=sample_diagnostics,
            sample_index=sample_index,
            true_label=true_label,
            clean_pred=clean_pred,
            unaware_adv_pred=unaware_adv_pred,
            unaware_success=unaware_success,
            unaware_l2=unaware_l2,
            aware_adv_pred=aware_adv_pred,
            aware_filtered_adv_pred=aware_filtered_pred,
            aware_success=aware_success,
            failure_reason=failure_reason,
            aware_l2=aware_l2,
            aware_linf=aware_linf,
            aware_adv=aware_adv,
            aware_filtered=aware_filtered,
            internal=internal,
            elapsed_seconds=elapsed,
            reproduction_config=reproduction,
        )

        if aware_success and success_logs < log_success_limit:
            success_logs += 1
            _log_aware_sample(
                success=True,
                count=success_logs,
                limit=log_success_limit,
                sample_index=sample_index,
                true_label=true_label,
                clean_pred=clean_pred,
                adv_pred=aware_adv_pred,
                filtered_pred=aware_filtered_pred,
                l2_value=aware_l2,
                linf_value=aware_linf,
                failure_reason=failure_reason,
                aware_adv=aware_adv,
                aware_filtered=aware_filtered,
                internal=internal,
            )
        if not aware_success and failure_logs < log_failure_limit:
            failure_logs += 1
            _log_aware_sample(
                success=False,
                count=failure_logs,
                limit=log_failure_limit,
                sample_index=sample_index,
                true_label=true_label,
                clean_pred=clean_pred,
                adv_pred=aware_adv_pred,
                filtered_pred=aware_filtered_pred,
                l2_value=aware_l2,
                linf_value=aware_linf,
                failure_reason=failure_reason,
                aware_adv=aware_adv,
                aware_filtered=aware_filtered,
                internal=internal,
            )

        last_started = sample_started
        _log_progress_if_needed(
            scanned=scanned,
            total_scanned=total_scanned,
            total_valid=total_valid,
            skipped_clean=skipped_clean,
            unaware_counts=unaware_counts,
            aware_counts=aware_counts,
            started=started,
            last_image_seconds=time.perf_counter() - last_started,
            log_progress_every=log_progress_every,
        )

    rows = [
        _row_for_counts(
            attack="defense_unaware",
            total_valid=total_valid,
            counts=unaware_counts,
        ),
        _row_for_counts(
            attack="defense_aware",
            total_valid=total_valid,
            counts=aware_counts,
            article_reference=ARTICLE_AWARE_SUCCESS_RATE_PERCENT,
        ),
    ]
    _log_final_summary(
        scanned=total_scanned,
        total_valid=total_valid,
        skipped_clean=skipped_clean,
        unaware_counts=unaware_counts,
        aware_counts=aware_counts,
    )
    return rows


def _log_progress_if_needed(
    *,
    scanned: int,
    total_scanned: int,
    total_valid: int,
    skipped_clean: int,
    unaware_counts: ScenarioCounts,
    aware_counts: ScenarioCounts,
    started: float,
    last_image_seconds: float,
    log_progress_every: int,
) -> None:
    if log_progress_every <= 0:
        return
    if scanned % log_progress_every != 0 and scanned != total_scanned:
        return
    elapsed = time.perf_counter() - started
    avg_scanned = elapsed / float(scanned) if scanned else 0.0
    avg_valid = elapsed / float(total_valid) if total_valid else 0.0
    eta = _format_seconds(avg_scanned * float(total_scanned - scanned))
    aware_rate = _percent(aware_counts.success, total_valid)
    logger.info(
        "Defense-aware progress %d/%d scanned | valid_attacked=%d skipped_clean=%d | "
        "unaware_success=%d unaware_failures=%d | aware_success=%d "
        "aware_failures=%d aware_success_rate=%.2f%% | last_image=%.1fs "
        "avg_per_scanned=%.1fs avg_per_valid=%.1fs eta=%s",
        scanned,
        total_scanned,
        total_valid,
        skipped_clean,
        unaware_counts.success,
        unaware_counts.failures,
        aware_counts.success,
        aware_counts.failures,
        aware_rate,
        last_image_seconds,
        avg_scanned,
        avg_valid,
        eta,
    )


def _log_final_summary(
    *,
    scanned: int,
    total_valid: int,
    skipped_clean: int,
    unaware_counts: ScenarioCounts,
    aware_counts: ScenarioCounts,
) -> None:
    unaware_rate = _percent(unaware_counts.success, total_valid)
    aware_rate = _percent(aware_counts.success, total_valid)
    aware_failure_rate = _percent(aware_counts.failures, total_valid)
    unaware_mean_l2 = (
        float(np.mean(unaware_counts.l2_distances)) if unaware_counts.l2_distances else 0.0
    )
    aware_mean_l2 = (
        float(np.mean(aware_counts.l2_distances)) if aware_counts.l2_distances else 0.0
    )
    logger.info(
        "Defense-aware complete | total_scanned=%d total_valid=%d skipped_clean=%d",
        scanned,
        total_valid,
        skipped_clean,
    )
    logger.info(
        "Defense-unaware summary | success=%d failures=%d success_rate=%.2f%% "
        "mean_l2=%.6f",
        unaware_counts.success,
        unaware_counts.failures,
        unaware_rate,
        unaware_mean_l2,
    )
    logger.info(
        "Defense-aware summary | success=%d failures=%d success_rate=%.2f%% "
        "failure_rate=%.2f%% mean_l2=%.6f",
        aware_counts.success,
        aware_counts.failures,
        aware_rate,
        aware_failure_rate,
        aware_mean_l2,
    )
    logger.info(
        "Defense-aware reference | article_success_rate=67.37%% "
        "article_failure_rate=32.63%% delta=%.2fpp",
        aware_rate - ARTICLE_AWARE_SUCCESS_RATE_PERCENT,
    )


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


def _dataset_bounds(config: Dict[str, Any]) -> tuple[int, int]:
    dataset_config = config.get("dataset", {})
    evaluation_config = config.get("evaluation", {})
    start = int(dataset_config.get("start", 9000))
    end = int(dataset_config.get("end", 10000))
    sample_override = evaluation_config.get("n_samples", dataset_config.get("samples"))
    if sample_override not in (None, "", "all"):
        end = min(end, start + int(sample_override))
    return start, end


def _load_images(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_config = config.get("dataset", {})
    if str(dataset_config.get("name", "")).lower() != "mnist":
        raise ValueError("defense_aware requires dataset.name=mnist.")
    start, end = _dataset_bounds(config)
    images, labels = load_mnist_test_slice(start, end)
    sample_indices = np.arange(start, end, dtype=np.int64)
    return images, labels, sample_indices


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


def _log_experiment_start(config: Dict[str, Any]) -> None:
    dataset_config = dict(config.get("dataset", {}))
    model_config = dict(config.get("model", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    reproduction_config = dict(config.get("reproduction", {}))
    attacks_config = dict(config.get("attacks", {}))
    detector_config = dict(config.get("detector", {}))
    unaware_config = dict(attacks_config.get("defense_unaware", {}))
    aware_config = dict(attacks_config.get("defense_aware", {}))
    start, end = _dataset_bounds(config)
    samples = end - start

    logger.info(
        "Defense-aware experiment start | dataset=%s start=%d end=%d samples=%d model=%s",
        dataset_config.get("name", ""),
        start,
        end,
        samples,
        model_config.get("name", ""),
    )
    logger.info(
        "Defense-aware reproduction mode | mode=%s use_internal_filter=%s "
        "use_project_transform_fn=%s return_space=%s",
        reproduction_config.get("mode", aware_config.get("reproduction_mode", "")),
        reproduction_config.get("use_internal_filter", aware_config.get("use_internal_filter")),
        reproduction_config.get(
            "use_project_transform_fn",
            aware_config.get("use_project_transform_fn"),
        ),
        reproduction_config.get("return_space", ""),
    )
    logger.info(
        "Defense-aware attacks | unaware=%s aware=%s",
        _attack_type(unaware_config),
        _attack_type(aware_config),
    )
    logger.info(
        "Defense-aware CW params | max_iterations=%s binary_search_steps=%s "
        "initial_const=%s learning_rate=%s confidence=%s targeted=%s "
        "batch_size=%s box=[%s,%s]",
        aware_config.get("max_iterations", ""),
        aware_config.get("binary_search_steps", ""),
        aware_config.get("initial_const", ""),
        aware_config.get("learning_rate", ""),
        aware_config.get("confidence", ""),
        aware_config.get("targeted", ""),
        aware_config.get("batch_size", evaluation_config.get("batch_size", "")),
        aware_config.get("boxmin", -0.5),
        aware_config.get("boxmax", 0.5),
    )
    logger.info(
        "Defense-aware scale contract | pipeline=[0,1] attack=[-0.5,0.5] "
        "model_input_shift=%s",
        aware_config.get("model_input_shift", 0.5),
    )
    logger.info(
        "Defense-aware detector config | entropy_low=%s entropy_medium=%s "
        "q_low=%s q_medium=%s q_high=%s spatial_radius=%s",
        detector_config.get("entropy_thresholds", {}).get("low"),
        detector_config.get("entropy_thresholds", {}).get("medium"),
        detector_config.get("quantization", {}).get("low_entropy_step"),
        detector_config.get("quantization", {}).get("medium_entropy_step"),
        detector_config.get("quantization", {}).get("high_entropy_step"),
        detector_config.get("spatial_filter", {}).get("radius"),
    )
    logger.info(
        "Defense-aware dependency | nn_robust_attacks_root=%s",
        aware_config.get(
            "nn_robust_attacks_root",
            unaware_config.get("nn_robust_attacks_root", ""),
        ),
    )


def save_defense_aware_outputs(
    *,
    rows: list[Dict[str, Any]],
    output_dir: Path,
    sample_diagnostics: list[Dict[str, Any]] | None = None,
    write_sample_diagnostics: bool = False,
    csv_name: str = "metrics.csv",
    json_name: str = "metrics.json",
) -> Dict[str, Path]:
    """Write official defense-aware output artifacts."""
    output_path = ensure_dir(output_dir)
    csv_path = write_metrics_csv(output_path / csv_name, rows, DEFENSE_AWARE_SCHEMA)
    json_path = write_metrics_json(output_path / json_name, rows_to_metrics_json(rows))
    outputs = {"csv": csv_path, "json": json_path}
    if write_sample_diagnostics and sample_diagnostics is not None:
        outputs["sample_diagnostics"] = write_metrics_csv(
            output_path / "sample_diagnostics.csv",
            sample_diagnostics,
            SAMPLE_DIAGNOSTIC_SCHEMA,
        )
    return outputs


def run_defense_aware_evaluation(
    config: Dict[str, Any],
    graph: Dict[str, Any] | None = None,
    sample_diagnostics: list[Dict[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    """Run the configured defense-aware MNIST M2 evaluation."""
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    _log_experiment_start(config)
    images, labels, sample_indices = _load_images(config)
    active_graph = graph or create_restored_mnist_m2_graph(_checkpoint_dir(config))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 256))
    predict_fn = _predict_fn(active_graph, batch_size=batch_size)
    transform_fn = build_final_adaptive_detection_filter(config.get("detector", {}))

    attacks_config = config.get("attacks", {})
    reproduction_config = dict(config.get("reproduction", {}))
    rows = evaluate_defense_aware_arrays(
        images=images,
        labels=labels,
        sample_indices=sample_indices,
        predict_fn=predict_fn,
        defense_unaware_attack_fn=_build_attack_fn(
            graph=active_graph,
            attack_config=dict(attacks_config.get("defense_unaware", {})),
            predict_fn=predict_fn,
            transform_fn=transform_fn,
        ),
        defense_aware_attack_fn=_build_attack_fn(
            graph=active_graph,
            attack_config=dict(attacks_config.get("defense_aware", {})),
            transform_fn=transform_fn,
            predict_fn=predict_fn,
        ),
        transform_fn=transform_fn,
        evaluation_config=dict(config.get("evaluation", {})),
        reproduction_config=reproduction_config,
        sample_diagnostics=sample_diagnostics,
    )
    return rows


def run_defense_aware_experiment(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Run defense-aware evaluation and write official outputs."""
    write_sample_diagnostics = bool(
        config.get("evaluation", {}).get("write_sample_diagnostics", False)
    )
    sample_diagnostics: list[Dict[str, Any]] | None = (
        [] if write_sample_diagnostics else None
    )
    rows = run_defense_aware_evaluation(
        config,
        sample_diagnostics=sample_diagnostics,
    )
    output_config = config.get("output", {})
    configured_dir = output_config.get("dir") or config.get("output_dir")
    output_dir = resolve_project_path(configured_dir)
    if output_dir is None:
        raise ValueError("defense_aware must define output.dir or output_dir.")
    save_defense_aware_outputs(
        rows=rows,
        output_dir=output_dir,
        sample_diagnostics=sample_diagnostics,
        write_sample_diagnostics=write_sample_diagnostics,
        csv_name=str(output_config.get("csv", "metrics.csv")),
        json_name=str(output_config.get("json", "metrics.json")),
    )
    return rows
